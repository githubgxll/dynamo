# SPDX-License-Identifier: Apache-2.0
"""C14 SR-only operator CLI. No H3 generation, package installation or weight conversion."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request

from contracts import AUDIO, BASE, PARENT, PROFILE, REVISION, plain, validate_baseline, validate_media, validate_model, validate_release
from media import (SR_ROUTE, command, compare_audio, curl_request, get_state, now, probe_media,
                   read_json, require, sha, write_new_json)

HERE = Path(__file__).resolve().parent
PORT = 8124
TIMEOUT = 10800
ID = re.compile(r"[a-z0-9][a-z0-9-]{0,47}\Z")


def write_state(path, value):
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def proc(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        return {"pid": int(pid), "state": fields[0], "pgid": int(fields[2]), "start_ticks": int(fields[19])}
    except (OSError, IndexError, ValueError):
        return None


def same_process(expected):
    current = proc(expected["pid"])
    return bool(current and current["state"] != "Z" and all(current[k] == expected[k] for k in ("pid", "pgid", "start_ticks")))


def members(pgid):
    return [record for p in Path("/proc").iterdir() if p.name.isdigit()
            if (record := proc(int(p.name))) and record["pgid"] == pgid and record["state"] != "Z"]


def gpu_state(*, idle):
    data = command(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used,power.limit", "--format=csv,noheader,nounits"], 30)
    rows = [list(map(str.strip, row)) for row in csv.reader(data.splitlines())]
    require(len(rows) == 8 and len({row[0] for row in rows}) == 8, "Exactly eight visible GPU UUIDs required")
    require(all(len(row) == 5 and "H100" in row[1] and "MIG" not in row[1] and
                70 * 1024 <= float(row[2]) <= 85 * 1024 for row in rows), "Expected full H100 80GB GPUs")
    apps = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], 30)
    active = [row for row in csv.reader(apps.splitlines()) if row and row[0].strip() in {x[0] for x in rows}]
    require(not idle or not active, "GPUs have active compute processes; no competing SR/H3 run allowed")
    return {"at": now(), "gpus": rows, "active_compute": active, "scope": "Read-only device enumeration; no resource or power settings changed"}


def identity(release_path):
    require(sys.platform == "linux", "Run the CLI inside the published Linux diagnostic image")
    from runtime import verify_installed
    proof = verify_installed()
    release = read_json(plain(release_path))
    image = validate_release(release, os.environ.get("H3_C14_SOURCE_COMMIT"))
    return {"deployment_image": image, "source_commit": os.environ["H3_C14_SOURCE_COMMIT"],
            "installed_source": proof, "release_sha256": sha(release_path)}


def prerequisites(baseline, model, release):
    proof = identity(release)
    for name in ("vllm-omni", "ffmpeg", "ffprobe", "curl", "nvidia-smi"):
        require(shutil.which(name), "Missing inherited executable: " + name)
    frozen, weights = validate_baseline(baseline), validate_model(model)
    return {"status": "PASS_PREFLIGHT_ONLY", "at": now(), "identity": proof, "profile": PROFILE,
            "baseline_sha256": sha(Path(baseline) / "manifest.json"),
            "model_manifest_sha256": sha(Path(model) / "conversion-manifest.json"),
            "model_outputs": weights["outputs"], "parent_base_sha256": frozen["parent_base_sha256"],
            "gpu": gpu_state(idle=True), "h3_started": False, "inference_run": False}


def import_base(args):
    require(ID.fullmatch(args.id), "Use a unique lowercase experiment ID")
    root = plain(args.root, directory=True)
    source = plain(args.baseline, directory=True)
    model = plain(args.model, directory=True)
    require(not root.is_relative_to(source) and not root.is_relative_to(model), "Output must be separate from readonly inputs")
    proof = prerequisites(source, model, args.release_evidence)
    require(shutil.disk_usage(root).free >= 51 * 1024**3, "Reserve at least 51 GiB before importing for diagnostic tensors")
    run = root / args.id
    run.mkdir(exist_ok=False)
    write_state(run / "state.json", {"status": "IMPORTING", "at": now()})
    try:
        with (source / "base768.mp4").open("rb") as src, (run / "base.mp4").open("xb") as dest:
            shutil.copyfileobj(src, dest, 8 * 1024**2)
        require(sha(run / "base.mp4") == BASE == sha(source / "base768.mp4"), "Base changed while importing")
        (run / "base.mp4").chmod(0o444)
        input_proof = probe_media(run / "base.mp4", run, "import-base")
        validate_media(input_proof, [1280, 768])
        require(input_proof["audio"]["packet_proof"]["sequence_sha256"] == AUDIO, "Original audio differs")
        write_new_json(run / "preflight.json", proof)
        write_new_json(run / "parent-baseline.json", read_json(source / "manifest.json"))
        write_new_json(run / "release.json", read_json(args.release_evidence))
        weights = proof["model_outputs"]
        capture = {"schema": "seedvr2-c14-profile/1", "run_id": args.id, "output_root": str(run / "capture"),
            "parent_base_sha256": BASE, "model_revision": REVISION,
            "model_sha256": {name: row["sha256"] for name, row in weights.items()},
            "capture_window_starts": PROFILE["capture_window_starts"], "max_bytes": 50 * 1024**3,
            "max_tensor_bytes": 1024**3, "min_free_bytes": 1024**3, "expected_world_size": 8,
            "parent_image": PARENT, "deployment_image": proof["identity"]["deployment_image"],
            "source_commit": proof["identity"]["source_commit"],
            "native_source_hashes": proof["identity"]["installed_source"]["seedvr2_source_sha256"]}
        write_new_json(run / "capture-profile.json", capture)
        write_new_json(run / "manifest.json", {"schema": "seedvr2-c14-run/1", "run_id": args.id, "at": now(),
            "profile": PROFILE, "capture_profile_sha256": sha(run / "capture-profile.json"),
            "model": str(model), "model_manifest_sha256": proof["model_manifest_sha256"],
            "model_outputs": weights, "base_sha256": BASE, "parent_run": "h3sr8-ref7-fix-a",
            "parent_job": "ref7-rest-1022-a-c14", "parent_request_sha256": read_json(source / "manifest.json")["parent_request_contract_sha256"],
            "identity": proof["identity"], "input_media": input_proof,
            "h3_resident": False, "quality_status": "NOT_ASSESSED"})
        for name in ("manifest.json", "capture-profile.json", "parent-baseline.json", "release.json"):
            (run / name).chmod(0o444)
        write_state(run / "state.json", {"status": "IMPORTED_VERIFIED_NO_INFERENCE", "at": now()})
    except BaseException as error:
        write_state(run / "state.json", {"status": "IMPORT_FAILED_DO_NOT_USE", "at": now(), "error": str(error)})
        raise
    print("PASS_IMPORTED_SR_ONLY", run, flush=True)


def check_run(run, *, model=True):
    run = plain(run, directory=True)
    manifest = read_json(plain(run / "manifest.json"))
    require(manifest.get("schema") == "seedvr2-c14-run/1" and manifest.get("run_id") == run.name, "Wrong run identity")
    require(manifest.get("profile") == PROFILE, "Fixed C14 profile changed")
    require(sha(plain(run / "capture-profile.json")) == manifest["capture_profile_sha256"], "Capture profile changed")
    require(sha(plain(run / "base.mp4")) == BASE == manifest["base_sha256"], "Imported Base changed")
    if model:
        ident = identity(run / "release.json")
        require(ident == manifest["identity"], "Runtime image/source changed since import")
        weights = validate_model(manifest["model"])
        require(weights["outputs"] == manifest["model_outputs"] and
                sha(Path(manifest["model"]) / "conversion-manifest.json") == manifest["model_manifest_sha256"], "FP16 model changed")
    return manifest


def service_command(model):
    return ["vllm-omni", "serve", model, "--omni", "--model-class-name", "SeedVR2Pipeline", "--dtype", "float16",
        "--enforce-eager", "--cache-backend", "none", "--num-gpus", "8", "--distributed-executor-backend", "mp", "--vae-use-tiling",
        "--stage-overrides", json.dumps({"0": {"ulysses_degree": 8, "vae_patch_parallel_size": 8, "vae_parallel_mode": "spatial_shard_height"}}, separators=(",", ":")),
        "--api-server-count", "1", "--tensor-parallel-size", "1", "--ring-degree", "1", "--cfg-parallel-size", "1", "--allgather-degree", "1",
        "--text-encoder-tp-size", "1", "--ulysses-mode", "strict", "--no-ulysses-a2a-permute", "--max-num-seqs", "1", "--request-batch-max-wait-ms", "0",
        "--host", "127.0.0.1", "--port", str(PORT), "--init-timeout", str(TIMEOUT), "--stage-init-timeout", str(TIMEOUT)]


def service_environment(run):
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(("DIFFUSION_CACHE_", "VLLM_OMNI_SEEDVR2_", "SEEDVR2_C14_")) or name in (
                "PYTHONPATH", "PYTHONHOME", "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(name, None)
    # Native long-video restores POST each window back to this same server.
    # Keep cluster bypasses while guaranteeing loopback stays inside the Pod.
    bypass = [part.strip() for key in ("NO_PROXY", "no_proxy")
              for part in env.get(key, "").split(",") if part.strip()]
    env["NO_PROXY"] = env["no_proxy"] = ",".join(dict.fromkeys([*bypass, "localhost", "127.0.0.1", "::1"]))
    for key, suffix in {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "HF_HOME": "cache/hf", "XDG_CACHE_HOME": "cache/xdg",
                       "TORCH_HOME": "cache/torch", "TRITON_CACHE_DIR": "cache/triton", "VLLM_CACHE_ROOT": "cache/vllm"}.items():
        folder = run / "attempt" / suffix
        folder.mkdir(parents=True, exist_ok=True)
        env[key] = str(folder)
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONHASHSEED="0", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1", VLLM_WORKER_MULTIPROC_METHOD="spawn", DINGO_ENABLE_MINIMAX_H3_PATCHES="0",
        OMNI_DIFFUSION_PROMPT_EMBED_CACHE="0", VLLM_OMNI_VAE_LEGACY_TEMPORAL="0", VLLM_OMNI_VAE_ENCODE_LEGACY_PREP="0",
        VLLM_OMNI_VIDEO_SYNC_TIMEOUT=str(TIMEOUT), VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=str(TIMEOUT), VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=str(TIMEOUT),
        NCCL_DEBUG="WARN", MASTER_ADDR="127.0.0.1", MASTER_PORT="29610", TZ="Asia/Shanghai",
        VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS="3791872", VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS="64461824",
        VLLM_OMNI_SEEDVR2_MAX_FRAMES="17", VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW="17",
        VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR=str(run / "attempt/native"), VLLM_OMNI_SEEDVR2_LONG_JOB_TTL_SECONDS="604800",
        SEEDVR2_C14_PROFILE=str(run / "capture-profile.json"))
    return env


def wait_healthy(process):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        require(process.poll() is None, "SR process exited during load; see engine.log")
        try:
            with opener.open(f"http://127.0.0.1:{PORT}/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError("SR load timeout; no automatic retry")


def finish_group(process, birth):
    # Called by the owning live supervisor only; group lifetime is continuously held.
    require(birth and birth["pgid"] == process.pid, "Missing process-group ownership")
    group = process.pid
    if process.poll() is None:
        require(same_process(birth), "Owned engine PID was reused; refusing signal")
    if members(group):
        os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + 45
    while members(group) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.5)
    if members(group):
        os.killpg(group, signal.SIGKILL)
    process.wait(timeout=15)
    require(not members(group), "Engine ranks remain alive; do not run next experiment")


def cleanup_engine(engine, birth):
    if engine is None:
        return
    if birth is not None:
        finish_group(engine, birth)
        return
    # Popen can succeed but the child may exit before /proc identity is sampled.
    # No ownership fingerprint means no signal based on a guessed process ID.
    require(engine.poll() is not None and not members(engine.pid),
            "Engine started but ownership could not be captured; cleanup UNCONFIRMED")
    engine.wait(timeout=15)


def mux_command(raw, base, final):
    return ["ffmpeg", "-nostdin", "-n", "-v", "error", "-xerror", "-copyts", "-noautorotate", "-i", str(raw),
        "-noautorotate", "-i", str(base), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy",
        "-vf", "scale=1800:1080:flags=lanczos,setsar=1,pad=1800:1080:0:0:color=black", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
        "-avoid_negative_ts", "disabled", "-map_metadata", "1", "-movflags", "+faststart", str(final)]


def capture_run(run):
    import fcntl
    run = plain(run, directory=True)
    manifest = check_run(run)
    require(read_json(run / "state.json")["status"] == "IMPORTED_VERIFIED_NO_INFERENCE", "Capture is create-once; use status/collect, never retry a POST")
    attempt = run / "attempt"
    attempt.mkdir(exist_ok=False)
    engine = birth = None
    previous = {}
    lock = (run.parent / ".c14-gpu.lock").open("a+")
    def stopped(signum, frame):
        raise InterruptedError("Operator requested stop of this run")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_new_json(attempt / "gpu-before.json", gpu_state(idle=True))
        require(shutil.disk_usage(run).free >= 50 * 1024**3 + 1024**3, "Insufficient disk for bounded diagnostic capture")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", PORT))
        write_new_json(attempt / "supervisor.json", proc(os.getpid()))
        previous = {s: signal.signal(s, stopped) for s in (signal.SIGTERM, signal.SIGINT)}
        argv, env = service_command(manifest["model"]), service_environment(run)
        write_new_json(attempt / "launch.json", {"argv": argv, "profile": PROFILE,
            "environment": {k: v for k, v in env.items() if k.startswith(("VLLM_OMNI_SEEDVR2_", "SEEDVR2_C14_"))}, "h3_started": False})
        write_state(run / "state.json", {"status": "LOADING_SR", "at": now()})
        with (attempt / "engine.log").open("xb") as log:
            # Defer termination until ownership has been recorded. Do not mask signals in exec'd workers.
            pending = []
            for sig in previous:
                signal.signal(sig, lambda signum, frame: pending.append(signum))
            engine = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            birth = proc(engine.pid)
            require(birth and birth["pgid"] == engine.pid, "Cannot establish engine ownership")
            write_new_json(attempt / "engine-owner.json", birth)
            for sig in previous:
                signal.signal(sig, stopped)
            if pending:
                raise InterruptedError("Stopped during engine launch")
            wait_healthy(engine)
            write_state(run / "state.json", {"status": "CAPTURING_A0", "at": now()})
            fields = {"prompt": " ", "size": "1840x1104", "num_frames": 362, "loop_input": "false", "seed": 7723,
                      "extra_params": json.dumps({"color_correction_method": "lab"}, separators=(",", ":"))}
            write_new_json(attempt / "wire-request.json", {"fields": fields, "base_sha256": BASE})
            curl_request(PORT, SR_ROUTE, attempt / "submission.json", fields, [(run / "base.mp4", "video/mp4")],
                         expected_status=202, content_type="application/json")
            job = read_json(attempt / "submission.json")
            require(re.fullmatch(r"[0-9a-f]{32}", str(job.get("id", ""))), "Invalid native job ID")
            route = SR_ROUTE + "/" + job["id"]
            deadline, last = time.monotonic() + TIMEOUT, None
            with (attempt / "progress.jsonl").open("x", encoding="utf-8") as progress:
                while time.monotonic() < deadline:
                    require(engine.poll() is None, "SR engine exited; no resubmission")
                    state = get_state(PORT, route, 30)
                    progress.write(json.dumps({"at": now(), **state}) + "\n")
                    progress.flush()
                    mark = (state.get("status"), state.get("frames"))
                    if mark != last:
                        print("C14", *mark, flush=True)
                        last = mark
                    if state.get("status") == "completed":
                        require(state.get("frames") == 362, "Native completion frame count differs")
                        break
                    require(state.get("status") in ("queued", "running"), "Native SR failed: " + str(state))
                    time.sleep(2)
                else:
                    raise TimeoutError("Native job timeout; no resubmission")
            curl_request(PORT, route + "/content", attempt / "raw1104.mp4", content_type="video/mp4")
        finish_group(engine, birth)
        engine = None
        write_new_json(attempt / "gpu-after.json", gpu_state(idle=True))
        raw = probe_media(attempt / "raw1104.mp4", attempt, "raw1104")
        validate_media(raw, [1840, 1104])
        argv = mux_command(attempt / "raw1104.mp4", run / "base.mp4", attempt / "final1080.mp4")
        write_new_json(attempt / "final-encoding-command.json", {"argv": argv, "raw_sha256": raw["sha256"], "base_sha256": BASE})
        command(argv)
        final = probe_media(attempt / "final1080.mp4", attempt, "final1080")
        validate_media(final, [1800, 1080])
        audio = compare_audio(manifest["input_media"]["audio"], final["audio"])
        from vllm_omni.diffusion.models.seedvr2.c14_capture import verify_run
        captures = verify_run(run / "capture" / run.name)
        write_new_json(attempt / "result.json", {"status": "PASS_CAPTURE_AND_MEDIA_ONLY", "raw": raw, "final": final,
            "audio": audio, "captures": captures, "human_quality_review": "REQUIRED", "c14_fixed": "UNDETERMINED"})
        write_state(run / "state.json", {"status": "A0_CAPTURE_COMPLETE_QUALITY_UNDETERMINED", "at": now()})
    except BaseException as error:
        write_state(run / "state.json", {"status": "FAILED_EVIDENCE_PRESERVED", "at": now(), "error": str(error)})
        raise
    finally:
        # Ignore repeated stop requests while reaping the owned rank group.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            cleanup_engine(engine, birth)
            # A failed request is not proof that CUDA contexts are gone. Also
            # detect workers which might have created their own process group.
            # Do not signal an unowned PID; block collection/next run instead.
            if birth is not None or engine is not None:
                write_state(attempt / "gpu-after-cleanup.json", gpu_state(idle=True))
            write_state(attempt / "cleanup.json", {"status": "OWNED_RANKS_STOPPED", "at": now(),
                "engine_started": birth is not None or engine is not None, "pgid": birth["pgid"] if birth else None})
        except Exception as error:
            write_state(attempt / "cleanup.json", {"status": "UNCONFIRMED", "at": now(), "error": str(error)})
            write_state(run / "state.json", {"status": "STOP_UNCONFIRMED_DO_NOT_START_NEXT", "at": now(), "error": str(error)})
            raise
        finally:
            lock.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def stop_run(run):
    run = plain(run, directory=True)
    check_run(run, model=False)
    owner = read_json(run / "attempt/supervisor.json")
    require(same_process(owner), "Original supervisor is not alive; refuse PID guessing")
    cmdline = (Path("/proc") / str(owner["pid"]) / "cmdline").read_bytes().split(b"\0")
    require(b"capture" in cmdline and str(run).encode() in cmdline and any(x.endswith(b"sr_only.py") for x in cmdline),
            "Process is not this run's supervisor")
    os.kill(owner["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 75
    while same_process(owner) and time.monotonic() < deadline:
        time.sleep(1)
    require(not same_process(owner), "Supervisor has not exited; do not start another job")
    engine = read_json(run / "attempt/engine-owner.json")
    require(not members(engine["pgid"]), "Owned engine ranks remain; stop is unconfirmed")
    print(json.dumps({"status": "OWNED_RUN_STOPPED", "gpu": gpu_state(idle=True)}))


def collect(run, output):
    run = plain(run, directory=True)
    state = read_json(run / "state.json")
    require(state["status"] in ("A0_CAPTURE_COMPLETE_QUALITY_UNDETERMINED", "FAILED_EVIDENCE_PRESERVED", "IMPORT_FAILED_DO_NOT_USE"),
            "Collect only after the run has settled; status does not restart inference")
    if state["status"] != "IMPORT_FAILED_DO_NOT_USE":
        check_run(run, model=False)
    else:
        require(ID.fullmatch(run.name), "Invalid failed import directory identity")
    if (run / "attempt").exists():
        cleanup = read_json(plain(run / "attempt/cleanup.json"))
        require(cleanup.get("status") == "OWNED_RANKS_STOPPED", "Owned ranks have not been confirmed stopped")
        if cleanup.get("pgid") and sys.platform == "linux":
            require(not members(cleanup["pgid"]), "Engine process group is still alive")
    output = Path(output).absolute()
    require(not output.is_relative_to(run), "Archive must be outside evidence directory")
    entries = {}
    # Do not export model/temp cache or uploaded native API copies. Keep all explicit diagnostic tensors and logs.
    files = [p for p in run.rglob("*") if p.is_file() and
             not any(x in ("cache", "tmp", "native") for x in p.relative_to(run).parts)]
    for p in files:
        plain(p)
        entries[p.relative_to(run).as_posix()] = {"sha256": sha(p), "bytes": p.stat().st_size}
    require(shutil.disk_usage(output.parent).free >= sum(x["bytes"] for x in entries.values()) + 1024**3,
            "Not enough free disk for evidence TAR plus 1 GiB margin; export to another mounted volume")
    with output.open("xb") as handle, tarfile.open(fileobj=handle, mode="w|") as archive:
        for p in sorted(files):
            archive.add(p, arcname=run.name + "/" + p.relative_to(run).as_posix(), recursive=False)
            require(sha(p) == entries[p.relative_to(run).as_posix()]["sha256"], "Evidence changed during archive creation")
    write_new_json(output.with_suffix(output.suffix + ".json"), {"schema": "seedvr2-c14-export/1", "run": run.name,
        "archive_sha256": sha(output), "files": entries, "quality_status": "NOT_INFERRED_FROM_EXPORT"})
    print("PASS_COLLECTED", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for name in ("preflight", "import-base"):
        sub = subs.add_parser(name)
        sub.add_argument("--baseline", type=Path, required=True)
        sub.add_argument("--model", type=Path, required=True)
        sub.add_argument("--release-evidence", type=Path, required=True)
        if name == "import-base":
            sub.add_argument("--root", type=Path, required=True)
            sub.add_argument("--id", required=True)
    for name in ("capture", "status", "verify", "collect", "stop-own-run"):
        sub = subs.add_parser(name)
        sub.add_argument("--run", type=Path, required=True)
        if name == "collect":
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "preflight":
        proof = prerequisites(args.baseline, args.model, args.release_evidence)
        print(json.dumps({"status": proof["status"], "image": proof["identity"]["deployment_image"],
            "source_commit": proof["identity"]["source_commit"], "gpu_count": len(proof["gpu"]["gpus"]),
            "baseline": "362 frames / 24 FPS / 1280x768 verified", "model": "Existing verified SeedVR2-3B FP16",
            "h3_started": False, "inference_run": False}, indent=2))
    elif args.action == "import-base":
        import_base(args)
    elif args.action == "capture":
        capture_run(args.run)
    elif args.action == "stop-own-run":
        stop_run(args.run)
    elif args.action == "collect":
        collect(args.run, args.output)
    elif args.action == "status":
        print(json.dumps(read_json(plain(args.run, directory=True) / "state.json"), indent=2))
    elif args.action == "verify":
        check_run(args.run)
        from vllm_omni.diffusion.models.seedvr2.c14_capture import verify_run
        proof = verify_run(args.run / "capture" / args.run.name)
        print(json.dumps({"status": proof["status"], "windows": len(proof["windows"]),
            "scope": proof["scope"]}, indent=2))


if __name__ == "__main__":
    main()
