"""Run one IC suite through eight persistent native workers in the packaged Linux image."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))

from native_worker import NativeWorker, release_info
from r1_media import media_probe, verify_delivery_video
from suite_contract import (STAGES, digest, document, persist_report, require, save, schedule,
                            utc, validate, validate_ranks, verified_result)


def preflight(rows):
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    require(ffprobe and ffmpeg, "Both ffprobe and ffmpeg are required")
    signatures, sources = {}, {}
    for row in rows:
        for label, key in (("source","source_sha256"), ("prompt_path","prompt_sha256")):
            path = row[label]
            if path not in signatures:
                signatures[path] = digest(path)
            require(signatures[path] == row[key], f"Input hash differs: {path}")
        source = row["source"]
        if source not in sources:
            output = subprocess.check_output([ffprobe, "-v", "error", "-show_streams", "-of", "json", source], timeout=120)
            streams = json.loads(output)["streams"]
            video = [s for s in streams if s["codec_type"] == "video"]
            audio = [s for s in streams if s["codec_type"] == "audio"]
            require(len(video) == 1 and len(audio) <= 1, "Expected one video and at most one audio stream")
            v = video[0]
            require(Fraction(v["avg_frame_rate"]) == Fraction(v["r_frame_rate"]) == 24, "Source is not 24 fps")
            sources[source] = {"width":v["width"], "height":v["height"], "frames":int(v["nb_frames"]),
                               "audio_policy":"required" if audio else "none", "source_sha256":signatures[source]}
        observed = sources[source]
        require(observed["frames"] == row["frames"] and row.get("audio_policy","preserve") in ("preserve",observed["audio_policy"]),
                "Source and request frame/audio contract differ")
    return {"sources": sources, "ffprobe":ffprobe, "ffmpeg":ffmpeg}


def validate_media(directory, row, observed, data):
    output = directory / "output.mp4"
    with output.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    report = directory / "media"
    report.mkdir()
    settings = {"tools":{name:{"path":observed[name]} for name in ("ffprobe","ffmpeg")},
                "command_timeout_seconds":1800, "tolerances":{"timestamp_seconds":0.001, "audio_end_seconds":0.08}}
    selector = {"num_frames":row["frames"], "audio_policy":observed["sources"][row["source"]]["audio_policy"]}
    media = media_probe(output, selector, settings, report, expected_size=(row["width"],row["height"]))
    media.update(sha256=digest(output), bytes=output.stat().st_size)
    save(report / "media.json", media)
    codec = verify_delivery_video(output)
    save(report / "codec.json", codec)
    return media, codec


def bind(root, plan, plan_bytes, release):
    root.mkdir(exist_ok=True, parents=True)
    copied = root / "plan.json"
    if copied.exists():
        require(copied.read_bytes() == plan_bytes, "Output directory is bound to a different plan")
    else:
        temporary = root / (".plan-" + uuid.uuid4().hex + ".tmp")
        temporary.write_bytes(plan_bytes)
        os.replace(temporary, copied)
    identity = root / "release.json"
    if identity.exists():
        require(document(identity) == release, "Output directory is bound to a different image release")
    else:
        save(identity, release)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    completed = {}
    for row in plan["requests"]:
        old = verified_result(root, row, plan_sha)
        if old is not None:
            completed[row["id"]] = old
    return plan_sha, completed


def take_lock(root):
    import fcntl
    root.mkdir(exist_ok=True, parents=True)
    handle = (root / ".suite.lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("Another suite driver owns this output directory")
    return handle


def request_directory(root, row, epoch, recovery):
    base = root / "boots" / epoch / "recovery" if recovery else root
    directory = base / row["id"]
    if directory.exists():
        require(not (directory / "measurement.json").exists(), "Refusing to overwrite a committed response")
        retired = root / "previous" / row["id"] / uuid.uuid4().hex
        retired.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(retired)
    directory.mkdir(parents=True)
    return directory


def run(plan_path, root):
    rank = int(os.environ["RANK"])
    require(os.environ.get("WORLD_SIZE") == "8" and int(os.environ["LOCAL_RANK"]) == rank and 0 <= rank < 8,
            "Use single-node torchrun with eight ranks")
    require(root.is_absolute(), "--out must be absolute")
    plan_bytes = plan_path.read_bytes()
    plan = validate(json.loads(plan_bytes.decode("utf-8-sig")))
    release = release_info(plan)
    token = os.environ["TORCHELASTIC_RUN_ID"]
    require(token and token not in (".","..") and Path(token).name == token, "Invalid torchrun rendezvous identity")
    # Torchrun --standalone allocates a fresh rendezvous id on each invocation.
    epoch = token + "-" + str(int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")))
    boot = root / "boots" / epoch
    native = lock = directory = None
    completed = {}
    owns = bound = False
    phase = "setup"
    began = time.perf_counter()
    try:
        if rank == 0:
            lock = take_lock(root)
            owns = True
            plan_sha, completed = bind(root, plan, plan_bytes, release)
            require(not (boot / "started.json").exists(), "Torchrun rendezvous ID was reused")
            bound = True
            boot.mkdir(exist_ok=True, parents=True)
            marker = {"status":"RUNNING","phase":"LOADING","worker_epoch":epoch,"plan_sha256":plan_sha,
                      "pod_uid":os.environ.get("POD_UID"),"pod":os.environ.get("HOSTNAME"),"started_utc":utc()}
            save(boot / "started.json", marker)
            save(root / "progress.json", marker)
            # Supersede a prior failure marker only after validating this plan and release.
            save(root / "terminal.json", {"status":"RUNNING", "worker_epoch":epoch})
            observed = preflight(plan["requests"])
            tasks = schedule(plan["requests"], completed)
            input_seconds = time.perf_counter()-began
        phase = "loading"
        native = NativeWorker(plan, rank)
        startup_rows = native.gather(native.startup)
        def ready():
            require([x["rank"] for x in startup_rows] == list(range(8)) and
                    all(len({x[f] for x in startup_rows}) == 8 for f in ("gpu_uuid","pid","worker_id")),
                    "Rank membership differs from eight unique resident GPU processes")
            ready_utc = utc()
            info = {**marker, "status":"READY", "phase":"LOADED", "release":release, "ranks":startup_rows,
                    "input_preflight_seconds":input_seconds, "owner_launch_to_ready_seconds":time.perf_counter()-began,
                    "native_load_seconds_max_rank":max(x["native_load_seconds"] for x in startup_rows),
                    "model_loading_started_utc":min(x["native_load_started_utc"] for x in startup_rows),
                    "model_ready_utc":ready_utc, "ready_utc":ready_utc,
                    "model_root":plan["model_root"], "audio_vae_path":plan["audio_vae_path"]}
            save(boot / "startup.json", info)
            save(root / "startup.json", info)
            persist_report(root,completed.values(),len(plan["requests"]),epoch)
            return tasks
        tasks = native.owner(ready)
        for row, recovery in tasks:
            phase = row["id"]
            def start_trial():
                path = request_directory(root, row, epoch, recovery)
                info = {"status":"RUNNING","phase":phase, "id":row["id"],"kind":row["kind"], "recovery_warmup":recovery,
                        "worker_epoch":epoch, "started_utc":utc(), "plan_sha256":plan_sha}
                save(path / "request.json", {**info, "request":row})
                save(root / "progress.json", info)
                print(f"START {phase}" + (" (recovery warmup)" if recovery else ""), flush=True)
                return str(path)
            directory = Path(native.owner(start_trial))
            start = time.perf_counter()
            local, payload = native.generate(plan["run_id"], row)
            rows = native.gather(local)
            def finish_trial():
                validate_ranks(rows, startup_rows)
                media, codec = validate_media(directory,row,observed,payload)
                max_rank = max(rows, key=lambda r:r["pipeline_seconds"])
                m = {"schema":"ic-r1-measurement/1","status":"PASS","request":row,"worker_epoch":epoch,
                     "release":release,"plan_sha256":plan_sha,"recovery_warmup":recovery,
                     "dispatch_to_validated_seconds":time.perf_counter()-start,
                     "native_pipeline_seconds_max_rank":max_rank["pipeline_seconds"],"slowest_rank":max_rank["rank"],
                     "stage_seconds":dict(max_rank["stage_seconds"]),
                     "stage_seconds_max_per_rank":{s:max(r["stage_seconds"][s] for r in rows) for s in sorted(STAGES)},
                     "source":observed["sources"][row["source"]], "ranks":rows,"media":media,"codec":codec,
                     "output_sha256":media["sha256"],"size_bytes":media["bytes"],"finished_utc":utc()}
                save(directory / "measurement.json", m)
                if not recovery:
                    completed[row["id"]] = m
                    persist_report(root,completed.values(),len(plan["requests"]),epoch)
                print(f"PASS {phase} {m['native_pipeline_seconds_max_rank']:.2f} s", flush=True)
            native.owner(finish_trial)
            del payload
            directory = None
        phase = "shutdown"
        native.close()
        native = None
        if rank == 0:
            terminal = {"status":"PASS","worker_epoch":epoch,"completed":len(completed),
                        "planned":len(plan["requests"]), "finished_utc":utc(), "plan_sha256":plan_sha}
            require(terminal["completed"] == terminal["planned"], "Missing planned measurements")
            save(boot / "terminal.json", terminal)
            save(root / "terminal.json", terminal)
            save(root / "progress.json", terminal)
    except BaseException as error:
        failure = {"status":"FAILED", "worker_epoch":epoch,"phase":phase, "rank":rank,"error":str(error),
                   "traceback":traceback.format_exc(),"observed_utc":utc()}
        # A second process that failed to obtain the lock cannot change the original run.
        if rank != 0 or (owns and bound):
            save(boot / f"failed-rank-{rank}.json", failure)
        if owns and bound:
            if directory and directory.exists() and not (directory / "measurement.json").exists():
                save(directory / "failure.json",failure)
            save(boot / "terminal.json",failure)
            save(root / "terminal.json",failure)
            save(root / "progress.json",failure)
        raise
    finally:
        if lock is not None:
            lock.close()
        # On rank failure let torchrun terminate peers. Calling distributed teardown
        # here can deadlock peers that still await a collective inside the model.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate the plan schema locally, without loading models")
    args = parser.parse_args()
    if args.check:
        plan = validate(document(args.plan))
        print(f"PASS_PLAN {len(plan['requests'])} requests")
    else:
        if not args.out:
            parser.error("--out is required for execution")
        run(args.plan,args.out)


if __name__ == "__main__":
    main()
