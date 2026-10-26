"""Own one C14 GPU replay group; share A0 admission lock, check idle, and reap ranks."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

from contracts import plain
from media import now, read_json, require, sha, write_new_json
from replay_common import load_window, tolerance
from sr_only import (ID, check_run, cleanup_engine, gpu_state, members, proc, same_process, write_state)

HERE = Path(__file__).resolve().parent
DEADLINE = 7200


def replay_command(phase, window, output, model, logs, policy=None):
    extra = ["--tolerance-policy", str(policy)] if policy else []
    if phase == "color":
        return [sys.executable, str(HERE / "replay_color.py"), "--window", str(window),
                "--output", str(output), "--device", "cuda:0", *extra]
    if phase != "decode":
        raise ValueError("Unsupported GPU replay phase")
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=8",
            "--max-restarts=0", "--monitor-interval=1", "--log-dir", str(logs), "--redirects", "3",
            str(HERE / "replay_decode.py"), "--window", str(window), "--output", str(output),
            "--vae-weights", str(Path(model) / "ema_vae_fp16.safetensors"), *extra]


def environment(directory, nonce):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("SEEDVR2_C14_", "VLLM_OMNI_SEEDVR2_", "DIFFUSION_CACHE_", "TORCHELASTIC_")) or key in (
            "PYTHONPATH", "PYTHONHOME", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"
        ):
            env.pop(key, None)
    for key, suffix in {"TMPDIR": "tmp", "TEMP": "tmp", "TMP": "tmp", "HF_HOME": "cache/hf", "TORCH_HOME": "cache/torch",
                        "XDG_CACHE_HOME": "cache/xdg", "TRITON_CACHE_DIR": "cache/triton", "VLLM_CACHE_ROOT": "cache/vllm"}.items():
        target = directory / suffix
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONHASHSEED="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
               DINGO_ENABLE_MINIMAX_H3_PATCHES="0", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
               SEEDVR2_C14_REPLAY_GUARD=str(directory / "guard.json"), SEEDVR2_C14_REPLAY_NONCE=nonce)
    return env


def cleanup_replay(directory, child, birth):
    """Reap launcher plus attested per-rank sessions; never kill by program name."""
    failures = []
    try:
        cleanup_engine(child, birth)
    except BaseException as error:
        failures.append(f"launcher cleanup: {error}")
    guard_path = directory / "guard.json"
    if guard_path.exists():
        guard = read_json(guard_path)
        records = []
        for path in sorted((directory / "rank-owners").glob("rank-*.json")):
            item = read_json(plain(path))
            require(item.get("schema") == "seedvr2-c14-replay-rank/1" and item.get("nonce") == guard["nonce"]
                    and item.get("phase") == guard["phase"], "Rank ownership evidence is mismatched")
            records.append(item["owner"])
        for owner in records:
            if not same_process(owner):
                continue
            try:
                require(owner["pgid"] == owner["pid"], "Live rank shares an unconfirmed session; manual investigation required")
                os.killpg(owner["pgid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
            except BaseException as error:
                failures.append(f"rank{owner['pid']} cleanup: {error}")
        deadline = time.monotonic() + 10
        while any(same_process(owner) for owner in records) and time.monotonic() < deadline:
            time.sleep(.25)
        for owner in records:
            if same_process(owner) and owner["pgid"] == owner["pid"]:
                os.killpg(owner["pgid"], signal.SIGKILL)
        deadline = time.monotonic() + 5
        while any(same_process(owner) for owner in records) and time.monotonic() < deadline:
            time.sleep(.25)
        for owner in records:
            if same_process(owner) or members(owner["pgid"]):
                failures.append(f"rank session{owner['pgid']} still exists")
    require(not failures, "; ".join(failures))


def launch(args):
    import fcntl
    require(sys.platform == "linux", "GPU guard runs inside the diagnostic Linux image")
    require(ID.fullmatch(args.id), "Use a new lowercase replay ID")
    run = plain(args.run, directory=True)
    manifest = check_run(run)
    require(read_json(run / "state.json")["status"] == "A0_CAPTURE_COMPLETE_QUALITY_UNDETERMINED",
            "GPU replay requires settled successful complete A0")
    window = plain(run / "capture" / run.name / "windows" / f"{args.window:06d}", directory=True)
    captured = load_window(window)
    policy = plain(args.tolerance_policy) if args.tolerance_policy else None
    tolerance(captured, policy)
    directory = run.parent / "replays" / (run.name + "-" + args.id)
    directory.parent.mkdir(exist_ok=True)
    directory.mkdir(exist_ok=False)
    child = birth = None
    previous = {}
    lock = (run.parent / ".c14-gpu.lock").open("a+")
    def stopped(signum, frame):
        raise InterruptedError("Operator requested stop of this replay")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_new_json(directory / "gpu-before.json", gpu_state(idle=True))
        owner = proc(os.getpid())
        require(owner is not None, "Cannot identify replay supervisor")
        nonce = secrets.token_hex(32)
        output = directory / "result"
        guard = {"schema": "seedvr2-c14-replay-guard/1", "phase": args.phase, "window": str(window),
                 "output": str(output), "parent_run": str(run), "run_id": args.id, "supervisor": owner,
                 "nonce": nonce, "parent_manifest_sha256": sha(run / "manifest.json"), "at": now()}
        write_new_json(directory / "guard.json", guard)
        (directory / "guard.json").chmod(0o444)
        argv = replay_command(args.phase, window, output, manifest["model"], directory / "rank-logs", policy)
        write_new_json(directory / "launch.json", {"argv": argv, "gpu_admission": "8 idle full H100s; same lock as A0",
                                                  "timeout_seconds": DEADLINE, "h3_started": False})
        write_state(directory / "state.json", {"status": "STARTING", "at": now()})
        previous = {sig: signal.signal(sig, stopped) for sig in (signal.SIGTERM, signal.SIGINT)}
        with (directory / "worker.log").open("xb") as log:
            pending = []
            for sig in previous:
                signal.signal(sig, lambda signum, frame: pending.append(signum))
            child = subprocess.Popen(argv, env=environment(directory, nonce), stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            birth = proc(child.pid)
            require(birth is not None and birth["pgid"] == child.pid, "Cannot establish replay process-group ownership")
            write_new_json(directory / "worker-owner.json", birth)
            for sig in previous:
                signal.signal(sig, stopped)
            if pending:
                raise InterruptedError("Stopped while starting replay")
            write_state(directory / "state.json", {"status": "RUNNING", "at": now()})
            print("C14 replay started:", directory, flush=True)
            deadline = time.monotonic() + DEADLINE
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("GPU replay exceeded2h; whole owned rank group will be stopped")
                time.sleep(1)
            require(child.returncode == 0, "GPU replay failed; see worker.log/rank-logs. No automatic retry")
        cleanup_replay(directory, child, birth)
        child = None
        write_new_json(directory / "gpu-after.json", gpu_state(idle=True))
        expected = 8 if args.phase == "decode" else 1
        require(sorted(p.name for p in (directory / "rank-owners").glob("rank-*.json")) ==
                [f"rank-{rank}.json" for rank in range(expected)], "Not every replay rank registered process ownership")
        require((output / "COMPLETE.json").is_file() and not list(output.glob("FAILED*.json")),
                "Replay child did not leave complete success evidence")
        write_state(directory / "state.json", {"status": "PASS_REPLAY_ONLY_QUALITY_PENDING", "at": now()})
        print("PASS guarded replay; quality review pending:", output, flush=True)
    except BaseException as error:
        write_state(directory / "state.json", {"status": "FAILED_EVIDENCE_PRESERVED", "at": now(), "error": str(error)})
        raise
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            cleanup_replay(directory, child, birth)
            if child is not None:
                write_new_json(directory / "gpu-after-failure.json", gpu_state(idle=True))
        except BaseException as error:
            write_state(directory / "state.json", {"status": "STOP_UNCONFIRMED_DO_NOT_START_NEXT", "at": now(), "error": str(error)})
            raise
        finally:
            lock.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def stop(directory):
    directory = plain(directory, directory=True)
    guard = read_json(plain(directory / "guard.json"))
    owner = guard["supervisor"]
    require(same_process(owner), "Guard supervisor is not alive; do not guess a PID")
    cmdline = (Path("/proc") / str(owner["pid"]) / "cmdline").read_bytes().split(b"\0")
    require(b"run" in cmdline and guard["run_id"].encode() in cmdline and guard["parent_run"].encode() in cmdline
            and any(item.endswith(b"gpu_guard.py") for item in cmdline), "PID is not this replay supervisor")
    os.kill(owner["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 75
    while same_process(owner) and time.monotonic() < deadline:
        time.sleep(1)
    require(not same_process(owner), "Replay supervisor has not exited; do not start another run")
    if (directory / "worker-owner.json").exists():
        child = read_json(directory / "worker-owner.json")
        require(not members(child["pgid"]), "Owned replay rank processes remain")
    for path in (directory / "rank-owners").glob("rank-*.json"):
        rank = read_json(path)["owner"]
        require(not same_process(rank) and not members(rank["pgid"]), "Attested replay rank session remains")
    print("STOPPED_OWN_REPLAY", gpu_state(idle=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    start = subs.add_parser("run")
    start.add_argument("--run", type=Path, required=True)
    start.add_argument("--phase", choices=("color", "decode"), required=True)
    start.add_argument("--window", type=int, choices=(130, 143, 156, 169, 182), required=True)
    start.add_argument("--id", required=True)
    start.add_argument("--tolerance-policy", type=Path, help="Only accepted when its SHA was frozen in the A0 profile")
    for action in ("status", "stop"):
        sub = subs.add_parser(action)
        sub.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        launch(args)
    elif args.action == "status":
        import json
        print(json.dumps(read_json(plain(args.replay, directory=True) / "state.json"), indent=2))
    else:
        stop(args.replay)


if __name__ == "__main__":
    main()
