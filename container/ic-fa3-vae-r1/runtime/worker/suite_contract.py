"""Plan validation, persisted measurements and restart-safe scheduling."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import uuid


PROFILES = {"stock", "stream", "owner", "batch2", "combo"}
STAGES = {"source_decode", "vae_encode", "qwen_encode", "latent_upscale", "ic_denoise", "vae_decode", "media_delivery"}
RELEASE = "ic-fa3-vae-r1"
FA3_REVISION = "c0a7c25685c2339ebc71a2a2d824b02f93cc725b"
FA3_BINARY = "3e7e7178a8d0c67e197101b4a0d52983a4891d0e4b970316582f4d37ddfb1bac"
HASH = re.compile(r"[0-9a-f]{64}")
IDENT = re.compile(r"[a-z0-9][a-z0-9_-]{0,95}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return Path(path).read_text(encoding="utf-8-sig")


def document(path):
    return json.loads(read(path))


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def is_absolute_media_path(value):
    return isinstance(value, str) and PurePosixPath(value).is_absolute() and ".." not in PurePosixPath(value).parts


def validate(plan):
    require(plan.get("release_id") == RELEASE, "Unexpected release_id")
    require(IDENT.fullmatch(plan["run_id"]), "Invalid run_id")
    for key in ("model_root", "audio_vae_path"):
        require(is_absolute_media_path(plan[key]), f"Absolute Pod path required for {key}")
    rows = plan["requests"]
    require(isinstance(rows, list) and rows, "Empty requests list")
    ids, warmups, samples = set(), set(), set()
    for r in rows:
        for key in ("id", "case_id"):
            require(IDENT.fullmatch(r[key]), f"Invalid {key}")
        require(r["id"] not in ids and r["id"] not in {"boots", "previous", "recovery"}, "Duplicate/reserved request ID")
        ids.add(r["id"])
        require(r["tier"] in {"1k", "2k", "4k"} and r["profile"] in PROFILES, "Unknown tier/profile")
        for key in ("source", "prompt_path"):
            require(is_absolute_media_path(r[key]), f"Absolute Pod path required for {key}")
        for key in ("source_sha256", "prompt_sha256"):
            require(HASH.fullmatch(r[key]), f"Invalid {key}")
        for key in ("frames", "width", "height", "repeat", "seed"):
            require(type(r[key]) is int and r[key] >= (1 if key in ("frames", "width", "height") else 0), f"Invalid {key}")
        require(r["height"] == {"1k":1080, "2k":1440, "4k":2160}[r["tier"]], "Output height differs from tier")
        require(r.get("audio_policy", "preserve") in {"preserve", "required", "none"}, "Unknown audio policy")
        key = (r["tier"], r["profile"])
        if r["kind"] == "warmup":
            require(r["repeat"] == 0 and key not in warmups, "Duplicate warmup or unexpected warmup repeat")
            warmups.add(key)
        else:
            require(r["kind"] == "measured" and r["repeat"] > 0 and key in warmups, "Measurement needs preceding tier/profile warmup")
            sample = (*key, r["case_id"], r["repeat"])
            require(sample not in samples, "Duplicate measured repeat")
            samples.add(sample)
    return plan


def verify_fa3(row, calls_per_block):
    require(row.get("backend") == "FLASH_ATTN_3_HUB" and row.get("repo") == "kernels-community/flash-attn3"
            and row.get("revision") == FA3_REVISION and row.get("binary_sha256") == FA3_BINARY,
            "Effective FA3 operator identity differs from release")
    counts = row.get("block_calls")
    require(row.get("num_blocks") == 50 and isinstance(counts, list) and len(counts) == 50 and
            all(type(v) is int and v == calls_per_block for v in counts) and row.get("calls") == sum(counts),
            "Unexpected FA3 calls per DiT block")


def verified_result(root, row, plan_sha):
    record = root / row["id"] / "measurement.json"
    if not record.exists():
        return None
    m = document(record)
    require(m.get("status") == "PASS" and m.get("request") == row and m.get("plan_sha256") == plan_sha,
            f"Completed receipt for {row['id']} differs from plan")
    path = record.parent / "output.mp4"
    require(path.is_file() and path.stat().st_size == m["size_bytes"] and digest(path) == m["output_sha256"],
            f"Completed video integrity check failed: {row['id']}")
    return m


def schedule(rows, completed):
    """Yield (row, is_recovery_warmup). Never reuse a warmup from another process."""
    warmups, ready, work = {}, set(), []
    for row in rows:
        key = (row["tier"], row["profile"])
        if row["kind"] == "warmup":
            warmups[key] = row
        if row["id"] in completed:
            continue
        if row["kind"] == "measured" and key not in ready:
            work.append((warmups[key], True))
        work.append((row, False))
        ready.add(key)
    return work


def validate_ranks(rows, startup):
    require(len(rows) == 8 and [r["rank"] for r in rows] == list(range(8)), "Incomplete rank evidence")
    for row, boot in zip(rows, startup):
        require(row["pid"] == boot["pid"] and row["gpu_uuid"] == boot["gpu_uuid"] and row["worker_id"] == boot["worker_id"],
                "Resident worker changed during request")
        require(set(row["stage_seconds"]) == STAGES and all(type(v) in (float, int) and math.isfinite(v) and v >= 0
                for v in row["stage_seconds"].values()) and math.isfinite(row["pipeline_seconds"]) and row["pipeline_seconds"] > 0,
                "Incomplete/invalid native stage samples")
        verify_fa3(row["fa3"], 3)
        require(all(row["tile_work"][k] > 0 for k in ("encode", "decode")), "Inactive VAE rank")


def make_report(completed, total, epoch):
    groups = defaultdict(list)
    for m in completed:
        r = m["request"]
        if r["kind"] == "measured":
            groups[(r["tier"], r["profile"], r["case_id"])].append(m)
    summaries = []
    for (tier, profile, case_id), ms in groups.items():
        timings = [m["native_pipeline_seconds_max_rank"] for m in ms]
        summaries.append({"tier": tier, "profile": profile, "case_id": case_id, "n": len(ms),
                          "native_median_seconds": statistics.median(timings), "native_min_seconds": min(timings),
                          "native_max_seconds": max(timings), "request_ids": [m["request"]["id"] for m in ms]})
    return {"worker_epoch": epoch, "completed": len(completed), "planned": total, "groups": summaries,
            "updated_utc": utc()}


def persist_report(root, completed, total, epoch):
    report = make_report(completed, total, epoch)
    save(root / "report.json", report)
    lines = ["# IC R1 benchmark", "", f"Completed {len(completed)}/{total}, including planned warmups.", "",
             "| Tier | Profile | Case | Measured runs | Native median (s) | Min (s) | Max (s) |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for r in report["groups"]:
        lines.append(f"| {r['tier']} | {r['profile']} | {r['case_id']} | {r['n']} | {r['native_median_seconds']:.2f} | "
                     f"{r['native_min_seconds']:.2f} | {r['native_max_seconds']:.2f} |")
    write_text(root / "report.md", "\n".join(lines) + "\n")
