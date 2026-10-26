# SPDX-License-Identifier: Apache-2.0
"""Pinned media verification/loopback transport subset; no H3 generation or GPU calls.
Derived without behavior edits from the successful Ref7 resident_media.py.
"""
from __future__ import annotations
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request
TIMEOUT = 10800
SR_ROUTE = "/v1/seedvr2/restore-long"
REFERENCE_MIME_EXTENSIONS = {"video/mp4": (".mp4",)}

def require(value, message):
    if not value:
        raise ValueError(message)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

def read_json(path):
    require(Path(path).stat().st_size <= 1024 * 1024, "JSON evidence exceeds 1 MiB")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def _reparse(path):
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)

def inside(root, value, *, exists=True):
    root = Path(root).absolute()
    require(root.exists() and not _reparse(root), "Job root must exist and not be a symlink/reparse point")
    root = root.resolve()
    target = Path(value)
    if not target.is_absolute():
        target = root / target
    absolute = target.absolute()
    require(root == absolute or root in absolute.parents, "Path must stay inside the job")
    for part in (absolute, *absolute.parents):
        if part == root:
            break
        require(not part.exists() or not _reparse(part), "Symlink/reparse path refused")
    resolved = target.resolve()
    require(root == resolved or root in resolved.parents, "Path traversal refused")
    require(not exists or resolved.exists(), f"Missing path: {resolved}")
    return resolved

def run_owned(argv, *, timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True):
    """Reap only the helper PID we started, including on daemon signal exceptions."""
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, text=text)
    try:
        output, errors = process.communicate(timeout=timeout)
    except BaseException:
        process.kill()
        process.wait()
        raise
    return subprocess.CompletedProcess(argv, process.returncode, output, errors)

def command(argv, timeout=TIMEOUT):
    result = run_owned(argv, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} exited {result.returncode}: {result.stderr[-6000:]}")
    return result.stdout

def local_url(port, route):
    require(type(port) is int and 1024 <= port <= 65535, "Engine port must be an integer 1024..65535")
    require(route == "/v1/videos/sync" or route == SR_ROUTE
            or re.fullmatch(re.escape(SR_ROUTE) + r"/[0-9a-f]{32}(/content)?", route),
            "Only native local engine routes are permitted")
    return f"http://127.0.0.1:{port}{route}"

def _response_headers(raw):
    blocks = [chunk for chunk in re.split(r"\r?\n\r?\n", raw) if chunk.startswith("HTTP/")]
    require(bool(blocks), "Missing HTTP response headers")
    lines = blocks[-1].splitlines()
    status = int(lines[0].split()[1])
    headers = {k.strip().lower(): v.strip() for line in lines[1:] if ":" in line
               for k, v in [line.split(":", 1)]}
    return status, headers

def curl_request(port, route, output, fields=None, references=(), timeout=TIMEOUT,
                 expected_status=200, content_type=None):
    """Uploads/downloads stream through curl; redirects and proxies are disabled."""
    output = Path(output)
    header_path = output.with_name(output.name + ".headers.txt")
    evidence = output.with_name(output.name + ".http.json")
    require(not header_path.exists() and not evidence.exists(), "HTTP evidence already exists")
    url = local_url(port, route)
    require(0 < timeout <= TIMEOUT, "HTTP timeout outside bounded profile")
    argv = ["curl", "--noproxy", "*", "--proto", "=http", "--silent", "--show-error",
            "--fail-with-body", "--connect-timeout", "15", "--max-time", str(timeout),
            "--max-redirs", "0", "--dump-header", str(header_path), "--output", "-"]
    for key, value in (fields or {}).items():
        require(re.fullmatch(r"[a-z_]+", key), "Invalid form field name")
        argv += ["--form-string", f"{key}={value}"]
    for path, mime in references:
        require(not any(c in str(path) for c in '\r\n;"'), "Unsafe multipart file path")
        require(mime in REFERENCE_MIME_EXTENSIONS, "Unsupported multipart reference MIME")
        argv += ["-F", f'input_references=@"{path}";type={mime}']
    argv.append(url)
    started = time.monotonic()
    with output.open("xb") as target:
        result = run_owned(argv, stdout=target, stderr=subprocess.PIPE,
                           text=True, timeout=timeout + 15)
        target.flush()
        os.fsync(target.fileno())
    raw = header_path.read_text(encoding="iso-8859-1") if header_path.exists() else ""
    write_new_json(evidence, {"url": url, "returncode": result.returncode,
                             "stderr": result.stderr[-6000:], "at": now(),
                             "elapsed_seconds": time.monotonic() - started,
                             "body_sha256": sha(output), "body_bytes": output.stat().st_size})
    require(result.returncode == 0, "HTTP failed; body and headers preserved; no automatic retry")
    status, headers = _response_headers(raw)
    require(status == expected_status, f"Expected HTTP {expected_status}, got {status}; no automatic retry")
    if content_type:
        require(headers.get("content-type", "").split(";", 1)[0].strip().lower() == content_type,
                "Unexpected response content type")
    require(output.stat().st_size > 0, "HTTP returned an empty body")
    return {"status": status, "headers": headers}

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Engine HTTP redirects are forbidden")

def get_state(port, route, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(local_url(port, route), timeout=min(30, timeout)) as response:
        require(response.status == 200, "SR status HTTP response is not 200")
        payload = response.read(1024 * 1024 + 1)
    require(len(payload) <= 1024 * 1024, "SR status response too large")
    result = json.loads(payload)
    require(isinstance(result, dict), "SR status must be an object")
    return result

def audio_packet_proof(path):
    data = json.loads(command(["ffprobe", "-v", "error", "-select_streams", "a:0",
                               "-show_packets", "-show_data_hash", "sha256", "-of", "json", str(path)]))
    packets = data.get("packets", [])
    require(bool(packets), "Audio stream has no readable packets")
    normalized = []
    for packet in packets:
        digest = packet.get("data_hash", "")
        require(re.fullmatch(r"SHA256:[0-9a-fA-F]{64}", digest) is not None,
                "Missing audio packet SHA256")
        for key in ("pts_time", "dts_time", "duration_time"):
            require(key in packet, "Audio packet lacks timing")
        normalized.append({"hash": digest.lower(), "size": int(packet["size"]),
                           "pts": str(Fraction(packet["pts_time"])),
                           "dts": str(Fraction(packet["dts_time"])),
                           "duration": str(Fraction(packet["duration_time"])),
                           "side_data": packet.get("side_data_list", [])})
    return {"packet_count": len(packets), "packets": normalized,
            "sequence_sha256": canonical_hash(normalized)}

def probe_media(path, directory, label, allow_no_audio=False):
    path, directory = Path(path), Path(directory)
    require(path.is_file() and path.stat().st_size > 0, "Missing or empty video")
    info = json.loads(command(["ffprobe", "-v", "error", "-count_frames", "-show_streams",
                               "-show_format", "-of", "json", str(path)]))
    write_new_json(directory / f"{label}.ffprobe.json", info)
    videos = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    audios = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    require(len(videos) == 1, "Expected exactly one video stream")
    require(len(audios) == 1 or (allow_no_audio and not audios), "Expected original single audio stream")
    video = videos[0]
    require(Fraction(video["avg_frame_rate"]) == 24 and Fraction(video["r_frame_rate"]) == 24,
            "Video must be constant 24 FPS")
    require(abs(float(video.get("start_time", 0))) < 1e-6, "Video must start at PTS zero")
    require(video.get("sample_aspect_ratio") in (None, "1:1", "0:1"), "Non-square pixel video is unsupported")
    require(all(float(d.get("rotation", 0)) == 0 for d in video.get("side_data_list", [])),
            "Rotated video metadata is unsupported")
    frames = int(video["nb_read_frames"])
    require(1 <= frames <= 7200, "Actual video frame count outside 1..7200")
    timing = json.loads(command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                                 "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)]))
    timestamps = timing.get("frames", [])
    require(len(timestamps) == frames and all(abs(float(item["best_effort_timestamp_time"]) - i / 24) < 2e-5
                                              for i, item in enumerate(timestamps)),
            "Every video timestamp must match zero-origin 24 FPS CFR")
    write_new_json(directory / f"{label}.video-timestamps.json", timing)
    audio = None
    if audios:
        stream = audios[0]
        require(stream["codec_name"] == "aac" and stream.get("channels") in (1, 2),
                "Original AAC mono/stereo is required for bitstream preservation")
        audio = {"codec": stream["codec_name"], "sample_rate": int(stream["sample_rate"]),
                 "channels": stream["channels"], "channel_layout": stream.get("channel_layout"),
                 "start": float(stream.get("start_time", 0)), "duration": float(stream["duration"]),
                 "packet_proof": audio_packet_proof(path)}
        write_new_json(directory / f"{label}.audio-packets.json", audio)
    # Decode every stream to the null muxer, preserving errors as failures.
    command(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
             "-map", "0:v:0", "-map", "0:a:0?", "-f", "null", "-"])
    proof = {"width": int(video["width"]), "height": int(video["height"]), "frames": frames,
             "fps": 24, "video_seconds": frames / 24, "audio": audio,
             "sha256": sha(path), "bytes": path.stat().st_size, "full_decode": True,
             "human_quality_review": "REQUIRED"}
    write_new_json(directory / f"{label}.media-proof.json", proof)
    return proof

def compare_audio(source, final):
    require((source is None) == (final is None), "Original audio missing or unexpected audio added")
    if source is None:
        return {"status": "ORIGINAL_EXPLICITLY_NO_AUDIO"}
    for key in ("codec", "sample_rate", "channels", "channel_layout"):
        require(source[key] == final[key], f"Original audio {key} changed")
    for key in ("start", "duration"):
        require(abs(source[key] - final[key]) < 1e-6, f"Original audio {key} changed")
    left, right = source["packet_proof"]["packets"], final["packet_proof"]["packets"]
    require(len(left) == len(right), "Original audio packet count changed")
    for before, after in zip(left, right):
        for key in ("hash", "size", "side_data"):
            require(before[key] == after[key], f"Original audio packet {key} changed")
        for key in ("pts", "dts", "duration"):
            require(abs(Fraction(before[key]) - Fraction(after[key])) <= Fraction(1, 1000000),
                    f"Original audio packet {key} changed")
    return {"status": "ORIGINAL_AUDIO_PACKET_HASH_AND_TIMING_PASS",
            "packet_count": len(left), "source_sequence_sha256": source["packet_proof"]["sequence_sha256"],
            "final_sequence_sha256": final["packet_proof"]["sequence_sha256"]}
