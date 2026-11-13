"""OpenH3-IR payload and result contract; no prompt rewriting on the Gateway."""
import hashlib
import subprocess
import re
from pathlib import Path
from dingo.common.h3_reference_order import group_references
from .client import UpstreamRejected


def prepare(normalized, task_root, attempt_root, derived_limit):
    assets, upload_files, expected = [], {}, []
    picture = video = audio = 0
    references = normalized["references"]
    groups = group_references(references, lambda entry: entry["kind"])
    derived = attempt_root / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    derived_bytes = 0
    for kind in ("image", "video", "audio"):
        for entry in groups[kind]:
            path = task_root / entry["path"]
            if path.is_symlink() or not path.is_file():
                raise ValueError("compiler input is not a regular owned file")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024*1024):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                raise ValueError("compiler input checksum changed")
            sha = entry["sha256"]
            upload_files[sha] = path
            item = {"sha256": sha, "kind": kind}
            # Preserve omission: a default must not become an explicit upstream instruction.
            # Older persisted requests lack this flag and retain their original semantics.
            if entry.get("role_stated", True):
                item["role"] = entry["role"]
            if normalized.get("workflow") == "ref2va" and kind == "image":
                # The selected checkpoint fixes content-reference versus frame-anchor semantics.
                # Keep the role omitted while preventing the upstream mode classifier from
                # moving a Ref2VA request onto I2VA/FL2VA based on ambiguous prose.
                item["provenance"] = {"intended_role": "reference"}
            if entry.get("note"):
                item["note"] = entry["note"]
            if entry.get("duration_s"):
                item["seconds"] = entry["duration_s"]
            labels = []
            paired_audio_sha = None
            if kind == "image":
                picture += 1
                labels = [f"<Picture {picture}>"]
            elif kind == "video":
                video += 1
                if entry["has_audio"]:
                    wav = derived / f"video-{entry['ordinal']:03d}-audio.wav"
                    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
                                    "-map", "0:a:0", "-vn", "-acodec", "pcm_s16le", "-ac", "2",
                                    "-ar", "32000", "-t", "15", "-fs", str(derived_limit + 1), str(wav)],
                                   check=True, capture_output=True, timeout=60)
                    derived_bytes += wav.stat().st_size
                    if derived_bytes > derived_limit:
                        raise ValueError("derived compiler audio exceeds byte budget")
                    wav_sha = hashlib.sha256(wav.read_bytes()).hexdigest()
                    paired_audio_sha = wav_sha
                    upload_files[wav_sha] = wav
                    audio += 1
                    labels.append(f"<Audio {audio}>")
                    assets.append({"sha256": wav_sha, "kind": "audio", "role": "bgm",
                                   "paired_video_sha256": sha,
                                   "soundtrack_policy": entry["soundtrack_policy"]})
                labels.append(f"<Video {video}>")
            else:
                audio += 1
                labels = [f"<Audio {audio}>"]
            assets.append(item)
            expected.append({"index": entry["ordinal"], "kind": kind, "sha256": sha,
                             "role": entry["role"], "role_stated": entry.get("role_stated", True), "labels": labels,
                             "paired_audio_sha256": paired_audio_sha,
                             "soundtrack_policy": entry.get("soundtrack_policy")})
    w, h = normalized["canvas"]
    payload = {"intent": normalized["intent"], "assets": assets,
               "seconds": normalized["seconds"], "aspect": f"{w}:{h}",
               "megapixels": w*h/1_000_000, "creativity": normalized["creativity"],
               "seed": normalized["context_ir_seed"], "dialogue": normalized["dialogue"],
               "silent": normalized["silent"]}
    return payload, upload_files, sorted(expected, key=lambda e:e["index"])


def validate_result(normalized, expected, status, response):
    if status not in {200, 201} or not isinstance(response, dict):
        raise UpstreamRejected("context_ir_upstream_rejected")
    degraded = response.get("status") == "degraded" and response.get("source") == "draft"
    if not degraded and (response.get("status") != "ready" or response.get("source") != "written"
                         or response.get("fallback_reason")):
        raise UpstreamRejected("context_ir_degraded_result")
    ir = response.get("ir", {})
    target = ir.get("target", {})
    if (ir.get("mode") != normalized["mode"] or target.get("frames") != normalized["frames"]
            or target.get("canvas") != normalized["canvas"]):
        raise UpstreamRejected("context_ir_contract_mismatch")
    prompt = ir.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise UpstreamRejected("context_ir_contract_mismatch")
    manifests = ir.get("manifest", [])
    actual = {e.get("label"): e for e in manifests}
    labels = [label for e in expected for label in e["labels"]]
    if len(actual) != len(manifests) or set(actual) != set(labels):
        raise UpstreamRejected("context_ir_contract_mismatch")
    for e in expected:
        main = actual[e["labels"][-1]]
        allowed_roles = {e["role"]}
        if not e.get("role_stated", True):
            allowed_roles = {"image": {"subject"}, "video": {"edit_source", "motion_reference"}}.get(e["kind"], allowed_roles)
        if main.get("sha256") != e["sha256"] or main.get("role") not in allowed_roles:
            raise UpstreamRejected("context_ir_contract_mismatch")
        if e.get("paired_audio_sha256"):
            audio = actual[e["labels"][0]]
            if audio.get("sha256") != e["paired_audio_sha256"] or audio.get("paired_with") != e["labels"][-1]:
                raise UpstreamRejected("context_ir_contract_mismatch")
            marker = re.search(r"^\s*"+re.escape(e["labels"][0])+r"(?:\s*\([^\n]*?\))?\s*:\s*(\w+)", prompt, re.M)
            allowed = {"weak_reference"} if e["soundtrack_policy"] == "ignore" else {"partially_copy", "fully_copy"}
            if marker is None or marker.group(1) not in allowed:
                raise UpstreamRejected("context_ir_soundtrack_contract_mismatch")
    # Return the actual fallback draft, never provenance.rejected_written.
    # Degraded results retain the same public schema and all contract checks.
    warnings = [{"code": "context_ir_degraded_result"}] if degraded else []
    for e in normalized["references"]:
        if e["kind"] == "video" and (e.get("duration_s") or 0) > normalized["frames"]/24:
            warnings.append({"code": "reference_video_truncated", "index": e["ordinal"]})
    return {"content": {"prompt": prompt}, "generation_parameters": normalized["generation_parameters"],
            "target": {"frames": normalized["frames"], "fps": 24},
            "references": [{**{k:v for k,v in e.items() if k in {"index", "kind", "sha256", "labels"}},
                            "role": actual[e["labels"][-1]]["role"]} for e in expected],
            "warnings": warnings}


def response_usage(response):
    """Accept only self-consistent aggregate usage; old backends may omit it."""
    if not isinstance(response, dict):
        return None
    detail = response.get("detail")
    value = response.get("usage")
    if value is None and isinstance(detail, dict):
        value = detail.get("usage")
    keys = ("prompt_tokens", "completion_tokens", "total_tokens", "llm_requests",
            "requests_with_usage", "requests_without_usage")
    if not isinstance(value, dict) or any(type(value.get(k)) is not int or not 0 <= value[k] <= 2**63-1 for k in keys):
        return None
    if (value["total_tokens"] != value["prompt_tokens"] + value["completion_tokens"]
            or value["llm_requests"] != value["requests_with_usage"] + value["requests_without_usage"]
            or type(value.get("complete")) is not bool
            or value["complete"] != (value["requests_without_usage"] == 0)
            or (value["requests_with_usage"] == 0 and value["total_tokens"] != 0)):
        return None
    result = {k: value[k] for k in (*keys, "complete")}
    if "prompt_tokens_details" in value:
        details = value["prompt_tokens_details"]
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if not value["complete"] or type(cached) is not int or not 0 <= cached <= value["prompt_tokens"]:
            cached = None
        result["prompt_tokens_details"] = {"cached_tokens": cached}
    return result
