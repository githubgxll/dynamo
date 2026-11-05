"""H3 compiler request semantics reuse the video adapter's media/shape checks."""
import hashlib
import json
from pathlib import Path
from dingo.video_gateway.errors import GatewayError
from dingo.video_gateway.adapters.h3_shape import align_frame_count
from dingo.video_gateway.task_runtime.transactions import encode

FIELDS = {"model", "prompt", "seconds", "num_frames", "size", "width", "height",
          "reference_options", "creativity", "context_ir_seed", "dialogue", "silent"}


def bad(code, message, field=None):
    return GatewayError(400, code, message, field)


def json_field(fields, name, default):
    try:
        return json.loads(fields[name][0]) if name in fields else default
    except (ValueError, TypeError) as exc:
        raise bad("invalid_json", f"{name} must be valid JSON", name) from exc


def audio_present(path):
    import av
    with av.open(str(path)) as container:
        streams = list(container.streams.audio)
        if not streams:
            return False
        return next(container.decode(streams[0]), None) is not None


def normalize(service, fields, uploads):
    unknown = set(fields) - FIELDS
    if unknown:
        raise bad("unsupported_parameter", "unsupported ContextIR parameters: " + ", ".join(sorted(unknown)))
    if any(len(v) != 1 for v in fields.values()):
        raise bad("duplicate_field", "ContextIR scalar fields may appear only once")
    if not fields.get("model") or not fields["model"][0].strip():
        raise bad("missing_required_field", "model is required", "model")
    model = fields["model"][0].strip()
    pool = service.resolve_pool(model)
    if model not in service.config.context_ir.model_bindings:
        raise bad("context_ir_not_configured", "ContextIR is not configured for this model", "model")
    if not fields.get("prompt") or not fields["prompt"][0].strip():
        raise bad("missing_required_field", "prompt must not be empty", "prompt")
    options = json_field(fields, "reference_options", [])
    dialogue = json_field(fields, "dialogue", [])
    if not isinstance(options, list) or not isinstance(dialogue, list):
        raise bad("invalid_json", "reference_options and dialogue must be arrays")
    per_index = {}
    for item in options:
        if not isinstance(item, dict) or set(item)-{"index", "role", "note", "soundtrack_policy"}:
            raise bad("invalid_reference_options", "invalid reference option")
        i = item.get("index")
        if type(i) is not int or not 0 <= i < len(uploads) or i in per_index:
            raise bad("invalid_reference_options", "reference indexes must be unique and in range")
        if "note" in item and (not isinstance(item["note"], str) or len(item["note"].encode()) > 4096):
            raise bad("invalid_reference_options", "reference note is invalid or too long")
        per_index[i] = item
    for line in dialogue:
        if (not isinstance(line, dict) or set(line)-{"text", "language", "speaker"}
                or not isinstance(line.get("text"), str) or not line["text"].strip()
                or any(not isinstance(v, str) for v in line.values())):
            raise bad("invalid_dialogue", "dialogue entries require text and optional language/speaker strings")
    silent_raw = fields.get("silent", ["false"])[0].lower()
    if silent_raw not in {"true", "false"}:
        raise bad("invalid_silent", "silent must be true or false")
    silent = silent_raw == "true"
    creativity = fields.get("creativity", ["balanced"])[0]
    if creativity not in {"restrained", "balanced", "bold", "extreme"}:
        raise bad("invalid_creativity", "invalid creativity")
    try:
        seed = int(fields.get("context_ir_seed", ["7"])[0])
    except ValueError as exc:
        raise bad("invalid_seed", "context_ir_seed must be an integer") from exc
    if not 0 <= seed < 2**63:
        raise bad("invalid_seed", "context_ir_seed must be between 0 and 2^63-1")
    video_fields = {k: list(v) for k, v in fields.items() if k in {"model", "prompt", "seconds", "num_frames", "size", "width", "height"}}
    workflow = pool.adapter.workflow
    if workflow == "fl2va" and uploads:
        roles = [per_index.get(i, {}).get("role", "frame_anchor_first" if i == 0 else "frame_anchor_last") for i in range(len(uploads))]
        allowed = [["frame_anchor_first"], ["frame_anchor_last"], ["frame_anchor_first", "frame_anchor_last"]]
        if roles not in allowed:
            raise bad("invalid_reference_role", "FL2VA anchors must match first/last upload order")
        video_fields["frame_indices"] = [encode([0 if role == "frame_anchor_first" else -1 for role in roles])]
    adapter = service.adapters[pool.pool_id]
    normalized = adapter.normalize_request(video_fields, uploads, model)
    if normalized.get("super_resolution"):
        raise bad("unsupported_context_ir_size", "ContextIR currently supports base generation sizes only", "size")
    entries = []
    for upload, meta in zip(uploads, normalized["reference_metadata"]):
        kind, i = meta["kind"], upload.ordinal
        opts = per_index.get(i, {})
        default = ("frame_anchor_first" if i == 0 else "frame_anchor_last") if workflow == "fl2va" else {"image": "subject", "video": "edit_source", "audio": None}[kind]
        role = opts.get("role", default)
        allowed = {"image": {"subject", "environment", "style"}, "video": {"edit_source", "motion_reference"}, "audio": {"voice_timbre", "bgm"}}[kind] if workflow == "ref2va" else {"frame_anchor_first", "frame_anchor_last"}
        if not isinstance(role, str) or role not in allowed:
            raise bad("invalid_reference_role", "reference role must be explicitly valid for the target workflow")
        has_audio = kind == "video" and audio_present(upload.path)
        policy = opts.get("soundtrack_policy")
        if (kind != "video" and policy is not None) or (policy is not None and policy not in {"preserve", "ignore"}):
            raise bad("invalid_soundtrack_policy", "soundtrack_policy applies to video and must be preserve or ignore")
        if has_audio and policy is None:
            raise bad("soundtrack_policy_required", "choose preserve or ignore for a video with audio")
        entries.append({**upload.manifest_entry(relative_to=upload.path.parent.parent), **meta,
                        "role": role, "role_stated": workflow != "ref2va" or "role" in opts,
                        "note": opts.get("note"), "has_audio": has_audio,
                        "soundtrack_policy": policy})
    # Explicit pixel ratio + megapixels reconstructs the exact /32 canvas.
    # It does not ask the compiler to infer a canvas from an attached picture.
    width, height = normalized["width"], normalized["height"]
    frames = align_frame_count(normalized["num_frames"])
    mode = "ref2va" if workflow == "ref2va" else (
        "t2va" if not entries else "fl2va" if len(entries) == 2 else
        "l2va" if entries[0]["role"] == "frame_anchor_last" else "i2va")
    generation = {"size": f"{width}x{height}", "fps": 24}
    duration_field = "seconds" if "seconds" in fields else "num_frames"
    generation[duration_field] = normalized[duration_field]
    if normalized.get("frame_indices"):
        generation["frame_indices"] = normalized["frame_indices"]
    return {"model": model, "target_pool_id": pool.pool_id, "workflow": workflow, "mode": mode,
            "intent": fields["prompt"][0], "seconds": normalized["seconds"],
            "frames": frames, "canvas": [width, height], "generation_parameters": generation,
            "references": entries, "context_ir_seed": seed, "creativity": creativity,
            "dialogue": dialogue, "silent": silent}


def request_digest(normalized):
    value = dict(normalized)
    value["references"] = [{k: v for k, v in entry.items() if k not in {"filename", "path"}}
                           for entry in value["references"]]
    return hashlib.sha256(encode(value).encode()).hexdigest()
