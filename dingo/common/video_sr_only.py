# SPDX-License-Identifier: Apache-2.0
"""Private SR-only request geometry; no model loading or cluster access."""
from dingo.common.video_super_resolution import H3_SR_PROFILE_V3, SR_PAYLOAD_KEY
from dingo.video_gateway.adapters.h3_output_policy import plan_h3_output

SOURCE_KEY = "dingo_sr_source"


def validate_sr_only(payload):
    if payload.get("operation") != "sr_only":
        raise ValueError("standalone SR Worker accepts only sr_only")
    source = payload.get(SOURCE_KEY)
    if not isinstance(source, dict):
        raise ValueError("missing SR source descriptor")
    media = source.get("media")
    if not isinstance(media, dict) or set(media) != {"width", "height", "frames", "fps", "audio"}:
        raise ValueError("invalid source media contract")
    if any(type(media[k]) is not int for k in ("width", "height", "frames", "fps")) or type(media["audio"]) is not bool:
        raise ValueError("invalid source media types")
    w, h = media["width"], media["height"]
    if not (256 <= w <= 2048 and 256 <= h <= 2048 and 0.4 <= w / h <= 2.5):
        raise ValueError("source canvas is outside SR-only bounds")
    if w * h > 1100000 or not 1 <= media["frames"] <= 362 or media["fps"] != 24:
        raise ValueError("SR-only requires a bounded 768P-class 24 FPS source")
    sr = payload.get(SR_PAYLOAD_KEY)
    if not isinstance(sr, dict):
        raise ValueError("missing SR geometry")
    plan = plan_h3_output(sr.get("output_width"), sr.get("output_height"), sr_profile=H3_SR_PROFILE_V3)
    if plan.sr_profile is None:
        raise ValueError("SR-only output must be an admitted high-resolution canvas")
    expected = dict(schema_version=1, profile=H3_SR_PROFILE_V3,
                    base_width=w, base_height=h, output_width=plan.output_width,
                    output_height=plan.output_height, sr_width=plan.sr_width, sr_height=plan.sr_height)
    if sr != expected or any(type(sr[k]) is not int for k in expected if k != "profile"):
        raise ValueError("invalid SR-only output plan")
    if payload.get("size") != f"{w}x{h}" or payload.get("nvext", {}).get("fps") != 24:
        raise ValueError("SR-only source geometry mismatch")
    return expected
