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
    from dingo.video_gateway.adapters.h3_resolution import restore_sr_plan
    plan = restore_sr_plan(sr, H3_SR_PROFILE_V3)
    if plan.sr_profile is None:
        raise ValueError("SR-only output must be an admitted high-resolution canvas")
    expected = dict(schema_version=1, profile=H3_SR_PROFILE_V3,
                    base_width=w, base_height=h, output_width=plan.output_width,
                    output_height=plan.output_height, sr_width=plan.sr_width, sr_height=plan.sr_height)
    if plan.adaptive_keyframe:
        if (w, h) != (plan.base_width, plan.base_height):
            raise ValueError("adaptive SR source does not match generated base")
        expected["adaptive_keyframe"] = list(plan.adaptive_keyframe)
    if sr != expected or any(type(sr[k]) is not int for k in expected if k not in ("profile", "adaptive_keyframe")):
        raise ValueError("invalid SR-only output plan")
    if payload.get("size") != f"{w}x{h}" or payload.get("nvext", {}).get("fps") != 24:
        raise ValueError("SR-only source geometry mismatch")
    if not plan.adaptive_keyframe:
        source_crop_box(w, h, plan.output_width, plan.output_height, plan.sr_width, plan.sr_height)
    return expected


def source_aspect_ratio(width, height):
    """One admission policy for Gateway uploads and native SR preprocessing."""
    from dingo.video_gateway.adapters.h3_resolution import BASE_CANVASES
    from dingo.video_gateway.adapters.h3_output_policy import H3_SR_V3_PRESETS
    from dingo.video_gateway.adapters.h3_shape import NAMED_ASPECT_RATIOS
    if (type(width) is not int or type(height) is not int
            or not 256 <= width <= 2048 or not 256 <= height <= 2048
            or width * height > 1100000 or not .4 <= width / height <= 2.5):
        raise ValueError("source canvas is outside SR-only bounds")
    size = (width, height)
    name = next((name for name, value in BASE_CANVASES.items() if value == size), None)
    if name is None:
        name = next((value[-1] for value in H3_SR_V3_PRESETS.values() if value[:2] == size), None)
    if name is None:
        ratio = width / height
        name = min(NAMED_ASPECT_RATIOS, key=lambda key: abs(ratio / NAMED_ASPECT_RATIOS[key] - 1))
        if abs(ratio / NAMED_ASPECT_RATIOS[name] - 1) > .03:
            raise ValueError("source ratio does not match an admitted SR canvas")
    return name


def source_crop_box(source_width, source_height, output_width, output_height, model_width, model_height):
    """Center-crop uploaded pixels to the fixed model aspect, without stretching.

    The integer crop has the exact model ratio. Even dimensions and offsets
    retain YUV420 chroma alignment; CUDA still uses the existing bounded canvas.
    """
    from math import gcd
    name = source_aspect_ratio(source_width, source_height)
    plan = plan_h3_output(output_width, output_height, sr_profile=H3_SR_PROFILE_V3)
    # Keep the two legacy 5:3 delivery presets available to internal pipeline
    # callers as well as the twelve public named-resolution presets.
    if plan.sr_profile != H3_SR_PROFILE_V3 or name != plan.aspect_ratio:
        raise ValueError("source aspect ratio does not match the delivery canvas")
    if (model_width, model_height) != (plan.sr_width, plan.sr_height):
        raise ValueError("uploaded SR requires the whitelisted internal canvas")
    divisor = gcd(model_width, model_height)
    unit_width, unit_height = model_width // divisor, model_height // divisor
    factor = min(source_width // unit_width, source_height // unit_height)
    factor -= factor % 2
    if factor < 2:
        raise ValueError("source is too small for an aligned aspect-preserving crop")
    width, height = unit_width * factor, unit_height * factor
    left = ((source_width - width) // 2) & ~1
    top = ((source_height - height) // 2) & ~1
    return left, top, width, height
