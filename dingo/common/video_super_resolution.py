# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private, dependency-light SR negotiation; not an arbitrary engine extension."""

from collections.abc import Mapping
from typing import Any

SR_PAYLOAD_KEY = "dingo_super_resolution"
SR_CAPABILITY = "video_super_resolution_v1"
H3_SR_PROFILE = "minimax_h3_seedvr2.v1"
H3_SR_PROFILE_V2 = "minimax_h3_seedvr2.v2"
H3_SR_PROFILE_V3 = "minimax_h3_seedvr2.v3"
H3_SR_PROFILES = frozenset({H3_SR_PROFILE, H3_SR_PROFILE_V2, H3_SR_PROFILE_V3})


def configured_sr_profile(adapter_name: str, options: Mapping[str, Any]) -> str | None:
    raw = options.get("super_resolution")
    if raw is None:
        return None
    if adapter_name != "minimax_h3":
        raise ValueError("super_resolution is not supported by this adapter")
    if not isinstance(raw, Mapping) or set(raw) - {"enabled", "profile"}:
        raise ValueError("invalid super_resolution configuration")
    enabled = raw.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("super_resolution.enabled must be a boolean")
    profile = raw.get("profile", H3_SR_PROFILE)
    if profile not in H3_SR_PROFILES:
        raise ValueError("unsupported model super_resolution profile")
    return profile if enabled else None


def negotiated_sr_profiles(response: Any) -> frozenset[str]:
    """Missing, malformed or non-ready SR advertisements never enable SR."""
    if (
        not isinstance(response, Mapping)
        or type(response.get("schema_version")) is not int
        or response.get("schema_version") != 1
        or response.get("accepting") is not True
    ):
        return frozenset()
    capabilities = response.get("capabilities")
    if not isinstance(capabilities, list) or SR_CAPABILITY not in capabilities:
        return frozenset()
    sr = response.get("super_resolution")
    if not isinstance(sr, Mapping) or sr.get("ready") is not True:
        return frozenset()
    profiles = sr.get("profiles")
    if not isinstance(profiles, list) or any(type(p) is not str for p in profiles):
        return frozenset()
    return frozenset(p for p in profiles if p in H3_SR_PROFILES)
