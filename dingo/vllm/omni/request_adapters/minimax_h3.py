# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 FL2VA and Ref2VA request compatibility adapter.

The adapter is loaded only when explicitly selected by the Omni worker. It
translates validation-era string envelopes into the native vLLM-Omni
``multi_modal_data`` contract without patching global classes.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import json
import logging
import os
import stat
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_FL2VA_KEYFRAME_COUNT = 2
_FL2VA_KEYFRAME_ENVELOPE_TYPE = "fl2va_keyframes_v1"
_REF2VA_MIXED_ENVELOPE_TYPE = "ref2va_mixed_v1"
_VIDEO_PREFIX = "data:video/mp4;base64,"
_AUDIO_PREFIX = "data:audio/wav;base64,"
_MAX_VIDEO_BYTES = 50 * 1024 * 1024
_MAX_AUDIO_BYTES = 15 * 1024 * 1024
_ASPECT_RATIOS = {
    "21:9": 21.0 / 9.0,
    "16:9": 16.0 / 9.0,
    "4:3": 4.0 / 3.0,
    "1:1": 1.0,
    "3:4": 3.0 / 4.0,
    "9:16": 9.0 / 16.0,
}
_MEDIA_CACHE_LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True)
class _StoredMedia:
    path: str
    bytes: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class _MixedReference:
    images: tuple[Any, ...]
    videos: tuple[_StoredMedia, ...]
    audios: tuple[_StoredMedia, ...]


@dataclasses.dataclass(frozen=True)
class _KeyframeReference:
    images: tuple[Any, ...]
    frame_indices: tuple[int, ...]


def _nearest_aspect_ratio(width: int, height: int) -> str:
    requested = float(width) / float(height)
    name, supported = min(
        _ASPECT_RATIOS.items(), key=lambda item: abs(requested - item[1])
    )
    relative_error = abs(requested - supported) / supported
    if relative_error > 0.05:
        raise ValueError(
            "MiniMax-H3 requested size does not match a supported aspect ratio: "
            f"size={width}x{height}, ratio={requested:.6f}, nearest={name} "
            f"({supported:.6f}), relative_error={relative_error:.2%}"
        )
    return name


def _decode_data_url(reference: str, prefix: str, limit: int, label: str) -> bytes:
    encoded = reference[len(prefix) :]
    max_encoded_bytes = ((limit + 2) // 3) * 4
    if not encoded or len(encoded) > max_encoded_bytes:
        raise ValueError(f"{label} reference is empty or exceeds the validation limit")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError(f"{label} reference has invalid base64 data") from exc
    if len(payload) > limit:
        raise ValueError(f"{label} reference exceeds the validation limit")
    return payload


def _persist_data_url(
    reference: str,
    *,
    prefix: str,
    limit: int,
    cache_root: Path,
    media_dir: Path,
    max_cache_bytes: int,
    suffix: str,
) -> _StoredMedia:
    label = "Ref2VA MP4" if suffix == ".mp4" else "Ref2VA WAV"
    payload = _decode_data_url(reference, prefix, limit, label)
    if suffix == ".mp4" and b"ftyp" not in payload[:64]:
        raise ValueError("Ref2VA video reference is not an MP4 file")
    if suffix == ".wav" and (
        len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE"
    ):
        raise ValueError("Ref2VA audio reference is not a WAV file")

    digest = hashlib.sha256(payload).hexdigest()
    with _MEDIA_CACHE_LOCK:
        media_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        media_dir.chmod(0o700)
        path = media_dir / f"{digest}{suffix}"
        if path.exists() or path.is_symlink():
            existing = path.lstat()
            if not stat.S_ISREG(existing.st_mode) or existing.st_size != len(payload):
                raise ValueError(f"Ref2VA media cache entry is invalid: {path}")
            path.chmod(0o600)
        else:
            cached_bytes = _media_cache_size(cache_root)
            if cached_bytes > max_cache_bytes - len(payload):
                raise ValueError(
                    "Ref2VA media cache capacity exceeded: "
                    f"current={cached_bytes}, requested={len(payload)}, "
                    f"limit={max_cache_bytes}"
                )
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=media_dir,
                    prefix=f".{digest}.",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(payload)
                temporary.chmod(0o600)
                os.replace(temporary, path)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    return _StoredMedia(str(path), len(payload), digest)


def _media_cache_size(cache_root: Path) -> int:
    total = 0
    for name in ("ref2va-inputs", "ref2va-audio-inputs"):
        directory = cache_root / name
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            try:
                metadata = entry.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
    return total


class MiniMaxH3RequestAdapter:
    """Translate one explicitly selected MiniMax-H3 checkpoint workflow."""

    def __init__(self, workflow: str | None, media_dir: str, media_max_bytes: int):
        if workflow not in {"fl2va", "ref2va"}:
            raise ValueError("MiniMax-H3 request adapter requires fl2va or ref2va")
        self.workflow = workflow
        if media_max_bytes <= 0:
            raise ValueError("MiniMax-H3 request adapter media limit must be > 0")
        root = Path(media_dir)
        self.media_root = root
        self.media_max_bytes = media_max_bytes
        self.video_dir = root / "ref2va-inputs"
        self.audio_dir = root / "ref2va-audio-inputs"

    async def load_reference(self, req: Any, loader: Any) -> Any | None:
        if req.input_reference is not None and req.input_references is not None:
            raise ValueError(
                "input_reference and input_references are mutually exclusive"
            )
        if req.input_references is not None:
            if not req.input_references:
                raise ValueError("input_references must not be empty")
            if self.workflow == "fl2va" and len(req.input_references) != 2:
                raise ValueError(
                    "MiniMax-H3 FL2VA requires exactly two ordered keyframes"
                )
            return list(
                await asyncio.gather(
                    *(loader.load_image(item) for item in req.input_references)
                )
            )
        if req.input_reference is None:
            return None

        reference = req.input_reference
        stripped = reference.lstrip()
        if self.workflow == "fl2va":
            return await self._load_fl2va_reference(stripped, loader)
        return await self._load_ref2va_reference(stripped, loader)

    async def _load_fl2va_reference(self, reference: str, loader: Any) -> Any:
        if reference.startswith(_VIDEO_PREFIX):
            raise ValueError(
                "MiniMax-H3 FL2VA worker does not accept Ref2VA video references"
            )
        if reference.startswith("{"):
            try:
                envelope = json.loads(reference)
            except json.JSONDecodeError as exc:
                raise ValueError("input_reference is not valid JSON") from exc
            if not isinstance(envelope, Mapping):
                raise ValueError("FL2VA keyframe envelope must be a JSON object")
            return await self._decode_keyframe_envelope(envelope, loader)
        if not reference.startswith("["):
            return await loader.load_image(reference)
        try:
            references = json.loads(reference)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "input_reference starts with '[' but is not a valid JSON array"
            ) from exc
        if (
            not isinstance(references, list)
            or len(references) != _FL2VA_KEYFRAME_COUNT
            or not all(isinstance(item, str) and item for item in references)
        ):
            raise ValueError(
                "MiniMax-H3 FL2VA input_reference must encode exactly two "
                "non-empty image URL strings"
            )
        return list(await asyncio.gather(*(loader.load_image(x) for x in references)))

    async def _decode_keyframe_envelope(
        self, envelope: Mapping[str, Any], loader: Any
    ) -> _KeyframeReference:
        allowed = {"type", "images", "frame_indices"}
        unknown = set(envelope) - allowed
        if unknown:
            raise ValueError(
                f"FL2VA keyframe envelope has unknown keys: {sorted(unknown)}"
            )
        if envelope.get("type") != _FL2VA_KEYFRAME_ENVELOPE_TYPE:
            raise ValueError(
                "FL2VA keyframe envelope type must be "
                f"{_FL2VA_KEYFRAME_ENVELOPE_TYPE!r}"
            )
        images = envelope.get("images")
        if (
            not isinstance(images, list)
            or len(images) != 1
            or not all(isinstance(item, str) and item for item in images)
        ):
            raise ValueError(
                "FL2VA single-tail envelope requires exactly one image URL"
            )
        if envelope.get("frame_indices") != [-1]:
            raise ValueError("FL2VA single-tail envelope requires frame_indices=[-1]")
        if not images[0].lstrip().startswith("data:image/"):
            raise ValueError("FL2VA single-tail image must use an image data URL")
        decoded = await loader.load_image(images[0])
        return _KeyframeReference((decoded,), (-1,))

    async def _load_ref2va_reference(self, reference: str, loader: Any) -> Any:
        if reference.startswith(_VIDEO_PREFIX):
            return await asyncio.to_thread(
                _persist_data_url,
                reference,
                prefix=_VIDEO_PREFIX,
                limit=_MAX_VIDEO_BYTES,
                cache_root=self.media_root,
                media_dir=self.video_dir,
                max_cache_bytes=self.media_max_bytes,
                suffix=".mp4",
            )
        if reference.startswith("["):
            raise ValueError(
                "MiniMax-H3 Ref2VA worker does not accept FL2VA keyframe arrays"
            )
        if not reference.startswith("{"):
            return await loader.load_image(reference)
        return await self._decode_mixed_envelope(reference, loader)

    async def _decode_mixed_envelope(
        self, reference: str, loader: Any
    ) -> _MixedReference:
        try:
            envelope = json.loads(reference)
        except json.JSONDecodeError as exc:
            raise ValueError("Ref2VA mixed input_reference is not valid JSON") from exc
        if not isinstance(envelope, dict):
            raise ValueError("Ref2VA mixed input_reference must be a JSON object")
        allowed = {"type", "images", "videos", "audios"}
        unknown = set(envelope) - allowed
        if unknown:
            raise ValueError(
                "Ref2VA mixed input_reference has unknown keys: " f"{sorted(unknown)}"
            )
        if envelope.get("type") != _REF2VA_MIXED_ENVELOPE_TYPE:
            raise ValueError(
                "Ref2VA mixed input_reference type must be "
                f"{_REF2VA_MIXED_ENVELOPE_TYPE!r}"
            )
        images = envelope.get("images", [])
        videos = envelope.get("videos", [])
        audios = envelope.get("audios", [])
        for name, values in (
            ("images", images),
            ("videos", videos),
            ("audios", audios),
        ):
            if not isinstance(values, list) or not all(
                isinstance(item, str) and item for item in values
            ):
                raise ValueError(
                    f"Ref2VA mixed {name} must be a list of non-empty strings"
                )
        if not images or not videos or not audios:
            raise ValueError(
                "Ref2VA mixed validation requires at least one image, video, and audio"
            )
        if len(images) > 9 or len(videos) > 3 or len(audios) > 3:
            raise ValueError(
                "Ref2VA mixed reference counts exceed the "
                "9 image/3 video/3 audio limits"
            )
        if len(images) + len(videos) + len(audios) > 12:
            raise ValueError("Ref2VA mixed references exceed the 12 item total limit")
        if not all(item.lstrip().startswith("data:image/") for item in images):
            raise ValueError("Ref2VA mixed images must use image data URLs")
        if not all(item.lstrip().startswith(_VIDEO_PREFIX) for item in videos):
            raise ValueError("Ref2VA mixed videos must use MP4 data URLs")
        if not all(item.lstrip().startswith(_AUDIO_PREFIX) for item in audios):
            raise ValueError("Ref2VA mixed audios must use WAV data URLs")

        decoded_images, decoded_videos, decoded_audios = await asyncio.gather(
            asyncio.gather(*(loader.load_image(item) for item in images)),
            asyncio.gather(
                *(
                    asyncio.to_thread(
                        _persist_data_url,
                        item.lstrip(),
                        prefix=_VIDEO_PREFIX,
                        limit=_MAX_VIDEO_BYTES,
                        cache_root=self.media_root,
                        media_dir=self.video_dir,
                        max_cache_bytes=self.media_max_bytes,
                        suffix=".mp4",
                    )
                    for item in videos
                )
            ),
            asyncio.gather(
                *(
                    asyncio.to_thread(
                        _persist_data_url,
                        item.lstrip(),
                        prefix=_AUDIO_PREFIX,
                        limit=_MAX_AUDIO_BYTES,
                        cache_root=self.media_root,
                        media_dir=self.audio_dir,
                        max_cache_bytes=self.media_max_bytes,
                        suffix=".wav",
                    )
                    for item in audios
                )
            ),
        )
        return _MixedReference(
            tuple(decoded_images), tuple(decoded_videos), tuple(decoded_audios)
        )

    def build_engine_inputs(
        self, req: Any, reference: Any, generic_builder: Callable[..., Any]
    ) -> Any:
        if isinstance(reference, _KeyframeReference):
            inputs = generic_builder(req, image=None)
            inputs.prompt["multi_modal_data"] = {"image": list(reference.images)}
            self._set_diffusion_arg(
                inputs, "frame_indices", list(reference.frame_indices)
            )
        elif isinstance(reference, _MixedReference):
            inputs = generic_builder(req, image=None)
            inputs.prompt["multi_modal_data"] = {
                "image": list(reference.images),
                "video": [item.path for item in reference.videos],
                "audio": [item.path for item in reference.audios],
            }
        elif isinstance(reference, _StoredMedia):
            inputs = generic_builder(req, image=None)
            inputs.prompt["multi_modal_data"] = {"video": reference.path}
        else:
            inputs = generic_builder(req, image=reference)

        if self.workflow == "fl2va" and reference is None:
            sampling = next(
                (
                    item
                    for item in inputs.sampling_params_list or []
                    if getattr(item, "width", None) and getattr(item, "height", None)
                ),
                None,
            )
            if sampling is None:
                raise ValueError(
                    "MiniMax-H3 T2VA request has no diffusion sampling parameters"
                )
            self._set_diffusion_arg(
                inputs,
                "aspect_ratio",
                _nearest_aspect_ratio(sampling.width, sampling.height),
                overwrite=False,
            )
        return inputs

    @staticmethod
    def _set_diffusion_arg(
        inputs: Any, key: str, value: Any, *, overwrite: bool = True
    ) -> None:
        patched = 0
        for sampling in inputs.sampling_params_list or []:
            if not getattr(sampling, "width", None) or not getattr(
                sampling, "height", None
            ):
                continue
            extra_args = getattr(sampling, "extra_args", None)
            if not isinstance(extra_args, dict):
                continue
            if overwrite:
                extra_args[key] = value
            else:
                extra_args.setdefault(key, value)
            patched += 1
        if patched == 0:
            raise ValueError(
                f"MiniMax-H3 request found no diffusion sampling params for {key}"
            )
