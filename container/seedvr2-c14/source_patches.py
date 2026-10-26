# SPDX-License-Identifier: Apache-2.0
"""Exact-source, default-off C14 hooks. No whole-directory replacement."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX = "vllm_omni/diffusion/models/seedvr2/"
IMPORT = "from vllm_omni.diffusion.models.seedvr2 import c14_capture as _c14\n"


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Pinned source anchor is absent or ambiguous: " + before[:100])
    return source.replace(before, after, 1)


def transform(relative, data):
    source = data.decode("utf-8")
    if relative.endswith("pipeline_seedvr2.py"):
        source = replace_once(source, "from vllm_omni.diffusion.models.seedvr2 import config as envs\n", IMPORT + "from vllm_omni.diffusion.models.seedvr2 import config as envs\n")
        source = replace_once(source, "def finish_video(decoded: torch.Tensor, sample: torch.Tensor, frame_count: int, method: str) -> torch.Tensor:",
                              "def finish_video(decoded: torch.Tensor, sample: torch.Tensor, frame_count: int, method: str, capture=None) -> torch.Tensor:")
        color_call = '''        corrected = correct_video_color(
            ((decoded[:, :, frame].float() + 1) / 2).clamp(0, 1),
            ((sample[:, :, frame].float() + 1) / 2).clamp(0, 1),
            method=method,
        )'''
        source = replace_once(source, color_call, "        if capture is None:\n" + "\n".join("    " + line for line in color_call.splitlines()) + '''
        else:
            color_input = ((decoded[:, :, frame].float() + 1) / 2).clamp(0, 1)
            color_reference = ((sample[:, :, frame].float() + 1) / 2).clamp(0, 1)
            corrected = correct_video_color(color_input, color_reference, method=method)
            _c14.color_frame(capture, index, color_input, color_reference, corrected)
            del color_input, color_reference''')
        source = replace_once(source, "    return video\n\n\ndef sample_noise", "    _c14.quantized(capture, video)\n    return video\n\n\ndef sample_noise")
        source = replace_once(source, "            prepared = request.prepared_layout\n            sample = prepare_video", "            prepared = request.prepared_layout\n            capture = None  # assigned by the scoped diagnostic session below\n            sample = prepare_video")
        source = replace_once(source, "            latent = self.vae.encode(sample).sample(generator=generator)", '''            if capture is None:
                latent = self.vae.encode(sample).sample(generator=generator)
            else:
                capture.tensor("input_decoded", prepared.frames, "THWC" if prepared.frames.dtype == torch.uint8 else "TCHW", "decoded RGB")
                capture.tensor("prepared_sample", sample, "BCTHW", "RGB [-1,1], padded")
                posterior = self.vae.encode(sample)
                capture.tensor("posterior_mean", posterior.mean, "BCTHW")
                capture.tensor("posterior_logvar", posterior.logvar, "BCTHW")
                capture.tensor("rng_before", generator.get_state(), "N", "generator state before posterior sample")
                latent = posterior.sample(generator=generator)
                capture.tensor("rng_after", generator.get_state(), "N", "generator state after posterior sample")
                capture.tensor("posterior_sample", latent, "BCTHW")
                del posterior''')
        source = replace_once(source, "            noise = sample_noise(condition, generator)\n", '''            noise = sample_noise(condition, generator)
            if capture is not None:
                capture.tensor("dit_condition", condition, "THWC")
                capture.tensor("dit_noise", noise, "THWC")
''')
        source = replace_once(source, "            # Reference Euler returns fp32, then VAE casts to fp16 before scaling.", '''            if capture is not None:
                capture.tensor("dit_velocity", velocity, "NC", "native NaDiT canonical rows")
            # Reference Euler returns fp32, then VAE casts to fp16 before scaling.''')
        source = replace_once(source, "            decoded = self.vae.decode((restored / 0.9152).permute(3, 0, 1, 2).unsqueeze(0))", '''            if capture is None:
                decoded = self.vae.decode((restored / 0.9152).permute(3, 0, 1, 2).unsqueeze(0))
            else:
                decode_input = (restored / 0.9152).permute(3, 0, 1, 2).unsqueeze(0)
                capture.tensor("vae_decode_input", decode_input, "BCTHW", "actual decode input after 0.9152 transform")
                decoded = self.vae.decode(decode_input)
                capture.tensor("decoded_rgb", decoded, "BCTHW", "clamp-before VAE RGB")
                del decode_input''')
        source = replace_once(source, "            video = finish_video(decoded, sample, prepared.frame_count, method)", "            video = finish_video(decoded, sample, prepared.frame_count, method, capture=capture)")
        start = source.index("            capture = None  # assigned by the scoped diagnostic session below\n")
        end = source.index("        return outputs", start)
        body = source[start:end].splitlines(keepends=True)[1:]
        source = source[:start] + "            with _c14.session(request, self, prepared) as capture:\n" + "".join("    " + line if line.strip() else line for line in body) + source[end:]
    elif relative.endswith("/vae.py"):
        source = replace_once(source, "from vllm_omni.diffusion.distributed.parallel_state import get_sp_group\n", "from vllm_omni.diffusion.distributed.parallel_state import get_sp_group\n" + IMPORT)
        source = replace_once(source, "            decoded = self.decoder(z, context)\n", "            decoded = self.decoder(z, context)\n            _c14.decode_chunk(0, z.shape[2], decoded)\n")
        source = replace_once(source, "                chunks.append(self.decoder(z[:, :, start : start + chunk_size], context))\n", "                chunks.append(self.decoder(z[:, :, start : start + chunk_size], context))\n                _c14.decode_chunk(start, min(start + chunk_size, z.shape[2]), chunks[-1])\n")
    elif relative.endswith("long_video.py"):
        source = replace_once(source, "from vllm_omni.diffusion.models.seedvr2 import config as envs\n", IMPORT + "from vllm_omni.diffusion.models.seedvr2 import config as envs\n")
        source = replace_once(source, "    authorization: str,\n) -> list[np.ndarray]:\n    response", "    authorization: str,\n    c14_context=None,\n) -> list[np.ndarray]:\n    segment = _segment(frames)\n    _c14.media(c14_context, \"window-input.mkv\", segment)\n    response")
        source = replace_once(source, '                {"video_codec_options": LOSSLESS, "color_correction_method": color_correction_method}\n', '                _c14.child_options(c14_context, {"video_codec_options": LOSSLESS, "color_correction_method": color_correction_method})\n')
        source = replace_once(source, 'files={"input_references": ("window.mkv", _segment(frames), "video/x-matroska")}', 'files={"input_references": ("window.mkv", segment, "video/x-matroska")}')
        source = replace_once(source, "    with av.open(io.BytesIO(response.content)) as container:\n", "    _c14.media(c14_context, \"window-output.mp4\", response.content)\n    with av.open(io.BytesIO(response.content)) as container:\n")
        source = replace_once(source, "    def windows() -> Iterator[list[av.VideoFrame]]:", "    def windows() -> Iterator[tuple[int, list[av.VideoFrame]]]:")
        source = replace_once(source, "            yield batch\n", "            yield start, batch\n")
        source = replace_once(source, "    def restore(frames: list[av.VideoFrame]) -> Future[list[np.ndarray]]:\n", "    def restore(batch: tuple[int, list[av.VideoFrame]]) -> Future[list[np.ndarray]]:\n        start, frames = batch\n")
        source = replace_once(source, "        return pool.submit(_restore, frames, width, height, seed, color_correction_method, port, authorization)", '''        context = _c14.window_context(job, start, len(frames), width, height, target, seed, color_correction_method)
        return pool.submit(_restore, frames, width, height, seed, color_correction_method, port, authorization, context)''')
        source = replace_once(source, "            nonlocal written\n            frame = av.VideoFrame.from_ndarray", "            nonlocal written\n            _c14.fused_frame(job, written, array)\n            frame = av.VideoFrame.from_ndarray")
        source = replace_once(source, '        video.options = {"preset": "veryfast", "crf": "18", "tune": "zerolatency", "bf": "0"}\n', '        video.options = {"preset": "veryfast", "crf": "18", "tune": "zerolatency", "bf": "0"}\n        _c14.long_encoder(job, video)\n')
        source = replace_once(source, "        subprocess.run(\n", "        _c14.run_mux(job, subprocess.run,\n")
    elif relative.endswith("serving_video.py"):
        source = replace_once(source, "        video_bytes = _encode_video_bytes(\n            artifacts.videos[0],\n            fps=artifacts.output_fps,\n            video_codec_options=video_codec_options,\n            frame_converter=self._video_frame_converter,\n            **encoding_options,\n        )", '''        from vllm_omni.diffusion.models.seedvr2 import c14_capture as _c14
        with _c14.encoding_context(request.extra_params, video_codec_options, self._video_frame_converter, encoding_options):
            video_bytes = _encode_video_bytes(
                artifacts.videos[0],
                fps=artifacts.output_fps,
                video_codec_options=video_codec_options,
                frame_converter=self._video_frame_converter,
                **encoding_options,
            )''')
    elif relative.endswith("video_api_utils.py"):
        source = replace_once(source, '    reason_field = "" if reason is None else f" reason={reason}"', '''    from vllm_omni.diffusion.models.seedvr2 import c14_capture as _c14
    _c14.encoding_route(selected_path, frames, frame_shape, common_dtype, fps, audio_sample_rate, effective_frame_conversion_workers, reason)
    reason_field = "" if reason is None else f" reason={reason}"''')
    elif relative.endswith("/media_utils.py"):
        # Both muxers are possible. Capture the actual merged options, without
        # changing defaults or selecting a different encoding implementation.
        source = replace_once(source, "\n    v_stream.options = options\n", "\n    v_stream.options = options\n    from vllm_omni.diffusion.models.seedvr2 import c14_capture as _c14\n    _c14.muxer_settings(v_stream, video_codec, fps, options)\n")
        source = replace_once(source, "\n        v_stream.options = options\n", "\n        v_stream.options = options\n        from vllm_omni.diffusion.models.seedvr2 import c14_capture as _c14\n        _c14.muxer_settings(v_stream, video_codec, fps, options)\n")
    else:
        raise ValueError("Unknown C14 source: " + relative)
    compile(source, relative, "exec")
    return source.encode("utf-8")


def locked_changes(root):
    """Validate all sources before returning bytes; caller owns atomic RECORD update."""
    lock = json.loads((HERE / "source-patches.lock.json").read_text(encoding="utf-8"))
    changes = []
    for relative, hashes in lock["files"].items():
        path = Path(root) / relative
        if path.is_symlink():
            raise ValueError("Refusing symlinked installed source")
        before = path.read_bytes()
        if hashlib.sha256(before).hexdigest() != hashes["before_sha256"]:
            raise ValueError("C14 pinned source mismatch: " + relative)
        after = transform(relative, before)
        if hashlib.sha256(after).hexdigest() != hashes["after_sha256"]:
            raise ValueError("C14 derived source mismatch: " + relative)
        changes.append((relative, before, after))
    return changes


def apply_sources(root):
    changes = locked_changes(root)
    for relative, before, after in changes:
        (Path(root) / relative).write_bytes(after)
    return changes
