# SPDX-License-Identifier: Apache-2.0
"""Instrument the exact C14-patched 031 pipeline without replacing math."""
from __future__ import annotations

import ast


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("SeedVR2 telemetry source anchor absent/ambiguous: " + before[:100])
    return source.replace(before, after, 1)


def wrap(source, begin, end, name):
    start = source.index(begin)
    stop = source.index(end, start)
    block = source[start:stop]
    indent = begin[:len(begin) - len(begin.lstrip())]
    body = "".join("    " + line if line.strip() else line for line in block.splitlines(keepends=True))
    return source[:start] + indent + f'with observation.stage("{name}"):\n' + body + source[stop:]


def transform(data):
    source = data.decode("utf-8")
    source = replace_once(source,
        "from vllm_omni.diffusion.models.seedvr2 import c14_capture as _c14\n",
        "from vllm_omni.diffusion.models.seedvr2 import c14_capture as _c14\nfrom vllm_omni.diffusion.models.seedvr2 import sr_telemetry as _sr_metrics\n")
    # Preserve the original .cpu().numpy() expression and its invocation count.
    source = replace_once(source,
        '    return {"payload": {**payload, "video": video.cpu().numpy()}, "metadata": output["metadata"]}',
        '    with _sr_metrics.post_process(output, video) as observation:\n        with observation.stage("device_to_host"):\n            return {"payload": {**payload, "video": video.cpu().numpy()}, "metadata": output["metadata"]}')
    source = wrap(source,
        "                sample = prepare_video(prepared.frames, params.height, params.width, self.device)\n",
        "                generator = params.generator\n", "prepare_video_gpu")
    source = wrap(source, "                if capture is None:\n                    latent = self.vae.encode(sample).sample(generator=generator)\n",
        "                condition = latent.permute", "vae_encode")
    source = wrap(source, "                condition = latent.permute",
        "                velocity = self.transformer", "latent_prepare")
    source = wrap(source, "                velocity = self.transformer",
        "                if capture is not None:\n                    capture.tensor(\"dit_velocity\"", "dit")
    source = wrap(source, "                # Reference Euler returns fp32",
        "                if capture is None:\n                    decoded = self.vae.decode", "solver")
    source = wrap(source, "                if capture is None:\n                    decoded = self.vae.decode",
        "                method = params.extra_args.get", "vae_decode")
    source = wrap(source, "                method = params.extra_args.get",
        "                payload: dict[str, object]", "color_uint8")
    source = replace_once(source,
        '                metadata: dict[str, object] = {"video": {"fps": fps}} if fps is not None else {}\n',
        '                metadata: dict[str, object] = {"video": {"fps": fps}} if fps is not None else {}\n                _sr_metrics.attach_metadata(metadata, observation)\n')
    source = wrap(source, "            with _c14.session(request, self, prepared) as capture:\n", "        return outputs", "pipeline_gpu_window")
    # The outer context owns event resolution and the per-rank file, rather
    # than an additional nested stage that would duplicate the parent timer.
    source = replace_once(source, '            with observation.stage("pipeline_gpu_window"):\n',
        "            with _sr_metrics.window(request, self.device) as observation:\n")
    ast.parse(source)
    return source.encode("utf-8")
