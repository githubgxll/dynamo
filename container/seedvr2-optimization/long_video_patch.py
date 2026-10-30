# SPDX-License-Identifier: Apache-2.0
"""Exact-anchor delta over the already C14-patched Omni 0.31 long-video source."""
from __future__ import annotations


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Pinned long-video anchor is absent or ambiguous: " + before[:100])
    return source.replace(before, after, 1)


RESTORE = '''def _restore(
    frames: list[av.VideoFrame], width: int, height: int, seed: int,
    color_correction_method: str, port: int, authorization: str,
    c14_context=None, measurement=None, metrics=None,
) -> list[np.ndarray]:
    metrics = metrics or _support.NullMetrics()
    measurement = measurement or {"window_index": 0, "measurement_mode": "off"}
    with metrics.stage("window_input_encode", window_index=measurement["window_index"]):
        segment = _segment(frames)
    _c14.media(c14_context, "window-input.mkv", segment)
    options = _c14.child_options(c14_context, {
        "video_codec_options": LOSSLESS, "color_correction_method": color_correction_method})
    if measurement["measurement_mode"] != "off":
        options = {**options, "seedvr2_measurement": measurement}
    with metrics.stage("window_model_http", window_index=measurement["window_index"]):
        response = requests.post(
            f"http://127.0.0.1:{port}/v1/videos/sync",
            data={"prompt": " ", "size": f"{width}x{height}", "num_frames": str(len(frames)),
                  "num_inference_steps": "1", "guidance_scale": "1", "seed": str(seed),
                  "extra_params": json.dumps(options)},
            files={"input_references": ("window.mkv", segment, "video/x-matroska")},
            headers={"Authorization": authorization} if authorization else {}, timeout=900)
    try:
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            raise JobError("SeedVR2 window restoration failed") from error
        _c14.media(c14_context, "window-output.mp4", response.content)
        with metrics.stage("window_output_decode", window_index=measurement["window_index"]):
            with av.open(io.BytesIO(response.content)) as container:
                video = container.streams.video[0]
                decoded = list(container.decode(video=0))
                if (video.width, video.height, video.average_rate, len(decoded)) != (width, height, Fraction(FPS), len(frames)):
                    raise JobError("SeedVR2 window restoration returned an unexpected geometry or frame count")
                if [frame.pts * frame.time_base for frame in decoded] != [Fraction(index, FPS) for index in range(len(frames))]:
                    raise JobError("SeedVR2 window restoration returned invalid timestamps")
                return [frame.to_ndarray(format="rgb24") for frame in decoded]
    finally:
        response.close()


'''


def transform(data):
    source = data.decode("utf-8")
    source = once(source, "from vllm_omni.diffusion.models.seedvr2 import config as envs\n",
                  "from vllm_omni.diffusion.models.seedvr2 import long_video_support as _support\n"
                  "from vllm_omni.diffusion.models.seedvr2 import config as envs\n")
    # C14 hooks are a required parent contract, not silently reapplied here.
    if "c14_context=None" not in source or "_c14.fused_frame(job, written, array)" not in source:
        raise ValueError("Expected the existing C14 long-video patch")
    first, last = source.index("def _restore(\n"), source.index("def _run(\n")
    source = source[:first] + RESTORE + source[last:]
    # Extend only the three private top-level execution functions; compatibility
    # defaults preserve old direct callers used by inherited diagnostics.
    signature = "    authorization: str,\n) -> None:"
    if source.count(signature) != 3:
        raise ValueError("Expected exactly _run, _restore_to, _background signatures")
    source = source.replace(signature, "    authorization: str,\n    optimization=None,\n) -> None:")
    source = once(source, "    window = _window(width, height)\n",
                  "    optimization = optimization or _support.options()\n"
                  "    window = _support.select_window(_window(width, height), optimization[\"window_frames\"])\n")
    source = once(source,
                  "            job, Path(scratch), window, width, height, target, loop, seed, color_correction_method, port, authorization\n",
                  "            job, Path(scratch), window, width, height, target, loop, seed, color_correction_method, port, authorization, optimization\n")
    source = once(source, '    source = job / "input.mp4"\n',
                  '    optimization = optimization or _support.options()\n'
                  '    metrics = _support.ServerMetrics(job, optimization, width, height, target, window)\n'
                  '    writer_spec = _support.writer_spec(optimization, width, height)\n'
                  '    delivery_width, delivery_height = writer_spec["width"], writer_spec["height"]\n'
                  '    source = job / "input.mp4"\n')
    source = once(source, '    source_frames = _frames(source, width, height, loop)\n',
                  '    source_frames = _frames(source, width, height, loop)\n'
                  '    def take(count):\n'
                  '        with metrics.stage("source_decode"):\n'
                  '            return list(itertools.islice(source_frames, count))\n')
    source = once(source, "        batch = list(itertools.islice(source_frames, min(window, target)))\n",
                  "        batch = take(min(window, target))\n")
    source = once(source,
                  "            batch = batch[-OVERLAP:] + list(itertools.islice(source_frames, min(window, target - start) - OVERLAP))\n",
                  "            batch = batch[-OVERLAP:] + take(min(window, target - start) - OVERLAP)\n")
    source = once(source,
                  "        return pool.submit(_restore, frames, width, height, seed, color_correction_method, port, authorization, context)",
                  "        measurement = metrics.window(len(metrics.windows), start, len(frames))\n"
                  "        return pool.submit(_restore, frames, width, height, seed, color_correction_method, port, authorization, context, measurement, metrics)")
    source = once(source, '        video.width, video.height, video.pix_fmt = width, height, "yuv420p"\n',
                  '        video.width, video.height, video.pix_fmt = delivery_width, delivery_height, "yuv420p"\n')
    source = once(source,
                  '        video.options = {"preset": "veryfast", "crf": "18", "tune": "zerolatency", "bf": "0"}\n',
                  '        video.options = writer_spec["options"]\n'
                  '        if optimization["encoding_mode"] == "single_encode":\n'
                  '            video.codec_context.sample_aspect_ratio = Fraction(1, 1)\n'
                  '        writer_spec["av_library_versions"] = {name: list(version) for name, version in av.library_versions.items()}\n')
    source = once(source,
                  '            frame = av.VideoFrame.from_ndarray(array, format="rgb24")\n'
                  '            frame.pts, frame.time_base = written, Fraction(1, FPS)\n'
                  '            for packet in video.encode(frame):\n'
                  '                container.mux(packet)\n',
                  '            frame = av.VideoFrame.from_ndarray(array, format="rgb24")\n'
                  '            if (width, height) != (delivery_width, delivery_height):\n'
                  '                with metrics.stage("delivery_resize"):\n'
                  '                    frame = frame.reformat(width=delivery_width, height=delivery_height, format="rgb24", interpolation="LANCZOS")\n'
                  '            frame.pts, frame.time_base = written, Fraction(1, FPS)\n'
                  '            with metrics.stage("final_video_encode"):\n'
                  '                for packet in video.encode(frame):\n'
                  '                    container.mux(packet)\n')
    source = once(source, "        while in_flight:\n",
                  "        while in_flight:\n"
                  "            if (job / \"cancel\").exists():\n"
                  "                raise JobCancelledError(\"SeedVR2 long-video job was cancelled\")\n")
    source = once(source,
                  "                    write(np.rint(pending[offset] * (1 - newer) + restored[offset] * newer).astype(np.uint8))\n",
                  '                    with metrics.stage("overlap_blend"):\n'
                  '                        blended = np.rint(pending[offset] * (1 - newer) + restored[offset] * newer).astype(np.uint8)\n'
                  '                    write(blended)\n                    del blended\n')
    source = once(source,
                  "        for packet in video.encode():\n            container.mux(packet)\n    if written != target:",
                  '        with metrics.stage("final_video_encode", flush=True):\n'
                  '            for packet in video.encode():\n'
                  '                container.mux(packet)\n'
                  '            container.close()\n'
                  '    source_frames.close()\n'
                  '    if (job / "cancel").exists():\n'
                  '        raise JobCancelledError("SeedVR2 long-video job was cancelled")\n'
                  '    if written != target:')
    # Keep the existing audio strategy. Final delivery still copies the original
    # Base audio in the external controller; this API mux is not that stage.
    source = once(source, '        _c14.run_mux(job, subprocess.run,\n',
                  '        with metrics.stage("audio_mux", audio_mode="aac_reencode_inherited_service"):\n'
                  '            _c14.run_mux(job, subprocess.run,\n')
    # Indent the existing mux call body by one extra level after wrapping.
    mux_start = source.index('            _c14.run_mux(job, subprocess.run,\n')
    mux_end = source.index('    else:\n        video_path.replace(output)', mux_start)
    mux = source[mux_start:mux_end].splitlines(keepends=True)
    source = source[:mux_start] + mux[0] + "".join("    " + line if line.strip() else line for line in mux[1:]) + source[mux_end:]
    source = once(source,
                  '        if (video.width, video.height, video.average_rate, decoded) != (width, height, Fraction(FPS), target):',
                  '        if (video.width, video.height, video.average_rate, decoded) != (delivery_width, delivery_height, Fraction(FPS), target):')
    source = once(source,
                  '        if has_audio and not container.streams.audio:\n',
                  '        writer_spec["effective_output"] = {"codec": video.codec_context.name,\n'
                  '            "profile": video.codec_context.profile, "has_b_frames": video.codec_context.has_b_frames,\n'
                  '            "pixel_format": video.codec_context.format.name if video.codec_context.format else None,\n'
                  '            "sample_aspect_ratio": str(video.sample_aspect_ratio),\n'
                  '            "tune": "not_serialized_in_h264; declared option/default policy recorded separately"}\n'
                  '        if optimization["encoding_mode"] == "single_encode" and (\n'
                  '            video.codec_context.name != "h264"\n'
                  '            or video.codec_context.format is None or video.codec_context.format.name != "yuv420p"\n'
                  '            or video.sample_aspect_ratio is None or Fraction(str(video.sample_aspect_ratio)) != 1):\n'
                  '            raise JobError("Single-encode output requires h264/yuv420p with square pixels")\n'
                  '        if has_audio and not container.streams.audio:\n')
    validate_start = source.index('    with av.open(str(output)) as container:\n')
    validate_end = source.index('    with output.open("rb") as content:', validate_start)
    validation = source[validate_start:validate_end]
    source = source[:validate_start] + '    with metrics.stage("media_validate"):\n' + "".join(
        "    " + line if line.strip() else line for line in validation.splitlines(keepends=True)) + source[validate_end:]
    before = '''    (job / "result.json").write_text(
        json.dumps({"frames": target, "width": width, "height": height, "fps": FPS, "sha256": digest}) + "\\n"
    )'''
    after = '''    _support.atomic_json(job / "result.json", {
        "frames": target, "width": delivery_width, "height": delivery_height, "fps": FPS, "sha256": digest,
        "schema": "seedvr2-long-optimization/1", "encoding_mode": optimization["encoding_mode"],
        "model_size": {"width": width, "height": height},
        "delivery_size": {"width": delivery_width, "height": delivery_height},
        "writer": writer_spec, "window_count": len(metrics.windows),
        "configured_window": optimization["window_frames"], "selected_window": window,
        "window_map": metrics.windows,
        "measurement_mode": optimization["measurement_mode"], "sr_request_id": job.name,
        "service_lossy_video_encode_count": 1,
        "final_lossy_video_encode_count": 1 if optimization["encoding_mode"] == "single_encode" else None,
        "external_lossy_encode_required": optimization["encoding_mode"] == "legacy",
        "window_input_lossless_encode_count": len(metrics.windows),
        "window_output_lossless_encode_count": len(metrics.windows),
        "original_audio_copy_required_in_controller": True})'''
    source = once(source, before, after)
    source = once(source, '    _status(job, "completed", target)\n',
                  '    _status(job, "completed", target)\n    metrics.finish("completed")\n')
    publish_start = source.index('    with output.open("rb") as content:', source.index('def _restore_to('))
    publish_end = source.index('    metrics.finish("completed")\n', publish_start)
    publish = source[publish_start:publish_end]
    source = source[:publish_start] + '    with metrics.stage("artifact_publish"):\n' + "".join(
        "    " + line if line.strip() else line for line in publish.splitlines(keepends=True)) + source[publish_end:]
    # Ensure decoder generator closes and metrics survive every failure. The
    # executor's context exit waits all already-submitted futures before return.
    body_start = source.index('    written = 0\n', source.index('def _restore_to('))
    body_end = source.index('\n\ndef _background(', body_start)
    body = source[body_start:body_end]
    source = source[:body_start] + '    try:\n' + "".join("    " + line if line.strip() else line for line in body.splitlines(keepends=True)) + '''    except BaseException as error:
        metrics.finish("cancelled" if isinstance(error, JobCancelledError) else "failed", type(error).__name__)
        raise
    finally:
        source_frames.close()
''' + source[body_end:]
    source = once(source,
                  "        _run(job, width, height, target, loop, seed, color_correction_method, port, authorization)\n",
                  "        _run(job, width, height, target, loop, seed, color_correction_method, port, authorization, optimization)\n")
    source = once(source, '    extra_params: str | None = Form(None),\n',
                  '    extra_params: str | None = Form(None),\n'
                  '    encoding_mode: str = Form("legacy"),\n'
                  '    delivery_size: str | None = Form(None),\n'
                  '    window_frames: int | None = Form(None),\n'
                  '    measurement_mode: str = Form("off"),\n'
                  '    measurement_context: str | None = Form(None),\n')
    source = once(source,
                  '    if _window(width, height) <= OVERLAP:\n'
                  '        raise HTTPException(400, "SeedVR2 long video output exceeds the configured SeedVR2 frame or clip budget")\n',
                  '    try:\n'
                  '        optimization = _support.options(encoding_mode, delivery_size, window_frames, measurement_mode, measurement_context)\n'
                  '        _support.select_window(_window(width, height), window_frames)\n'
                  '    except ValueError as error:\n'
                  '        raise HTTPException(400, str(error)) from error\n')
    source = once(source,
                  '            await asyncio.to_thread(_store_upload, input_references, job / "input.mp4")\n',
                  '            await asyncio.to_thread(_store_upload, input_references, job / "input.mp4")\n'
                  '            optimization["server_started_ns"] = time.perf_counter_ns()\n'
                  '            if optimization["encoding_mode"] == "single_encode":\n'
                  '                try:\n'
                  '                    with av.open(str(job / "input.mp4")) as source_container:\n'
                  '                        source_video = source_container.streams.video[0]\n'
                  '                        _support.validate_source(optimization, source_video.width, source_video.height, width, height)\n'
                  '                except Exception as error:\n'
                  '                    raise HTTPException(400, "Source/delivery geometry rejected: " + str(error)) from error\n'
                  '            _support.atomic_json(job / "optimization.json", optimization)\n')
    source = once(source,
                  '                raw_request.headers.get("authorization", ""),\n',
                  '                raw_request.headers.get("authorization", ""),\n                optimization,\n')
    source = once(source,
                  '    return {"id": job_id, "status": "queued", "max_frames": MAX_FRAMES}\n',
                  '    return {"id": job_id, "status": "queued", "max_frames": MAX_FRAMES,\n'
                  '            "sr_request_id": job_id, "encoding_mode": encoding_mode,\n'
                  '            "measurement_schema": _support.SCHEMA}\n')
    # Small JSON retrieval avoids downloading media for performance-only runs.
    source = once(source, '\n\n@router.delete("/v1/seedvr2/restore-long/{job_id}", status_code=202)', '''

@router.get("/v1/seedvr2/restore-long/{job_id}/metrics")
async def get_long_video_metrics(job_id: str):
    _job_dir(job_id)  # Existing strict UUID validation.
    try:
        return await asyncio.to_thread(_support.metrics_bundle, job_id)
    except FileNotFoundError as error:
        raise HTTPException(404, "SeedVR2 measurement record is not ready") from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@router.get("/v1/seedvr2/restore-long/{job_id}/manifest")
async def get_long_video_manifest(job_id: str):
    result = _job_dir(job_id) / "result.json"
    if not result.is_file():
        raise HTTPException(404, "SeedVR2 long-video output is not ready")
    return json.loads(result.read_text(encoding="utf-8"))


@router.delete("/v1/seedvr2/restore-long/{job_id}", status_code=202)''')
    compile(source, "long_video.py", "exec")
    return source.encode("utf-8")
