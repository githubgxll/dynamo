"""Exercise the actual Omni CPU mux, cv2 decode and framework imports without downloads."""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(os.environ.get("H3_BUILD_INFO", "/opt/dynamo/build-info")))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import av
    import cv2
    import numpy as np
    import soxr
    import torch
    import torchcodec
    from torchcodec.decoders import VideoDecoder
    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
    from vllm_omni.entrypoints.openai import video_api_utils
    from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
    from dingo.common.video_encoding import VideoEncoder
    from transformers import AutoTokenizer, UMT5EncoderModel
    # Import checks intentionally use real Linux packages; this script is not run
    # as a simulated local Windows acceptance test.
    _ = (soxr, torchcodec, AutoTokenizer, UMT5EncoderModel)
    frames = np.zeros((8, 32, 32, 3), dtype=np.uint8)
    frames[:, :, :, 1] = 120
    fps, sample_rate = 24, 32000
    waveform = np.zeros((2, round(sample_rate * len(frames) / fps)), dtype=np.float32)
    payload = mux_video_audio_bytes(frames, waveform, fps=float(fps), audio_sample_rate=sample_rate)
    # Exercise the colleague service's real 0.31 encoder integration as well as
    # native Omni muxing. No engine, GPU, model, URL or external download is used.
    normalized = OmniOpenAIServingVideo._normalize_video_outputs(list(frames))
    assert len(normalized) == 1, "Gateway video normalization changed"
    encoder = VideoEncoder(2)
    try:
        gateway_payload = encoder.encode(video_api_utils, normalized[0], fps=fps,
                                         audio=waveform, audio_sample_rate=sample_rate,
                                         video_codec_options={"preset": "ultrafast", "threads": "0"})
    finally:
        encoder.close()
    with av.open(io.BytesIO(gateway_payload), mode="r") as container:
        assert container.streams.video[0].codec_context.name == "h264"
        assert container.streams.audio[0].codec_context.name == "aac"
    with av.open(io.BytesIO(payload), mode="r") as container:
        video, audio = container.streams.video[0], container.streams.audio[0]
        assert video.codec_context.name == "h264", video.codec_context.name
        assert audio.codec_context.name == "aac", audio.codec_context.name
        assert int(video.average_rate) == fps
        assert audio.codec_context.sample_rate == sample_rate
        assert abs(float(video.duration * video.time_base) - float(audio.duration * audio.time_base)) <= 1 / fps
    with tempfile.TemporaryDirectory(prefix="h3-media-smoke-") as directory:
        clip = Path(directory) / "mux.mp4"
        clip.write_bytes(payload)
        cap = cv2.VideoCapture(str(clip))
        try:
            ok, frame = cap.read()
            assert ok and frame.shape == (32, 32, 3), "cv2 cannot decode the generated MP4"
        finally:
            cap.release()
        decoded = VideoDecoder(str(clip), device="cpu")[0]
        assert tuple(decoded.shape) == (3, 32, 32), "TorchCodec CPU cannot decode the generated MP4"
        result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(clip)],
                                capture_output=True, text=True, check=True)
        streams = json.loads(result.stdout)["streams"]
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    report = {"status": "PASS_REAL_CPU_MEDIA_SMOKE", "torch": torch.__version__,
              "mux_bytes": len(payload), "ffmpeg": ffmpeg,
              "gateway_encoder_mux_bytes": len(gateway_payload), "gateway_encoder_cpu_smoke": True,
              "streams": [{k: s.get(k) for k in ("codec_type", "codec_name", "width", "height", "sample_rate")} for s in streams],
              "cv2_mp4_decode": True, "torchcodec_cpu_mp4_decode": True,
              "scope": "Synthetic CPU H.264/AAC encode/decode and imports; no model weights, GPU kernels or perceptual quality acceptance"}
    (args.output_dir / "media-smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
