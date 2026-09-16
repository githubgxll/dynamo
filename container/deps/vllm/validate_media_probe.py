"""CPU-only FFmpeg/ffprobe smoke after replacing the upstream media packages."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import av


def main():
    for executable in ("ffmpeg", "ffprobe"):
        path = shutil.which(executable)
        if path != "/usr/local/bin/" + executable:
            raise RuntimeError(f"Expected in-tree {executable}, found {path!r}")
        subprocess.run([path, "-version"], check=True, capture_output=True, timeout=15)

    with tempfile.TemporaryDirectory(prefix="dingo-media-probe-") as directory:
        path = Path(directory) / "sample.mp4"
        with av.open(str(path), "w", format="mp4") as output:
            video = output.add_stream("libx264", rate=8)
            video.width, video.height, video.pix_fmt = 32, 32, "yuv420p"
            audio = output.add_stream("aac", rate=48000)
            audio.layout = "stereo"
            for index in range(4):
                frame = av.VideoFrame(32, 32, "yuv420p")
                frame.pts = index
                for plane in frame.planes:
                    plane.update(bytes(plane.buffer_size))
                for packet in video.encode(frame):
                    output.mux(packet)
            for packet in video.encode(None):
                output.mux(packet)
            frame = av.AudioFrame(format="fltp", layout="stereo", samples=24000)
            frame.sample_rate, frame.pts = 48000, 0
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            for packet in audio.encode(frame):
                output.mux(packet)
            for packet in audio.encode(None):
                output.mux(packet)
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        document = json.loads(result.stdout)
        streams = {stream["codec_type"]: stream for stream in document["streams"]}
        assert streams["video"]["codec_name"] == "h264"
        assert (streams["video"]["width"], streams["video"]["height"]) == (32, 32)
        assert int(streams["video"]["nb_read_frames"]) == 4
        assert streams["audio"]["codec_name"] == "aac"
        assert int(streams["audio"]["sample_rate"]) == 48000
        assert int(streams["audio"]["channels"]) == 2
        assert float(document["format"]["duration"]) > 0
    print(
        "DINGO_MEDIA_PROBE=PASS (H.264 video + stereo AAC, ffprobe JSON and frame count)"
    )


if __name__ == "__main__":
    main()
