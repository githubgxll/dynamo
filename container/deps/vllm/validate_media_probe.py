"""CPU-only smoke for the upstream FFmpeg path required by Ref2VA."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    for executable in ("ffmpeg", "ffprobe"):
        path = shutil.which(executable)
        if not path:
            raise RuntimeError(f"Missing upstream {executable}")
        if Path(path).resolve().is_relative_to(Path("/usr/local")):
            raise RuntimeError(
                f"Expected upstream {executable}, found in-tree replacement {path!r}"
            )
        subprocess.run([path, "-version"], check=True, capture_output=True, timeout=15)

    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    if "libx264rgb" not in encoders:
        raise RuntimeError("Upstream FFmpeg is missing the libx264rgb encoder")

    with tempfile.TemporaryDirectory(prefix="dingo-media-probe-") as directory:
        directory_path = Path(directory)
        source = directory_path / "source.rgb"
        encoded = directory_path / "prepared.mp4"
        decoded = directory_path / "decoded.rgb"
        frame = bytes((index % 251 for index in range(32 * 32 * 3)))
        source.write_bytes(frame * 4)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "32x32",
                "-r",
                "8",
                "-i",
                str(source),
                "-frames:v",
                "4",
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "rgb24",
                str(encoded),
            ],
            check=True,
            timeout=30,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(encoded),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                str(decoded),
            ],
            check=True,
            timeout=30,
        )
        if decoded.read_bytes() != source.read_bytes():
            raise RuntimeError("libx264rgb lossless round trip changed RGB pixels")
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
                str(encoded),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        document = json.loads(result.stdout)
        streams = {stream["codec_type"]: stream for stream in document["streams"]}
        assert streams["video"]["codec_name"] == "h264"
        assert streams["video"]["pix_fmt"] == "gbrp"
        assert (streams["video"]["width"], streams["video"]["height"]) == (32, 32)
        assert int(streams["video"]["nb_read_frames"]) == 4
        assert float(document["format"]["duration"]) > 0
    print(
        "DINGO_MEDIA_PROBE=PASS (upstream libx264rgb + rawvideo + ffprobe)"
    )


if __name__ == "__main__":
    main()
