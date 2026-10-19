"""Download and hash the exact released Omni wheel before the joint install.

Some uv versions omit archive hashes from direct_url.json. Keep the verified
wheel and a receipt in build-info so later offline audits can rehash the bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urlsplit
from urllib.request import urlopen


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_wheel(policy, directory, *, opener=urlopen, attempts=3):
    url, fragment = urldefrag(policy["omni_wheel"])
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.netloc != "files.pythonhosted.org"
            or parts.query or not re.fullmatch(r"sha256=[0-9a-f]{64}", fragment)):
        raise ValueError("Expected a checksum-pinned official PyPI wheel URL")
    expected = fragment.removeprefix("sha256=")
    filename = parts.path.rsplit("/", 1)[-1]
    if filename != "vllm_omni-0.31.0rc1-py3-none-any.whl":
        raise ValueError("Unexpected released Omni wheel filename")
    directory = Path(directory).resolve()
    wheels = directory / "omni-wheels"
    if wheels.is_symlink():
        raise ValueError("Wheel directory cannot be a symlink")
    wheels.mkdir(parents=True, exist_ok=True)
    wheel = wheels / filename
    if wheel.is_symlink():
        raise ValueError("Wheel file cannot be a symlink")
    if wheel.exists() and file_sha256(wheel) != expected:
        raise ValueError("Existing retained Omni wheel checksum differs; inspect build inputs")
    if not wheel.exists():
        for attempt in range(1, attempts + 1):
            temporary = None
            try:
                print(f"Acquiring checksum-pinned Omni wheel (attempt {attempt}/{attempts})", file=sys.stderr, flush=True)
                with tempfile.NamedTemporaryFile(dir=wheels, prefix="omni-", suffix=".part", delete=False) as output:
                    temporary = Path(output.name)
                    with opener(url, timeout=60) as response:
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            output.write(chunk)
                if file_sha256(temporary) != expected:
                    raise ValueError("Downloaded Omni wheel SHA256 differs from policy")
                temporary.replace(wheel)
                break
            except (URLError, TimeoutError, ConnectionError) as exc:
                if isinstance(exc, HTTPError) and exc.code != 429 and exc.code < 500:
                    raise
                if attempt == attempts:
                    raise
                time.sleep(attempt * 2)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    receipt = {"schema": "h3-verified-omni-wheel/1", "url": url,
               "sha256": expected, "wheel_path": str(wheel), "wheel_uri": wheel.as_uri(),
               "size_bytes": wheel.stat().st_size}
    (directory / "omni-wheel-download.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    receipt = prepare_wheel(policy, args.directory)
    # A local file URI with a #sha256 fragment is not supported by older uv.
    # The retained bytes are checked above and again by the offline auditor.
    print("vllm-omni @ " + receipt["wheel_uri"])


if __name__ == "__main__":
    main()
