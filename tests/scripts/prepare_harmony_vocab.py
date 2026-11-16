#!/usr/bin/env python3
"""Prepare the hash-pinned Harmony vocabulary before isolated Rust tests run."""

import hashlib
import os
from pathlib import Path
import sys
import tempfile
from urllib.request import urlopen

VOCAB_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
VOCAB_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
VOCAB_CACHE_KEY = "fb374d419588a4632f3f557e76b4b70aebbca790"


def prepare(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / VOCAB_CACHE_KEY
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == VOCAB_SHA256:
        return target
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".o200k-")
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as output, urlopen(VOCAB_URL, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != VOCAB_SHA256:
            raise ValueError("Harmony vocabulary SHA-256 mismatch")
        # Concurrent preparations publish only complete, verified files.
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


if __name__ == "__main__":
    print(prepare(sys.argv[1]))
