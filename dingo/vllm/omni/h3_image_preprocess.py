# SPDX-License-Identifier: Apache-2.0
"""H3-only CPU image sizing. Standalone child imports no model/CUDA runtime."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

PROFILE = "fit_1080p_v1"
MAX_DERIVED_BYTES = 8 * 1024**2


async def kill_and_wait(process):
    """Do not abandon a CPU child even if its owner is cancelled repeatedly."""
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    waiter = asyncio.create_task(process.wait())
    interrupted = False
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            interrupted = True
    waiter.result()
    if interrupted:
        raise asyncio.CancelledError


def fit_size(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("invalid image dimensions")
    scale = min(1.0, 1920 / max(width, height), 1080 / min(width, height))
    return (
        max(1, math.floor(width * scale + 0.5)),
        max(1, math.floor(height * scale + 0.5)),
    )


def fit_image(image):
    from PIL import Image

    width, height = image.size
    if not (
        256 <= min(width, height)
        and max(width, height) <= 5760
        and 0.4 <= width / height <= 2.5
    ):
        raise ValueError("invalid H3 reference image dimensions")
    target = fit_size(*image.size)
    if target == image.size:
        return image
    # Match the existing ImageLoader: RGB decoded pixels, no EXIF rotation.
    # Rotating here would disagree with Gateway's original canvas calculation.
    result = image.resize(target, Image.Resampling.LANCZOS)
    return result


class SizingLoader:
    """Bound legacy inline decode+resize before collecting multiple images."""

    def __init__(self, loader, semaphore):
        self.loader, self.semaphore = loader, semaphore

    def __getattr__(self, key):
        return getattr(self.loader, key)

    async def load_image(self, reference):
        async def load():
            image = await self.loader.load_image(reference)
            if not isinstance(reference, str) or not reference.lstrip().startswith(
                "data:"
            ):
                # HTTP ImageLoader may return a shared cache object. Own a
                # copy before resizing/closing; never invalidate that cache.
                image = await asyncio.to_thread(image.copy)
            try:
                fitted = await asyncio.to_thread(fit_image, image)
            except BaseException:
                image.close()
                raise
            if fitted is not image:
                logging.getLogger(__name__).info(
                    "h3_inline_image_resized profile=%s source=%s target=%s",
                    PROFILE,
                    image.size,
                    fitted.size,
                )
                image.close()
            return fitted

        async with self.semaphore:
            operation = asyncio.create_task(load())
            try:
                return await asyncio.shield(operation)
            except asyncio.CancelledError:
                # Decode/resize may still own a thread. Do not release the CPU
                # permit or image resources while that work is running.
                while not operation.done():
                    try:
                        await asyncio.shield(operation)
                    except asyncio.CancelledError:
                        continue
                    except Exception:  # noqa: BLE001 - preserve caller cancellation
                        break
                if not operation.cancelled() and operation.exception() is None:
                    operation.result().close()
                raise


def _prepare_one(root, entry):
    from PIL import Image

    source = root / str(entry["ordinal"])
    stamp = time.monotonic()
    with Image.open(source, formats=["JPEG", "PNG", "WEBP"]) as opened:
        size = opened.size
        if not (
            256 <= min(size) and max(size) <= 5760 and 0.4 <= size[0] / size[1] <= 2.5
        ):
            raise ValueError("invalid H3 reference image dimensions")
        target = fit_size(*size)
        row = {
            "ordinal": entry["ordinal"],
            "source_size": list(size),
            "target_size": list(target),
            "source_sha256": entry["sha256"],
            "resized": target != size,
        }
        if target != size:
            image = opened.convert("RGB")
            try:
                fitted = fit_image(image)
                try:
                    name = f"image-{entry['ordinal']}.png"
                    destination = root / name
                    temporary = destination.with_suffix(".part")
                    with temporary.open("xb") as stream:
                        fitted.save(stream, format="PNG", compress_level=1)
                    length = temporary.stat().st_size
                    if length > MAX_DERIVED_BYTES:
                        raise ValueError("prepared image exceeds byte limit")
                    row.update(
                        path=name,
                        bytes=length,
                        sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),
                    )
                    os.replace(temporary, destination)
                finally:
                    if fitted is not image:
                        fitted.close()
            finally:
                image.close()
    row["elapsed_s"] = time.monotonic() - stamp
    return row


@asynccontextmanager
async def group_permits(slots, count, group_lock):
    # Serialize bulk acquisition, including partial acquisitions. Legacy inline
    # operations acquire one permit each and share this same global allowance.
    async with group_lock:
        held = 0
        try:
            for _ in range(count):
                await slots.acquire()
                held += 1
            yield
        finally:
            for _ in range(held):
                slots.release()


def prepare_images(root, entries, parallelism=1):
    if type(parallelism) is not int or not 1 <= parallelism <= 3:
        raise ValueError("image parallelism must be in [1, 3]")
    root = Path(root)
    images = [e for e in entries if e["content_type"].startswith("image/")]
    started = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=parallelism)
    try:
        # map preserves reference order even if decoding finishes out of order.
        rows = list(executor.map(lambda entry: _prepare_one(root, entry), images))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    result = {
        "profile": PROFILE,
        "parallelism": parallelism,
        "images": rows,
        "elapsed_s": time.monotonic() - started,
        "derived_bytes": sum(r.get("bytes", 0) for r in rows),
        # Each shared request owns a fresh child process. Linux ru_maxrss is KiB.
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    temporary = root / "image-preprocess.part"
    temporary.write_text(json.dumps(result))
    os.replace(temporary, root / "image-preprocess.json")
    return result


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(2)
    prepare_images(
        sys.argv[1],
        json.loads(sys.stdin.buffer.read(256 * 1024 + 1)),
        int(sys.argv[2]) if len(sys.argv) == 3 else 1,
    )
