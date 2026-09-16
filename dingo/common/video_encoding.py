"""Optional bounded Omni frame conversion; no GPU dependencies at import time."""

import inspect
import os
import threading


def frame_conversion_workers() -> int:
    """Deployment-only control; eight workers matches the native Omni service."""
    try:
        value = int(os.environ.get("DINGO_VIDEO_FRAME_CONVERSION_WORKERS", "8"))
    except ValueError as exc:
        raise ValueError(
            "DINGO_VIDEO_FRAME_CONVERSION_WORKERS must be an integer from 1 to 16"
        ) from exc
    if not 1 <= value <= 16:
        raise ValueError(
            "DINGO_VIDEO_FRAME_CONVERSION_WORKERS must be an integer from 1 to 16"
        )
    return value


class VideoEncoder:
    """One reusable native converter per formatter, shared across requests.

    Shutdown fences new work and waits for active encodes before closing the
    converter. This also covers encode threads finishing after task cancellation.
    """

    def __init__(self, workers: int):
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError("frame conversion workers must be an integer from 1 to 16")
        self.workers = workers
        self._condition = threading.Condition()
        self._active = 0
        self._closed = False
        self._initialized = False
        self._converter = None

    def encode(self, api, video, **kwargs) -> bytes:
        with self._condition:
            if self._closed:
                raise RuntimeError("video encoder is closed")
            if not self._initialized:
                factory = getattr(api, "_PlanarFrameConverter", None)
                try:
                    supported = (
                        "frame_converter"
                        in inspect.signature(api._encode_video_bytes).parameters
                    )
                except (TypeError, ValueError):
                    supported = False
                if self.workers > 1 and factory is not None and supported:
                    self._converter = factory(max_workers=self.workers)
                self._initialized = True
            converter = self._converter
            self._active += 1
        try:
            if converter is not None:
                return api._encode_video_bytes(
                    video, frame_converter=converter, **kwargs
                )
            return api._encode_video_bytes(video, **kwargs)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            while self._active:
                self._condition.wait()
            converter, self._converter = self._converter, None
        if converter is not None:
            converter.shutdown()
