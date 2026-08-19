# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free monotonic ULIDs used for externally visible video IDs."""

from __future__ import annotations

import secrets
import threading
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_RANDOM_MASK = (1 << 80) - 1


class MonotonicULIDGenerator:
    """Generate lexicographically sortable ULIDs within one process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_ms = -1
        self._last_random = 0

    def new(self, *, now_ms: int | None = None) -> str:
        timestamp_ms = int(time.time() * 1000) if now_ms is None else now_ms
        if timestamp_ms < 0 or timestamp_ms >= 1 << 48:
            raise ValueError("ULID timestamp must fit in 48 bits")

        with self._lock:
            if timestamp_ms > self._last_ms:
                self._last_ms = timestamp_ms
                self._last_random = int.from_bytes(secrets.token_bytes(10), "big")
            else:
                timestamp_ms = self._last_ms
                self._last_random = (self._last_random + 1) & _RANDOM_MASK
                if self._last_random == 0:
                    self._last_ms += 1
                    timestamp_ms = self._last_ms

            value = (timestamp_ms << 80) | self._last_random

        encoded = ["0"] * 26
        for index in range(25, -1, -1):
            encoded[index] = _ALPHABET[value & 31]
            value >>= 5
        return "video-" + "".join(encoded)


default_generator = MonotonicULIDGenerator()


def new_video_id() -> str:
    return default_generator.new()
