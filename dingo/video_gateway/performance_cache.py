# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, process-local performance snapshots; no store, media or network I/O."""

from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

from dingo.video_gateway.config import PerformanceCacheConfig


class PerformanceCache:
    def __init__(
        self,
        config: PerformanceCacheConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.instance_id = uuid.uuid4().hex
        self._clock = clock
        self._lock = threading.Lock()
        self._records: deque[tuple[int, float, bytes]] = deque()
        self._sequence = 0
        self._bytes = 0
        self._evicted_through = 0
        self._expired = 0
        self._capacity_dropped = 0
        self._read_window = clock()
        self._reads = 0

    def admit_read(self) -> bool:
        with self._lock:
            now = self._clock()
            if now - self._read_window >= 1:
                self._read_window, self._reads = now, 0
            self._reads += 1
            return self._reads <= 20

    def _remove(self, expired: bool) -> None:
        seq, _stamp, raw = self._records.popleft()
        self._bytes -= len(raw)
        self._evicted_through = max(self._evicted_through, seq)
        if expired:
            self._expired += 1
        else:
            self._capacity_dropped += 1

    def _expire(self) -> None:
        deadline = self._clock() - self.config.retention_s
        while self._records and self._records[0][1] <= deadline:
            self._remove(expired=True)

    def append(self, event: dict[str, Any]) -> bool:
        # Serialize once to detach from mutable task state and bound record size.
        raw = json.dumps(event, separators=(",", ":"), allow_nan=False).encode()
        with self._lock:
            self._expire()
            self._sequence += 1
            if len(raw) > min(self.config.max_bytes, 256 * 1024):
                self._capacity_dropped += 1
                self._evicted_through = self._sequence
                return False
            while self._records and (
                len(self._records) >= self.config.max_records
                or self._bytes + len(raw) > self.config.max_bytes
            ):
                self._remove(expired=False)
            self._records.append((self._sequence, self._clock(), raw))
            self._bytes += len(raw)
        return True

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._expire()
            return {
                "records": len(self._records),
                "bytes": self._bytes,
                "expired_total": self._expired,
                "capacity_dropped_total": self._capacity_dropped,
            }

    def page(self, cursor: str | None = None, limit: int = 100) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        after, upper, changed = 0, None, False
        if cursor is not None:
            try:
                if not isinstance(cursor, str) or len(cursor) > 1024:
                    raise ValueError
                value = json.loads(
                    base64.b64decode(cursor, altchars=b"-_", validate=True)
                )
                if not isinstance(value, list) or len(value) != 3:
                    raise ValueError
                instance, after, upper = value
                if not isinstance(instance, str) or type(after) is not int or after < 0:
                    raise ValueError
                if upper is not None and (type(upper) is not int or upper < after):
                    raise ValueError
                changed = instance != self.instance_id
                if changed:
                    after, upper = 0, None
            except (ValueError, TypeError, UnicodeError) as exc:
                raise ValueError("invalid performance cursor") from exc
        with self._lock:
            self._expire()
            if after > self._sequence or (upper is not None and upper > self._sequence):
                raise ValueError("cursor is ahead of this instance")
            upper = self._sequence if upper is None else upper
            items = []
            size = 0
            has_more = False
            # Snapshot bytes only; JSON decoding and HTTP serialization run outside the lock.
            for seq, _stamp, raw in self._records:
                if after < seq <= upper:
                    if len(items) >= limit or (items and size + len(raw) > 1024 * 1024):
                        has_more = True
                        break
                    items.append((seq, raw))
                    size += len(raw)
            next_seq = items[-1][0] if has_more else upper
            next_cursor = base64.urlsafe_b64encode(
                json.dumps(
                    [self.instance_id, next_seq, upper if has_more else None],
                    separators=(",", ":"),
                ).encode()
            ).decode()
            metadata = {
                "schema_version": 1,
                "instance_id": self.instance_id,
                "retention_s": self.config.retention_s,
                "oldest_seq": self._records[0][0]
                if self._records
                else self._sequence + 1,
                "latest_seq": self._sequence,
                "next_seq": next_seq,
                "next_cursor": next_cursor,
                "has_more": has_more,
                "gap_detected": after < min(upper, self._evicted_through),
                "instance_changed": changed,
            }
        metadata["records"] = [
            {"seq": seq, "event": json.loads(raw)} for seq, raw in items
        ]
        return metadata
