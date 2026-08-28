# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded-cardinality metrics and lifecycle audit events for Video Gateway."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dingo.video_gateway.models import TERMINAL_STATUSES, VideoTask

audit_logger = logging.getLogger("dingo.video_gateway.audit")

_TASK_DURATION_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1200.0,
    1800.0,
)
_ETCD_DURATION_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


def _labels(value: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(item)) for key, item in (value or {}).items()))


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(labels: Sequence[tuple[str, str]]) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{key}="{_escape_label(value)}"' for key, value in labels) + "}"


def _format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return "0"
    return format(value, ".12g")


@dataclass(slots=True)
class _Histogram:
    buckets: tuple[float, ...]
    bucket_counts: list[int]
    count: int = 0
    total: float = 0.0

    @classmethod
    def create(cls, buckets: Sequence[float]) -> _Histogram:
        ordered = tuple(sorted(set(float(bucket) for bucket in buckets)))
        return cls(ordered, [0] * len(ordered))

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        for index, upper in enumerate(self.buckets):
            if value <= upper:
                self.bucket_counts[index] += 1


class GatewayTelemetry:
    """Small in-process Prometheus registry scoped to one Gateway replica."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
        self._histograms: dict[
            tuple[str, tuple[tuple[str, str], ...]], _Histogram
        ] = {}

    def increment(
        self,
        name: str,
        *,
        labels: Mapping[str, str] | None = None,
        amount: int = 1,
    ) -> None:
        if amount < 0:
            raise ValueError("counter increments must not be negative")
        with self._lock:
            self._counters[(name, _labels(labels))] += amount

    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, str] | None = None,
        buckets: Sequence[float],
    ) -> None:
        numeric = max(0.0, float(value))
        key = (name, _labels(labels))
        with self._lock:
            histogram = self._histograms.get(key)
            if histogram is None:
                histogram = _Histogram.create(buckets)
                self._histograms[key] = histogram
            histogram.observe(numeric)

    def record_submission(
        self, pool_id: str, outcome: str, delivery_mode: str
    ) -> None:
        self.increment(
            "dingo_video_task_submissions_total",
            labels={
                "pool": pool_id,
                "outcome": outcome,
                "delivery_mode": delivery_mode,
            },
        )

    def record_stage_duration(self, pool_id: str, stage: str, seconds: float) -> None:
        self.observe(
            "dingo_video_task_stage_duration_seconds",
            seconds,
            labels={"pool": pool_id, "stage": stage},
            buckets=_TASK_DURATION_BUCKETS,
        )

    def record_terminal(self, task: VideoTask) -> None:
        error_code = task.error.code if task.error is not None else "none"
        self.increment(
            "dingo_video_task_terminal_total",
            labels={
                "pool": task.pool_id,
                "status": task.status.value,
                "error_code": error_code,
            },
        )

    def record_etcd_request(
        self, operation: str, seconds: float, *, succeeded: bool
    ) -> None:
        self.observe(
            "dingo_video_etcd_request_duration_seconds",
            seconds,
            labels={"operation": operation},
            buckets=_ETCD_DURATION_BUCKETS,
        )
        if not succeeded:
            self.increment(
                "dingo_video_etcd_request_errors_total",
                labels={"operation": operation},
            )

    def audit_task(
        self,
        event: str,
        task: VideoTask,
        *,
        gateway_generation: str | None = None,
        previous_status: str | None = None,
        revision: int | None = None,
        reason: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "log_type": "video_task_lifecycle",
            "event": event,
            "timestamp_ms": int(time.time() * 1000),
            "task_id": task.id,
            "deployment_id": task.deployment_id,
            "pool_id": task.pool_id,
            "model": task.model,
            "delivery_mode": task.delivery_mode,
            "attempt": task.attempt,
            "status": task.status.value,
        }
        optional = {
            "previous_status": previous_status,
            "gateway_generation": gateway_generation,
            "owner_generation": task.owner_generation,
            "worker_key": task.worker_key,
            "worker_instance_id": task.worker_instance_id,
            "revision": revision,
            "reason": reason,
            "error_code": task.error.code if task.error is not None else None,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        if extra:
            payload.update(extra)
        audit_logger.info(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    def record_transition(
        self,
        event: str,
        before: VideoTask,
        after: VideoTask,
        *,
        gateway_generation: str | None,
        revision: int,
        reason: str | None = None,
    ) -> None:
        self.audit_task(
            event,
            after,
            gateway_generation=gateway_generation,
            previous_status=before.status.value,
            revision=revision,
            reason=reason,
        )
        if before.status != after.status and after.status in TERMINAL_STATUSES:
            self.record_terminal(after)

    def render_prometheus(self) -> list[str]:
        with self._lock:
            counters = list(self._counters.items())
            histograms = [
                (key, _Histogram(
                    value.buckets,
                    list(value.bucket_counts),
                    value.count,
                    value.total,
                ))
                for key, value in self._histograms.items()
            ]

        lines: list[str] = []
        counter_names = sorted({name for (name, _), _value in counters})
        for name in counter_names:
            lines.append(f"# TYPE {name} counter")
            for (metric_name, labels), value in sorted(counters):
                if metric_name == name:
                    lines.append(
                        f"{name}{_format_labels(labels)} {_format_number(value)}"
                    )

        histogram_names = sorted({name for (name, _), _value in histograms})
        for name in histogram_names:
            lines.append(f"# TYPE {name} histogram")
            for (metric_name, labels), histogram in sorted(histograms):
                if metric_name != name:
                    continue
                for upper, count in zip(
                    histogram.buckets, histogram.bucket_counts, strict=True
                ):
                    bucket_labels = tuple(labels) + (("le", _format_number(upper)),)
                    lines.append(
                        f"{name}_bucket{_format_labels(bucket_labels)} {count}"
                    )
                infinite_labels = tuple(labels) + (("le", "+Inf"),)
                lines.append(
                    f"{name}_bucket{_format_labels(infinite_labels)} {histogram.count}"
                )
                lines.append(
                    f"{name}_sum{_format_labels(labels)} "
                    f"{_format_number(histogram.total)}"
                )
                lines.append(
                    f"{name}_count{_format_labels(labels)} {histogram.count}"
                )
        return lines
