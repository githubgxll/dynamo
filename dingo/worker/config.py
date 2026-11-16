# SPDX-License-Identifier: Apache-2.0
"""Optional finite-worker settings, validated against the bound backend policy."""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class WorkerSettings:
    adapter: str
    execution_capacity: int
    prefetch_capacity: int


def load_worker_settings(
    path,
    *,
    adapter,
    execution_capacity,
    prefetch_capacity=0,
    max_execution_capacity=None,
    max_prefetch_capacity=0,
):
    values = {}
    if path:
        with Path(path).open("rb") as source:
            raw = source.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise ValueError("worker config exceeds 64 KiB")
        document = yaml.safe_load(raw)
        if not isinstance(document, dict) or set(document) != {"worker"}:
            raise ValueError("worker config requires only the worker section")
        values = document["worker"]
        allowed = {"adapter", "execution_capacity", "prefetch_capacity"}
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError("unsupported worker configuration field")
    selected = values.get("adapter", adapter)
    if selected != adapter:
        raise ValueError("configured worker adapter does not match the entry point")
    capacity = values.get("execution_capacity", execution_capacity)
    prefetch = values.get("prefetch_capacity", prefetch_capacity)
    if type(capacity) is not int or capacity < 1:
        raise ValueError("execution capacity must be positive")
    if max_execution_capacity is not None and capacity > max_execution_capacity:
        raise ValueError("execution capacity exceeds backend capacity")
    if type(prefetch) is not int or not 0 <= prefetch <= max_prefetch_capacity:
        raise ValueError("prefetch capacity is unsupported by the bound worker policy")
    return WorkerSettings(selected, capacity, prefetch)
