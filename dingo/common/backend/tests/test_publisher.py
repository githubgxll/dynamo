# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Optional

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
)

from dynamo._core import Context

from dingo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dingo.common.backend.publisher import ComponentSnapshot, PushSource, ZmqSource
from dingo.common.constants import DisaggregationMode

pytestmark = [
    pytest.mark.unit,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

import json


def test_source_descriptors_carry_payload_and_defaults():
    zmq = ZmqSource(endpoint="tcp://127.0.0.1:5557")
    assert (zmq.endpoint, zmq.topic, zmq.dp_rank) == ("tcp://127.0.0.1:5557", "", 0)

    seen: list[object] = []
    push = PushSource(on_ready=seen.append, dp_rank=2)
    push.on_ready("publisher")
    assert seen == ["publisher"]
    assert push.dp_rank == 2

    # ComponentSnapshot is the payload engines push into SnapshotPublisher.
    snap = ComponentSnapshot(
        kv_used_blocks=42, kv_total_blocks=100, gpu_cache_usage=0.42, dp_rank=0
    )
    assert snap.dp_rank == 0
    assert snap.kv_cache_hit_rate is None


class _MinimalEngine(LLMEngine):
    @classmethod
    async def from_args(cls, argv: Optional[list[str]] = None):
        raise NotImplementedError

    async def start(self, worker_id: int) -> EngineConfig:
        return EngineConfig(model="minimal")

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        yield {"token_ids": [], "index": 0, "finish_reason": "stop"}

    async def cleanup(self) -> None:
        pass


@pytest.mark.asyncio
async def test_abc_source_methods_default_to_empty_list():
    engine = _MinimalEngine()
    assert await engine.kv_event_sources() == []
    assert engine.component_metrics_dp_ranks() == []
    # attach_snapshot_publisher default is no-op
    engine.attach_snapshot_publisher(object())


@pytest.mark.asyncio
async def test_register_prometheus_default_is_noop():
    assert await _MinimalEngine().register_prometheus(metrics=object()) is None


@pytest.mark.asyncio
async def test_sample_engine_declares_dp_ranks_and_kv_event_source():
    from dingo.common.backend.sample_engine import SampleLLMEngine

    engine = SampleLLMEngine.__new__(SampleLLMEngine)

    engine.disaggregation_mode = DisaggregationMode.AGGREGATED
    assert engine.component_metrics_dp_ranks() == [0]
    sources = await engine.kv_event_sources()
    assert len(sources) == 1
    assert isinstance(sources[0], PushSource)
    assert sources[0].dp_rank == 0

    # Encode workers host neither the component gauges nor a KV-event source.
    engine.disaggregation_mode = DisaggregationMode.ENCODE
    assert engine.component_metrics_dp_ranks() == []
    assert await engine.kv_event_sources() == []


def test_sample_engine_publish_loop_pushes_component_snapshot():
    """An idle publish tick pushes a ComponentSnapshot to the attached
    publisher — the same path real engines use to feed the snapshot gauge."""
    from dingo.common.backend.sample_engine import SampleLLMEngine

    engine = SampleLLMEngine.__new__(SampleLLMEngine)
    engine._kv_used_blocks = 25
    engine._publish_stop = threading.Event()

    published: list[tuple[int, ComponentSnapshot]] = []

    def _publish(rank, snapshot):
        published.append((rank, snapshot))
        engine._publish_stop.set()  # one tick, then let the loop exit

    engine.attach_snapshot_publisher(SimpleNamespace(publish=_publish))
    assert engine._snapshot_publisher is not None

    class _AlwaysEmpty:
        def get(self, timeout):
            raise queue.Empty

    engine._publish_queue = _AlwaysEmpty()

    engine._publish_loop(publisher=None)

    assert published == [
        (
            0,
            ComponentSnapshot(
                kv_used_blocks=25,
                kv_total_blocks=1000,
                gpu_cache_usage=0.025,
                kv_cache_hit_rate=None,
                dp_rank=0,
            ),
        )
    ]


@pytest.mark.asyncio
async def test_vllm_kv_event_sources_return_one_zmq_source_per_dp_rank(monkeypatch):
    mod = pytest.importorskip(
        "dingo.vllm.llm_engine", reason="vLLM backend dependencies not installed"
    )
    from dingo.common.constants import DisaggregationMode

    engine = mod.VllmLLMEngine.__new__(mod.VllmLLMEngine)
    engine.engine_args = SimpleNamespace(
        enable_prefix_caching=True,
        kv_events_config=SimpleNamespace(
            enable_kv_cache_events=True,
            endpoint="tcp://*:5557",
        ),
    )
    engine.disaggregation_mode = DisaggregationMode.AGGREGATED
    engine._vllm_config = object()
    engine._dp_range = (2, 3)

    monkeypatch.setattr(
        mod.ZmqEventPublisher,
        "offset_endpoint_port",
        staticmethod(
            lambda endpoint, data_parallel_rank: f"{endpoint}-{data_parallel_rank}"
        ),
    )

    sources = await engine.kv_event_sources()

    assert all(isinstance(source, ZmqSource) for source in sources)
    assert [(source.endpoint, source.dp_rank) for source in sources] == [
        ("tcp://127.0.0.1:5557-2", 2),
        ("tcp://127.0.0.1:5557-3", 3),
        ("tcp://127.0.0.1:5557-4", 4),
    ]

    engine.engine_args.enable_prefix_caching = False
    assert await engine.kv_event_sources() == []


@pytest.mark.asyncio
async def test_sglang_kv_event_sources_return_one_zmq_source_per_local_dp_rank(
    monkeypatch,
):
    mod = pytest.importorskip(
        "dingo.sglang.llm_engine", reason="SGLang backend dependencies not installed"
    )

    engine = mod.SglangLLMEngine.__new__(mod.SglangLLMEngine)
    engine.server_args = SimpleNamespace(
        kv_events_config=json.dumps({"endpoint": "tcp://*:5557"}),
        dp_size=8,
        enable_dp_attention=True,
        nnodes=2,
        node_rank=1,
    )

    monkeypatch.setattr(mod, "get_local_ip_auto", lambda: "127.0.0.1")
    monkeypatch.setattr(
        mod.ZmqEventPublisher,
        "offset_endpoint_port",
        staticmethod(lambda _endpoint, dp_rank: f"tcp://*:{6000 + dp_rank}"),
    )

    sources = await engine.kv_event_sources()

    assert all(isinstance(source, ZmqSource) for source in sources)
    assert [(source.endpoint, source.dp_rank) for source in sources] == [
        ("tcp://127.0.0.1:6004", 4),
        ("tcp://127.0.0.1:6005", 5),
        ("tcp://127.0.0.1:6006", 6),
        ("tcp://127.0.0.1:6007", 7),
    ]
