# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dingo.video_gateway.dingo_adapter import (
    DingoEndpointClient,
    create_pool_clients,
)
from tests.video_gateway.test_dispatcher import _pool


class _BindingClient:
    def __init__(self, instance_id: int) -> None:
        self.instance_id = instance_id
        self.calls = []

    def instance_ids(self):
        return [self.instance_id]

    async def direct(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return "annotated-stream"


class _Endpoint:
    def __init__(self, client) -> None:
        self._client = client

    async def client(self):
        return self._client


class _Runtime:
    def __init__(self) -> None:
        self.paths = []
        self.clients = {}

    def endpoint(self, path):
        self.paths.append(path)
        client = _BindingClient(len(self.paths))
        self.clients[path] = client
        return _Endpoint(client)


async def test_each_pool_uses_its_complete_configured_discovery_target(
    make_gateway_config,
):
    config = make_gateway_config(
        pools=[
            _pool("fl", "model-fl", "dyn://scope-one.component-a.generate"),
            _pool(
                "ref",
                "model-ref",
                "dyn://unrelated-scope.component-b.infer",
                workflow="ref2va",
            ),
        ]
    )
    runtime = _Runtime()

    clients = await create_pool_clients(runtime, config)

    assert runtime.paths == [
        "scope-one.component-a.generate",
        "unrelated-scope.component-b.infer",
    ]
    assert set(clients) == {"fl", "ref"}
    assert clients["fl"].instance_ids() == [1]
    assert clients["ref"].instance_ids() == [2]


async def test_direct_wrapper_preserves_instance_context_and_annotated_stream():
    binding = _BindingClient(41)
    client = DingoEndpointClient(binding)
    context = object()

    result = await client.direct({"model": "backend-model"}, 41, context)

    assert result == "annotated-stream"
    assert binding.calls == [
        (
            {"model": "backend-model"},
            {"instance_id": 41, "annotated": True, "context": context},
        )
    ]
