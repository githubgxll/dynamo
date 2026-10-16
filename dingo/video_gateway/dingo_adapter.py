# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Narrow wrappers around Dynamo endpoint clients and cancellation Contexts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from dingo.video_gateway.config import GatewayConfig


class EndpointClient(Protocol):
    def instance_ids(self) -> list[int]: ...

    async def direct(
        self, payload: dict[str, Any], instance_id: int, context: Any
    ) -> AsyncIterator[Any]: ...


class ContextFactory(Protocol):
    def __call__(self, task_id: str, metadata: dict[str, str]) -> Any: ...


class DingoEndpointClient:
    def __init__(self, client: Any) -> None:
        self._client = client

    def instance_ids(self) -> list[int]:
        return list(self._client.instance_ids())

    async def direct(
        self, payload: dict[str, Any], instance_id: int, context: Any
    ) -> AsyncIterator[Any]:
        return await self._client.direct(
            payload, instance_id=instance_id, annotated=True, context=context
        )


def create_context(task_id: str, metadata: dict[str, str]) -> Any:
    from dynamo._core import Context

    return Context(id=task_id, metadata=metadata)


async def create_pool_clients(
    runtime: Any, config: GatewayConfig
) -> dict[str, DingoEndpointClient]:
    result: dict[str, DingoEndpointClient] = {}
    for pool in config.pools:
        endpoint = runtime.endpoint(pool.endpoint_path)
        result[pool.pool_id] = DingoEndpointClient(await endpoint.client())
    return result
