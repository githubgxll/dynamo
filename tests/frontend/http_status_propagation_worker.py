# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker exercising HTTP-like, validation, and unexpected errors over the wire."""

from __future__ import annotations

import argparse
import asyncio

import uvloop

from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from tests.frontend.test_http_status_propagation import (
    ENDPOINT_PATH,
    EXPECTED_MESSAGE,
    EXPECTED_STATUS,
    MODEL_NAME,
)
from tests.utils.constants import QWEN


class _StatusLikeError(Exception):
    """Duck-typed `.status` + `.message` — same shape as
    `dingo.common.http.HttpStatusError`."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


def make_generate(error_kind: str, message: str, after_chunk: bool):
    async def generate(request, context):
        if after_chunk:
            yield {"token_ids": [9707]}
        if error_kind == "invalid_argument":
            from dynamo.llm.exceptions import InvalidArgument

            raise InvalidArgument(message)
        if error_kind == "value_error":
            raise ValueError(message)
        if error_kind == "unknown":
            raise RuntimeError(message)
        raise _StatusLikeError(status=EXPECTED_STATUS, message=message)

    return generate


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-kind", default="http_status")
    parser.add_argument("--message", default=EXPECTED_MESSAGE)
    parser.add_argument("--after-chunk", action="store_true")
    args = parser.parse_args()
    runtime = DistributedRuntime(asyncio.get_running_loop(), "etcd", "tcp")
    endpoint = runtime.endpoint(ENDPOINT_PATH)
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        QWEN,
        model_name=MODEL_NAME,
        worker_type=WorkerType.Aggregated,
    )
    await endpoint.serve_endpoint(
        make_generate(args.error_kind, args.message, args.after_chunk)
    )


if __name__ == "__main__":
    uvloop.run(main())
