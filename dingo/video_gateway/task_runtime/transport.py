# SPDX-License-Identifier: Apache-2.0
"""Single-response control calls; execution and context ownership stay outside."""
import asyncio


async def control_request(client, instance, payload, context, *, timeout_s=None):
    """Validate framing for either supported finite-task wire protocol.

    The caller owns the Context, which may also track a detached execution.
    In particular, finishing a capability/submit call must not stop that Context.
    Streaming video wait notifications use their own protocol adapter.
    """
    async with asyncio.timeout(timeout_s):
        stream = await client.direct(payload, int(instance), context)
        response = None
        async for item in stream:
            if hasattr(item, "is_error") and item.is_error():
                comments = item.comments() if hasattr(item, "comments") else []
                raise RuntimeError(
                    "; ".join(comments) or "Worker control request failed"
                )
            value = item.data() if hasattr(item, "data") else item
            if response is not None or not isinstance(value, dict):
                raise ValueError("invalid Worker control response framing")
            response = value
        if response is None:
            raise ValueError("empty Worker control response")
        return response
