# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate known vLLM request validation failures at the engine boundary."""

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

from dynamo.llm.exceptions import InvalidArgument

try:
    from vllm.exceptions import VLLMValidationError
except ImportError:
    # Older vLLM releases do not expose this exception. An empty tuple catches
    # nothing: unrelated errors must retain their existing handling.
    _VALIDATION_ERRORS: tuple[type[Exception], ...] = ()
else:
    _VALIDATION_ERRORS = (VLLMValidationError,)

_T = TypeVar("_T")
logger = logging.getLogger(__name__)


def generate_with_validation(
    generate: Callable[..., AsyncIterator[_T]], *args: Any, **kwargs: Any
) -> AsyncIterator[_T]:
    """Normalize validation errors from both construction and iteration.

    Keep construction eager so callers' existing EngineDeadError handlers still
    run at the same point. Cancellation and all other exceptions pass through.
    """
    try:
        stream = generate(*args, **kwargs)
    except _VALIDATION_ERRORS as exc:
        raise InvalidArgument(str(exc)) from exc

    async def iterate() -> AsyncIterator[_T]:
        completed = False
        try:
            async for output in stream:
                yield output
            completed = True
        except _VALIDATION_ERRORS as exc:
            raise InvalidArgument(str(exc)) from exc
        finally:
            # Forward explicit close as well as normal/error completion without
            # requiring arbitrary AsyncIterator implementations to have aclose.
            close = getattr(stream, "aclose", None)
            if close is not None:
                try:
                    await close()
                except BaseException:
                    if completed:
                        raise
                    # Cleanup must not replace a validation/engine failure,
                    # cancellation, or GeneratorExit already being propagated.
                    logger.warning(
                        "Failed to close vLLM generation stream", exc_info=True
                    )

    return iterate()
