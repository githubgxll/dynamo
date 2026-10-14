# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-Omni integration for Dynamo."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_handler import BaseOmniHandler
    from .omni_handler import OmniHandler
    from .realtime_handler import RealtimeOmniHandler

__all__ = ["BaseOmniHandler", "OmniHandler", "RealtimeOmniHandler"]

_HANDLER_MODULES = {
    "BaseOmniHandler": ".base_handler",
    "OmniHandler": ".omni_handler",
    "RealtimeOmniHandler": ".realtime_handler",
}


def __getattr__(name: str):
    # Protocol helpers and detached task management also run outside an Omni
    # worker. Load the optional engine only when a handler is requested.
    if name not in _HANDLER_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_HANDLER_MODULES[name], __name__), name)
    globals()[name] = value
    return value
