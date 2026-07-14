# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal utilities for Dynamo components."""

from collections.abc import Callable

from dingo.common.constants import EmbeddingTransferMode
from dingo.common.multimodal.async_encoder_cache import AsyncEncoderCache
from dingo.common.multimodal.audio_loader import AudioLoader
from dingo.common.multimodal.embedding_transfer import (
    AbstractEmbeddingReceiver,
    AbstractEmbeddingSender,
    LocalEmbeddingReceiver,
    LocalEmbeddingSender,
    NixlReadEmbeddingReceiver,
    NixlReadEmbeddingSender,
    NixlWriteEmbeddingReceiver,
    NixlWriteEmbeddingSender,
    TransferRequest,
)
from dingo.common.multimodal.image_loader import ImageLoader
from dingo.common.multimodal.video_loader import VideoLoader

EMBEDDING_SENDER_FACTORIES: dict[
    EmbeddingTransferMode, Callable[[], AbstractEmbeddingSender]
] = {
    EmbeddingTransferMode.LOCAL: LocalEmbeddingSender,
    EmbeddingTransferMode.NIXL_WRITE: NixlWriteEmbeddingSender,
    EmbeddingTransferMode.NIXL_READ: NixlReadEmbeddingSender,
}

EMBEDDING_RECEIVER_FACTORIES: dict[
    EmbeddingTransferMode, Callable[[], AbstractEmbeddingReceiver]
] = {
    EmbeddingTransferMode.LOCAL: LocalEmbeddingReceiver,
    EmbeddingTransferMode.NIXL_WRITE: NixlWriteEmbeddingReceiver,
    # [gluo FIXME] can't use pre-registered tensor as NIXL requires descriptors
    # to be at matching size, need to overwrite nixl connect library
    EmbeddingTransferMode.NIXL_READ: lambda: NixlReadEmbeddingReceiver(max_items=0),
}

__all__ = [
    "AsyncEncoderCache",
    "AudioLoader",
    "EMBEDDING_RECEIVER_FACTORIES",
    "EMBEDDING_SENDER_FACTORIES",
    "ImageLoader",
    "VideoLoader",
    "NixlReadEmbeddingReceiver",
    "NixlReadEmbeddingSender",
    "NixlWriteEmbeddingSender",
    "NixlWriteEmbeddingReceiver",
    "TransferRequest",
    "LocalEmbeddingReceiver",
    "LocalEmbeddingSender",
]
