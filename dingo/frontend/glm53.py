# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-only frontend policy, verified against the provider's API probes."""


def is_glm53_model(model: str) -> bool:
    name = model.rsplit("/", 1)[-1].lower().replace("-", "").replace("_", "")
    return name == "glm5.3"


def output_budget(
    model: str, requested: int | None, prompt_tokens: int, context_length: int | None
) -> int | None:
    """Keep the requested cap when it fits; never dispatch an impossible GLM cap.

    This is our bounded implementation of the provider's acceptance behavior,
    not a claim about its internal allocation. Other models retain their policy.
    """
    if not is_glm53_model(model) or requested is None:
        return requested
    if requested > 131072:
        raise ValueError("GLM-5.3 max_tokens must not exceed 131072")
    if context_length is None or context_length <= 0:
        return requested
    remaining = context_length - prompt_tokens
    if remaining <= 0:
        raise ValueError("GLM-5.3 prompt exhausts the model context window")
    return min(requested, remaining)


def model_context_length(runtime: dict, card: dict) -> int | None:
    """Match MDC runtime > architectural > legacy precedence; zero is unbounded."""
    value = runtime.get("context_length")
    if value is None:
        value = card.get("architectural_max_context_length")
    if value is None:
        value = card.get("context_length")
    return value if type(value) is int and value > 0 else None
