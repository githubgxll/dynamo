# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests: load the boundary helper without importing GPU dependencies."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.vllm, pytest.mark.gpu_0, pytest.mark.pre_merge]


class VLLMValidationError(Exception):
    """New vLLM type is neither ValueError nor an HTTP exception."""


class InvalidArgument(Exception):
    pass


class EngineDeadError(Exception):
    pass


@pytest.fixture
def load_helper(monkeypatch):
    def load(version="new"):
        vllm = ModuleType("vllm")
        vllm.__path__ = []
        exceptions = ModuleType("vllm.exceptions")
        if version == "new":
            exceptions.VLLMValidationError = VLLMValidationError
        dynamo_exceptions = ModuleType("dynamo.llm.exceptions")
        dynamo_exceptions.InvalidArgument = InvalidArgument
        monkeypatch.setitem(sys.modules, "vllm", vllm)
        monkeypatch.setitem(
            sys.modules,
            "vllm.exceptions",
            None if version == "no_module" else exceptions,
        )
        monkeypatch.setitem(sys.modules, "dynamo.llm.exceptions", dynamo_exceptions)
        path = Path(__file__).parents[1] / "generation_errors.py"
        spec = importlib.util.spec_from_file_location("generation_errors_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.generate_with_validation

    return load


def _generate(error, phase):
    def generate(*args, **kwargs):
        if phase == "construction":
            raise error

        async def stream():
            if phase == "after_output":
                yield "first"
            raise error

        return stream()

    return generate


@pytest.mark.parametrize("phase", ["construction", "iteration", "after_output"])
def test_validation_error_is_invalid_argument(load_helper, phase):
    wrap = load_helper()
    error = VLLMValidationError("max_tokens exceeds context length; details preserved")
    assert not isinstance(error, ValueError)
    assert not hasattr(error, "status_code")
    outputs = []

    async def run():
        async for output in wrap(_generate(error, phase)):
            outputs.append(output)

    with pytest.raises(InvalidArgument) as caught:
        asyncio.run(run())
    assert str(caught.value) == str(error)
    assert caught.value.__cause__ is error
    assert outputs == (["first"] if phase == "after_output" else [])


@pytest.mark.parametrize("phase", ["construction", "iteration", "after_output"])
@pytest.mark.parametrize(
    "error_type", [RuntimeError, ValueError, EngineDeadError, asyncio.CancelledError]
)
def test_other_errors_pass_through_unchanged(load_helper, phase, error_type):
    wrap = load_helper()
    # Validation-looking text must never cause an unrelated error to be converted.
    error = error_type("max_tokens exceeds context length")

    async def run():
        async for _ in wrap(_generate(error, phase)):
            pass

    with pytest.raises(error_type) as caught:
        asyncio.run(run())
    assert caught.value is error


@pytest.mark.parametrize("version", ["no_class", "no_module"])
def test_older_vllm_imports_and_does_not_convert_errors(load_helper, version):
    wrap = load_helper(version)
    error = VLLMValidationError("not a known type in this vLLM version")

    async def run():
        async for _ in wrap(_generate(error, "iteration")):
            pass

    with pytest.raises(VLLMValidationError) as caught:
        asyncio.run(run())
    assert caught.value is error


def test_success_preserves_arguments_output_and_eager_construction(load_helper):
    wrap = load_helper()
    outputs = [object(), object()]

    async def stream():
        for output in outputs:
            yield output

    generate = Mock(return_value=stream())
    wrapped = wrap(generate, "prompt", "params", "request", priority=3)
    generate.assert_called_once_with("prompt", "params", "request", priority=3)

    async def run():
        return [output async for output in wrapped]

    assert asyncio.run(run()) == outputs


def test_early_close_immediately_closes_inner_generator(load_helper):
    wrap = load_helper()
    closed = []

    async def stream():
        try:
            yield "first"
            yield "second"
        finally:
            closed.append(True)

    async def run():
        wrapped = wrap(stream)
        assert await anext(wrapped) == "first"
        assert closed == []
        await wrapped.aclose()
        # Assert before asyncio.run shuts down remaining async generators.
        assert closed == [True]

    asyncio.run(run())


def test_iterator_without_aclose_remains_supported(load_helper):
    wrap = load_helper()

    class Stream:
        def __init__(self):
            self.outputs = iter(["first", "second"])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.outputs)
            except StopIteration:
                raise StopAsyncIteration

    async def run():
        return [output async for output in wrap(Stream)]

    assert asyncio.run(run()) == ["first", "second"]


@pytest.mark.parametrize(
    "error_type", [VLLMValidationError, EngineDeadError, asyncio.CancelledError]
)
def test_close_failure_does_not_mask_original_error(load_helper, error_type):
    wrap = load_helper()
    error = error_type("original failure")
    closed = []

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise error

        async def aclose(self):
            closed.append(True)
            raise RuntimeError("cleanup failure")

    async def run():
        async for _ in wrap(Stream):
            pass

    expected_type = InvalidArgument if error_type is VLLMValidationError else error_type
    with pytest.raises(expected_type) as caught:
        asyncio.run(run())
    if error_type is VLLMValidationError:
        assert caught.value.__cause__ is error
        assert str(caught.value) == str(error)
    else:
        assert caught.value is error
    assert closed == [True]


def test_close_failure_without_original_error_propagates(load_helper):
    wrap = load_helper()
    error = RuntimeError("cleanup failure")

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise error

    async def run():
        async for _ in wrap(Stream):
            pass

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(run())
    assert caught.value is error
