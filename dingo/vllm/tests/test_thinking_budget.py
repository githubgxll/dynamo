# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests of the real guard/error helpers with an incremental fake engine."""

import ast
import asyncio
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]
ROOT = Path(__file__).parents[3]


def load(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def helper(monkeypatch):
    monkeypatch.delenv("DYN_VLLM_DEFAULT_MAX_THINKING_TOKENS", raising=False)
    core = ModuleType("dynamo._core")
    for name in (
        "Cancelled",
        "CannotConnect",
        "ConnectionTimeout",
        "Disconnected",
        "DynamoException",
        "EngineShutdown",
        "InvalidArgument",
        "RouterQueueLimitExceeded",
        "SelectionServiceError",
        "StreamIncomplete",
        "Unknown",
    ):
        setattr(core, name, type(name, (Exception,), {}))
    monkeypatch.setitem(sys.modules, "dynamo._core", core)
    load(
        "dynamo.llm.exceptions",
        ROOT / "lib/bindings/python/src/dynamo/llm/exceptions.py",
        monkeypatch,
    )
    monkeypatch.setitem(sys.modules, "vllm.exceptions", None)
    package = ModuleType("budget_test")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "budget_test", package)
    load(
        "budget_test.generation_errors",
        ROOT / "dingo/vllm/generation_errors.py",
        monkeypatch,
    )
    module = load(
        "budget_test.thinking_budget",
        ROOT / "dingo/vllm/thinking_budget.py",
        monkeypatch,
    )

    class Parser:
        reasoning_end_token_ids = frozenset({2})

        def is_reasoning_end(self, ids):
            # A GLM-like initial reasoning state and an assistant turn boundary.
            for token in reversed(ids):
                if token == 2:
                    return True
                if token in (1, 9):
                    return False
            return False

    monkeypatch.setattr(module, "_make_parser", lambda engine, kwargs: Parser())
    return module


class Params(SimpleNamespace):
    def __init__(self, budget=None, kind="DELTA"):
        super().__init__(
            thinking_token_budget=budget,
            output_kind=SimpleNamespace(name=kind),
            min_tokens=999,
        )

    def clone(self):
        return copy.deepcopy(self)


def result(tokens, index=0, prompt=(9, 1)):
    return SimpleNamespace(
        prompt_token_ids=None if prompt is None else list(prompt),
        outputs=[SimpleNamespace(index=index, token_ids=tokens)],
    )


class Engine:
    def __init__(self, results):
        self.results = results
        self.calls = []
        self.aborts = []
        self.closed = False
        self.vllm_config = SimpleNamespace(
            reasoning_config=SimpleNamespace(
                enabled=True,
                reasoning_parser="glm45",
                reasoning_start_token_ids=[1],
                natural_reasoning_end_token_ids=[2],
            )
        )

    def generate(self, prompt, params, request_id, **kwargs):
        self.calls.append((params, kwargs))

        async def stream():
            try:
                for output in self.results:
                    yield output
            finally:
                self.closed = True

        return stream()

    async def abort(self, request_id):
        self.aborts.append(request_id)


def collect(helper, engine, params, **kwargs):
    outputs = []

    async def run():
        async for value in helper.generate_with_thinking_budget(
            engine, {"prompt_token_ids": [9, 1]}, params, "request", **kwargs
        ):
            outputs.append(value)

    return outputs, run


@pytest.mark.parametrize(
    "name", ["build_sampling_params", "build_sampling_params_openai"]
)
@pytest.mark.parametrize("budget", [None, 0, 8])
def test_real_request_builders_preserve_thinking_budget_without_changing_max_tokens(
    helper, name, budget
):
    class BuilderParams(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(thinking_token_budget=None, **kwargs)

    tree = ast.parse((ROOT / "dingo/vllm/handlers.py").read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace = {
        "Dict": dict,
        "Any": Any,
        "SamplingParams": BuilderParams,
        "HttpError": helper.HttpError,
        "_shared_logprobs": SimpleNamespace(
            parse_logprob_options=lambda _: (None, None)
        ),
        "_DELTA_REQUEST_OUTPUT_KIND": "DELTA",
    }
    exec(  # noqa: S102 - execute the repository's real builder without GPU imports
        compile(ast.Module(body=[node], type_ignores=[]), "handlers.py", "exec"),
        namespace,
    )
    request = (
        {"stop_conditions": {"max_thinking_tokens": budget}}
        if name == "build_sampling_params"
        else {"nvext": {"max_thinking_tokens": budget}}
    )
    params = namespace[name](request, {"max_tokens": 2048})
    assert params.thinking_token_budget == budget
    assert params.max_tokens == 2048


def test_empty_engine_output_does_not_open_deferred_abort_window(helper):
    engine = Engine([result([]), result([11])])
    events = []

    class Deferred:
        def signal_first_token(self):
            events.append("token")

        async def abort(self):
            assert events == ["empty", "token"]
            await engine.abort("request")

    async def run():
        async for output in helper.generate_with_thinking_budget(
            engine,
            {"prompt_token_ids": [9, 1]},
            Params(0),
            "request",
            abort_guard=Deferred(),
        ):
            assert output.outputs[0].token_ids == []
            assert not events
            events.append("empty")

    with pytest.raises(helper.HttpError, match="thinking_token_budget_exceeded"):
        asyncio.run(run())
    assert engine.aborts == ["request"]


@pytest.mark.parametrize("kind", ["DELTA", "CUMULATIVE"])
def test_exact_limit_aborts_and_returns_trusted_error(helper, kind):
    engine = Engine(
        [result([11]), result([12] if kind == "DELTA" else [11, 12]), result([2, 99])]
    )
    params = Params(2, kind)
    outputs, run = collect(helper, engine, params, priority=3)
    with pytest.raises(helper.HttpError) as error:
        asyncio.run(run())
    assert error.value.code == 400
    assert "thinking_token_budget_exceeded" in error.value.message
    assert "(2)" in error.value.message
    assert len(outputs) == 1
    assert engine.aborts == ["request"] and engine.closed
    assert engine.calls[0][0].thinking_token_budget is None
    assert params.thinking_token_budget == 2
    assert engine.calls[0][1] == {"priority": 3}


def test_mtp_batch_cannot_cross_budget_and_deliver_answer(helper):
    engine = Engine([result([11, 12, 2, 99, 100])])
    outputs, run = collect(helper, engine, Params(2))
    with pytest.raises(helper.HttpError, match="thinking_token_budget_exceeded"):
        asyncio.run(run())
    assert not outputs and engine.aborts == ["request"] and engine.closed


def test_natural_reasoning_end_does_not_count_answer_tokens(helper):
    engine = Engine([result([11, 2] + [50] * 100)])
    outputs, run = collect(helper, engine, Params(2))
    asyncio.run(run())
    assert len(outputs) == 1 and not engine.aborts


def test_implicit_reasoning_uses_current_turn_not_historical_end(helper):
    engine = Engine([result([11, 12], prompt=[1, 77, 2, 9])])
    _, run = collect(helper, engine, Params(2))
    with pytest.raises(helper.HttpError, match="thinking_token_budget_exceeded"):
        asyncio.run(run())
    assert engine.aborts == ["request"]


def test_any_choice_exhaustion_aborts_whole_request(helper):
    batch = result([11])
    batch.outputs.append(SimpleNamespace(index=1, token_ids=[11, 12]))
    engine = Engine([batch])
    outputs, run = collect(helper, engine, Params(2))
    with pytest.raises(helper.HttpError, match="choice 1"):
        asyncio.run(run())
    assert not outputs and engine.aborts == ["request"]


def test_deferred_abort_signalled_before_first_chunk_error(helper):
    engine = Engine([result([11, 12])])
    events = []

    class Deferred:
        def signal_first_token(self):
            events.append("first")

        async def abort(self):
            assert events == ["first"]
            events.append("abort")
            await engine.abort("request")

    _, run = collect(helper, engine, Params(2), abort_guard=Deferred())
    with pytest.raises(helper.HttpError):
        asyncio.run(run())
    assert events == ["first", "abort"] and engine.closed


@pytest.mark.parametrize("explicit", [None, 0, 1, 100])
def test_default_backfill_never_overrides_explicit_budget(
    helper, monkeypatch, explicit
):
    monkeypatch.setenv("DYN_VLLM_DEFAULT_MAX_THINKING_TOKENS", "2")
    engine = Engine([result([11, 12, 13])])
    outputs, run = collect(helper, engine, Params(explicit))
    if explicit == 100:
        asyncio.run(run())
        assert len(outputs) == 1 and not engine.aborts
    else:
        with pytest.raises(helper.HttpError, match="thinking_token_budget_exceeded"):
            asyncio.run(run())
        assert engine.aborts == ["request"]


def test_zero_budget_allows_prompt_that_already_ended_reasoning(helper):
    engine = Engine([result([50] * 20, prompt=[9, 2])])
    _, run = collect(helper, engine, Params(0))
    asyncio.run(run())
    assert not engine.aborts


def test_reopened_reasoning_counts_against_same_budget(helper):
    engine = Engine([result([11, 2, 50, 1, 12])])
    _, run = collect(helper, engine, Params(2))
    with pytest.raises(helper.HttpError):
        asyncio.run(run())
    assert engine.aborts == ["request"]


def test_unconfigured_budget_preserves_original_sampling_and_no_parser(
    helper, monkeypatch
):
    monkeypatch.setattr(
        helper, "_make_parser", lambda *args: pytest.fail("no parser needed")
    )
    engine = Engine([result([11] * 100)])
    engine.vllm_config = None
    params = Params()
    outputs, run = collect(helper, engine, params)
    asyncio.run(run())
    assert len(outputs) == 1 and engine.calls[0][0] is params and not engine.aborts


def test_template_disabled_thinking_does_not_count_plain_answer(helper, monkeypatch):
    metadata = {"chat_template_kwargs": {"enable_thinking": False}}

    def parser(engine, kwargs):
        assert kwargs == metadata
        return SimpleNamespace(
            reasoning_end_token_ids=frozenset({2}),
            is_reasoning_end=lambda ids: True,
        )

    monkeypatch.setattr(helper, "_make_parser", parser)
    engine = Engine([result([50] * 100, prompt=[9])])
    outputs, run = collect(helper, engine, Params(0), budget_parser_kwargs=metadata)
    asyncio.run(run())
    assert len(outputs) == 1 and not engine.aborts
    assert engine.calls[0][1] == {}


def test_abort_failure_is_not_reported_as_successful_budget_abort(helper):
    engine = Engine([result([11, 12])])

    async def fail_abort(request_id):
        raise RuntimeError("engine unavailable")

    engine.abort = fail_abort
    _, run = collect(helper, engine, Params(2))
    with pytest.raises(RuntimeError, match="engine unavailable"):
        asyncio.run(run())
    assert engine.closed


def test_abort_is_not_cancelled_with_client_task(helper):
    async def run():
        engine = Engine([result([11, 12])])
        started, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def abort(request_id):
            started.set()
            await release.wait()
            engine.aborts.append(request_id)
            completed.set()

        engine.abort = abort
        _, consume = collect(helper, engine, Params(2))
        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.wait_for(completed.wait(), timeout=1)
        assert engine.aborts == ["request"] and engine.closed

    asyncio.run(run())


@pytest.mark.parametrize("budget", [-1, True, "2", 1.5])
def test_invalid_budget_fails_before_engine_submission(helper, budget):
    engine = Engine([])
    _, run = collect(helper, engine, Params(budget))
    with pytest.raises(helper.HttpError, match="nonnegative integer"):
        asyncio.run(run())
    assert not engine.calls


@pytest.mark.parametrize("value", ["", "-1", "1.5", "true", " 2"])
def test_invalid_default_is_rejected(helper, monkeypatch, value):
    monkeypatch.setenv("DYN_VLLM_DEFAULT_MAX_THINKING_TOKENS", value)
    with pytest.raises(ValueError):
        helper.default_thinking_budget()


def test_unsupported_markers_fail_closed(helper):
    engine = Engine([])
    engine.vllm_config.reasoning_config.reasoning_start_token_ids = [1, 3]
    _, run = collect(helper, engine, Params(2))
    with pytest.raises(helper.HttpError, match="thinking_budget_unavailable"):
        asyncio.run(run())
    assert not engine.calls


def test_final_only_output_is_not_accepted_for_live_enforcement(helper):
    engine = Engine([])
    _, run = collect(helper, engine, Params(2, "FINAL_ONLY"))
    with pytest.raises(helper.HttpError, match="incremental"):
        asyncio.run(run())
    assert not engine.calls


def test_missing_prompt_state_aborts_instead_of_silently_disabling(helper):
    engine = Engine([result([11], prompt=None)])

    async def run():
        async for _ in helper.generate_with_thinking_budget(
            engine, "text", Params(2), "request"
        ):
            pytest.fail("unknown-state output must not escape")

    with pytest.raises(helper.HttpError, match="prompt token IDs"):
        asyncio.run(run())
    assert engine.aborts == ["request"] and engine.closed
