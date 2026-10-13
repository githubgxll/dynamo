# SPDX-License-Identifier: Apache-2.0
"""Cache isolation through the real HTTP and Python preprocessing boundaries."""

import json
import logging
import time
import uuid

import pytest
import requests

from tests.router.test_router_e2e_with_sglang import SGLangProcess
from tests.utils.managed_process import DynamoFrontendProcess

MODEL = "Qwen/Qwen3-0.6B"
logger = logging.getLogger(__name__)
pytestmark = [
    pytest.mark.sglang,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.model(MODEL),
    pytest.mark.profiled_vram_gib(6.0),
    pytest.mark.requested_sglang_kv_tokens(8192),
    pytest.mark.timeout(400),
]


def _wait_for_model(url):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            response = requests.get(url + "/v1/models", timeout=2)
            if response.ok and any(
                model["id"] == MODEL for model in response.json().get("data", [])
            ):
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    pytest.fail("HTTP frontend did not materialize the cache isolation model")


def _cached_tokens(url, prompt, salt, source, stream):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 7,
        "max_tokens": 4,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {}
    if source == "header":
        headers["x-tenant-id"] = salt
        body["nvext"] = {"cache_salt": "body-must-not-win"}
        body["cache_salt"] = "legacy-must-not-win"
    elif source == "nvext":
        body["nvext"] = {"cache_salt": salt}
        body["cache_salt"] = "legacy-must-not-win"
    elif source == "legacy":
        body["cache_salt"] = salt
    else:
        body["nvext"] = {"cache_salt": ""}
        body["cache_salt"] = salt
    if stream:
        body["stream_options"] = {"include_usage": True}
    with requests.post(
        url + "/v1/chat/completions",
        json=body,
        headers=headers,
        stream=stream,
        timeout=60,
    ) as response:
        assert response.status_code == 200, response.text
        if stream:
            usage = None
            for line in response.iter_lines():
                if line.startswith(b"data: ") and line != b"data: [DONE]":
                    event = json.loads(line[6:])
                    assert "error" not in event, event
                    if event.get("usage") is not None:
                        usage = event["usage"]
            assert usage is not None, "stream did not include final usage"
        else:
            usage = response.json()["usage"]
    # The worker omits prompt_tokens_details on a zero-cache terminal usage.
    # A2 below must report a positive value, so missing telemetry cannot pass.
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


@pytest.mark.parametrize("processor", ["sglang", "dynamo"])
@pytest.mark.parametrize("disable_nvext", [False, True])
def test_http_cache_salt_isolation(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    processor,
    disable_nvext,
):
    with SGLangProcess(
        request,
        sglang_args={
            "model": MODEL,
            "page_size": 16,
            "context_length": 2048,
            "max_total_tokens": 8192,
            "disable_cuda_graph": True,
        },
        num_workers=1,
        single_gpu=True,
    ) as worker, DynamoFrontendProcess(
        request,
        frontend_port=0,
        router_mode="kv",
        extra_args=["--dyn-chat-processor", processor],
        extra_env={
            "DYN_NAMESPACE": worker.namespace,
            "DYN_DISABLE_FRONTEND_NVEXT": "1" if disable_nvext else "0",
        },
    ) as frontend:
        url = f"http://localhost:{frontend.http_port}"
        _wait_for_model(url)
        for stream in (False, True):
            for source in ("header", "nvext", "legacy", "empty-canonical"):
                identity = uuid.uuid4().hex
                prompt = ("Cache isolation regression: " + identity + ". ") * 16
                salt_a, salt_b = f"{identity}-A", f"{identity}-B"
                observed = [
                    _cached_tokens(url, prompt, salt, source, stream)
                    for salt in (salt_a, salt_a, salt_b)
                ]
                logger.info(
                    "cache isolation processor=%s nvext_disabled=%s source=%s "
                    "stream=%s cached_tokens_A1_A2_B1=%s",
                    processor,
                    disable_nvext,
                    source,
                    stream,
                    observed,
                )
                assert observed[0] == 0, observed
                assert observed[1] > 0, observed
                assert observed[2] == 0, observed
