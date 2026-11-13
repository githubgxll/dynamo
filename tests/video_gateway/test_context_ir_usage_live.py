# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Gateway HTTP replay of real multimodal LLM usage evidence."""
import json
import os
from pathlib import Path
import pytest
from tests.video_gateway.test_context_ir_http import setup, form, wait_task

@pytest.mark.skipif(not os.getenv('DINGO_CONTEXTIR_USAGE_EVIDENCE'),reason='requires recorded live Qwen usage evidence')
async def test_real_multimodal_aggregate_survives_gateway_http(setup):
    evidence=json.loads(Path(os.environ['DINGO_CONTEXTIR_USAGE_EVIDENCE']).read_text())
    usage=evidence['aggregate']
    assert usage['complete'] and usage['llm_requests']==3
    compiler,service,client=setup
    compiler.usage=usage
    response=await client.post('/v1/context-ir',data=form(),headers={'Idempotency-Key':'multimodal-usage'})
    task=await response.json();result=await wait_task(client,task['id'])
    assert result['usage']==usage
    diagnostic=await (await client.get('/v1/context-ir/'+task['id']+'/diagnostics')).json()
    assert diagnostic['usage']==usage
    replay=await client.post('/v1/context-ir',data=form(),headers={'Idempotency-Key':'multimodal-usage'})
    assert (await replay.json())['usage']==usage and compiler.calls==1
