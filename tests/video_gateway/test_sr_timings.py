"""SR measurements survive the actual detached result and Gateway handoff chain."""
import copy
import json
from pathlib import Path

import pytest

from dingo.common.video_sr_timings import (
    SR_TIMINGS_KEY, aggregate_sr_timings, normalize_sr_timings, with_sr_timings,
)
from dingo.common.video_result_file import BINARY_RESULT_WRITER, normalize_inline_result
from dingo.video_gateway.models import TaskStatus
from dingo.video_gateway.performance import performance_record
from dingo.video_gateway.telemetry import GatewayTelemetry
from tests.video_gateway.test_task_store import _task
from tests.video_gateway.test_seedvr2_handler import pipeline, generate


def timing(**stages):
    return {"schema_version": 1, "unit": "seconds", "stages": stages}


def native():
    rows = []
    for stage in ('window_input_encode', 'window_output_decode', 'window_model_http'):
        for i, (start, end) in enumerate(((0, 2_000_000_000), (1_000_000_000, 3_000_000_000))):
            rows.append(dict(stage=stage, sr_request_id='sr-a', window_index=i,
                             start_ns=start, end_ns=end, cpu_wall_ms=2000,
                             clock_domain='perf_counter_ns pid=42', status='completed'))
    rows.append(dict(stage='final_video_encode', sr_request_id='sr-a',
                     activity_cpu_wall_ms=6090, cpu_wall_ms=47000, failed_call_count=0))
    return {'schema': 'sr-timing-memory/2', 'metrics_status': 'PASS_COMPLETE', 'files': {
        'request.json': {'sr_request_id': 'sr-a'},
        'summary.json': {'sr_request_id': 'sr-a', 'status': 'completed', 'window_count': 2},
        'server-stages.jsonl': rows}}


def test_cumulative_activity_and_overlapping_window_intervals():
    value = aggregate_sr_timings(native(), {'sr_service_s': 62})
    assert value == timing(service=62, final_video_encode=6.09,
                           window_input_encode=4, window_output_decode=4, window_model_wait=3)
    assert 'delivery_crop' not in value['stages']


@pytest.mark.parametrize('corruption', ['clock', 'missing', 'duplicate', 'failed', 'unknown', 'request', 'old_activity'])
def test_incomplete_native_details_do_not_erase_worker_wall_time(corruption):
    metrics = native(); rows = metrics['files']['server-stages.jsonl']
    if corruption == 'clock': rows[5]['clock_domain'] = 'other process'
    if corruption == 'missing': rows.pop(4)
    if corruption == 'duplicate': rows[5]['window_index'] = 0
    if corruption == 'failed': rows[4]['status'] = 'failed'
    if corruption == 'unknown': metrics['schema'] = 'future'
    if corruption == 'request': metrics['files']['summary.json']['sr_request_id'] = 'other'
    if corruption == 'old_activity': rows[-1].pop('activity_cpu_wall_ms')
    result = aggregate_sr_timings(metrics, {'sr_service_s': 62})['stages']
    assert result['service'] == 62
    if corruption == 'old_activity': assert 'final_video_encode' not in result
    else: assert 'window_model_wait' not in result


@pytest.mark.parametrize('invalid', [True, -1, float('nan'), float('inf'), '1', 10**1000])
def test_optional_numbers_are_strict_and_never_fail_result(invalid):
    value = timing(service=1, final_video_encode=invalid, task_id=1)
    assert normalize_sr_timings(value) == timing(service=1)
    assert normalize_sr_timings({**value, 'schema_version': True}) is None
    assert normalize_sr_timings({**value, 'unit': 'ms'}) is None
    assert with_sr_timings({SR_TIMINGS_KEY: value, 'seed': 42}, None) == {'seed': 42}


def test_histogram_only_observes_committed_handoff_and_serializes_details():
    running = _task('video-sr'); running.status = TaskStatus.IN_PROGRESS
    finalizing = copy.deepcopy(running); finalizing.status = TaskStatus.FINALIZING
    finalizing.normalized_request = with_sr_timings(finalizing.normalized_request, timing(service=62))
    completed = copy.deepcopy(finalizing); completed.status = TaskStatus.COMPLETED
    telemetry = GatewayTelemetry()
    for a, b in [(running, running), (running, finalizing), (finalizing, finalizing), (finalizing, completed)]:
        telemetry.record_transition('test', a, b, gateway_generation='gw', revision=1)
    lines = '\n'.join(telemetry.render_prometheus())
    prefix = 'dingo_video_super_resolution_stage_duration_seconds'
    assert prefix+'_count{pool="pool-a",stage="service"} 1' in lines
    assert prefix+'_sum{pool="pool-a",stage="service"} 62' in lines
    assert prefix+'_bucket{pool="pool-a",stage="service",le="+Inf"} 1' in lines
    assert 'task_id=' not in lines and 'schema=' not in lines
    public = completed.public_dict()['metrics']['super_resolution']
    assert public == performance_record(completed)['super_resolution'] == timing(service=62)
    buckets = [float(line.rsplit(' ', 1)[1]) for line in lines.splitlines() if line.startswith(prefix+'_bucket')]
    # Renderer orders labels lexicographically, so test numeric-bound order explicitly.
    import re
    pairs = [(float(re.search(r'le="([^"]+)"', line)[1]), float(line.rsplit(' ', 1)[1])) for line in lines.splitlines() if line.startswith(prefix+'_bucket')]
    assert [n for _, n in sorted(pairs)] == sorted(buckets)


@pytest.mark.parametrize('early_release', [False, True])
async def test_sr_timings_survive_detached_transport_and_result_commit(make_gateway_config, early_release):
    from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
    from tests.video_gateway.test_dispatcher import _MINIMAL_MP4, _DetachedClient, _pool, _stack, _submit
    expected = timing(service=0.03, final_video_encode=0.01)
    class Handler:
        async def generate(self, request, context):
            artifact = await BINARY_RESULT_WRITER.get().write(_MINIMAL_MP4)
            yield {'status': 'completed', 'data': [{'output_format': 'mp4', 'artifact': artifact}],
                   'super_resolution_timings': expected}
    pool = _pool('fl-pool', 'public-fl', 'dyn://scope.backend.generate')
    pool['execution_mode'] = 'detached'; pool['scheduling']['early_release_slot'] = early_release
    config = make_gateway_config(pools=[pool])
    manager = DetachedOmniTaskManager(Handler(), config.artifact_store.root, binary_results=True, inline_results=True)
    store, artifacts, dispatcher, service = _stack(config, {'fl-pool': _DetachedClient(manager)})
    await dispatcher.start()
    try:
        submitted = await _submit(service, 'public-fl')
        terminal = await dispatcher.wait_terminal(submitted.stored.task.id, 3)
        assert terminal.task.status == TaskStatus.COMPLETED
        assert terminal.task.public_dict()['metrics']['super_resolution'] == expected
        text = '\n'.join(dispatcher.telemetry.render_prometheus())
        assert 'dingo_video_super_resolution_stage_duration_seconds_count{pool="fl-pool",stage="service"} 1' in text
    finally:
        await dispatcher.stop(); await manager.shutdown()


async def test_h3_model_timings_preserved_in_sr_terminal_and_reused_checkpoint(pipeline, tmp_path):
    handler, h3, sr = pipeline
    base = h3.generate
    expected = {'schema': 'minimax_h3.v1', 'unit': 'seconds', 'stages': {'diffuse': 1}}
    async def measured(request, context):
        async for chunk in base(request, context):
            yield {**chunk, 'model_execution': expected}
    h3.generate = measured
    for attempt in ('1-token', '2-token'):
        chunks = await generate(handler, tmp_path/'attempts'/attempt)
        assert chunks[0]['model_execution'] == expected
        inline = normalize_inline_result(chunks[0])
        assert inline['model_execution'] == expected
        assert inline['super_resolution_timings']['stages']['service'] >= 0
    assert len(h3.calls) == 1
    checkpoint = tmp_path/'attempts'/'1-token'/'base-checkpoint.json'
    data = json.loads(checkpoint.read_text()); data.pop('model_execution'); checkpoint.write_text(json.dumps(data))
    chunks = await generate(handler, tmp_path/'attempts'/'3-token')
    assert chunks[0]['model_execution'] is None and len(h3.calls) == 1


async def test_non_sr_request_passes_through_without_sr_measurements(pipeline, tmp_path):
    from types import SimpleNamespace
    from tests.video_gateway.test_seedvr2_handler import payload
    from dingo.common.video_result_file import BinaryResultWriter
    from dingo.common.video_super_resolution import SR_PAYLOAD_KEY
    handler, h3, sr = pipeline
    request = payload(); request.pop(SR_PAYLOAD_KEY)
    token = BINARY_RESULT_WRITER.set(BinaryResultWriter(tmp_path))
    try:
        chunks = [chunk async for chunk in handler.generate(request, SimpleNamespace(is_stopped=lambda: False))]
    finally:
        BINARY_RESULT_WRITER.reset(token)
    assert len(h3.calls) == 1 and not sr.calls
    assert 'super_resolution_timings' not in chunks[0]
    normalized = normalize_inline_result(chunks[0])
    assert 'super_resolution_timings' not in normalized


def test_invalid_optional_telemetry_does_not_fail_inline_video():
    artifact = {'schema_version': 1, 'filename': 'worker-video-'+ 'a'*32 + '.mp4', 'bytes': 10, 'sha256': 'b'*64}
    chunk = {'status': 'completed', 'data': [{'output_format': 'mp4', 'artifact': artifact}],
             'super_resolution_timings': {'schema_version': 500, 'unit': 'ms', 'stages': {'service': 42}}}
    value = normalize_inline_result(chunk)
    assert value['data'][0]['artifact'] == artifact and 'super_resolution_timings' not in value


@pytest.mark.parametrize('size,tier', [((1920,1080),'1080p'),((1080,1920),'1080p'),((2208,936),'1080p'),((2944,1248),'2k'),((1440,2560),'2k'),((2400,1440),'2k')])
@pytest.mark.parametrize('terminal', [TaskStatus.COMPLETED,TaskStatus.FAILED,TaskStatus.CANCELLED,TaskStatus.EXPIRED])
def test_sr_terminal_counter_groups_canvases_and_never_counts_retry(size,tier,terminal):
    from dingo.common.video_super_resolution import H3_SR_PROFILE_V3
    t = _task('video-sr-count');t.status=TaskStatus.IN_PROGRESS
    t.normalized_request.update(width=size[0],height=size[1],super_resolution={'profile':H3_SR_PROFILE_V3})
    queued=copy.deepcopy(t);queued.status=TaskStatus.QUEUED
    ended=copy.deepcopy(t);ended.status=terminal
    telemetry=GatewayTelemetry()
    for before,after in [(t,queued),(queued,t),(t,ended),(ended,ended)]:
        telemetry.record_transition('test',before,after,gateway_generation='g',revision=1)
    lines=[x for x in telemetry.render_prometheus() if x.startswith('dingo_video_super_resolution_requests_total{')]
    assert lines==[f'dingo_video_super_resolution_requests_total{{pool="pool-a",status="{terminal.value}",tier="{tier}"}} 1']


def test_tier_counts_all_presets_and_ignores_base_and_invalid_shapes():
    from dingo.common.video_super_resolution import H3_SR_PROFILE_V3
    from dingo.video_gateway.adapters.h3_output_policy import H3_SR_V3_PRESETS,h3_sr_tier
    tiers=[h3_sr_tier({'width':w,'height':h,'super_resolution':{'profile':H3_SR_PROFILE_V3}}) for w,h in H3_SR_V3_PRESETS]
    assert tiers.count('1080p')==tiers.count('2k')==7
    for request in [{}, {'width':1920,'height':1080}, {'width':1344,'height':768,'super_resolution':{'profile':H3_SR_PROFILE_V3}}, {'width':864,'height':480,'super_resolution':{'profile':H3_SR_PROFILE_V3}}]:
        assert h3_sr_tier(request) is None
        t=_task('no-sr-count');t.status=TaskStatus.COMPLETED;t.normalized_request=request
        telemetry=GatewayTelemetry();telemetry.record_terminal(t)
        assert 'dingo_video_super_resolution_requests_total' not in '\n'.join(telemetry.render_prometheus())


def test_counter_initialization_preserves_counts_without_fake_histogram_samples():
    telemetry=GatewayTelemetry()
    telemetry.initialize_super_resolution_requests('p')
    assert len([x for x in telemetry.render_prometheus() if x.startswith('dingo_video_super_resolution_requests_total{')])==8
    telemetry.increment('dingo_video_super_resolution_requests_total',labels={'pool':'p','tier':'2k','status':'completed'})
    telemetry.initialize_super_resolution_requests('p')
    text='\n'.join(telemetry.render_prometheus())
    assert 'dingo_video_super_resolution_requests_total{pool="p",status="completed",tier="2k"} 1' in text
    assert 'dingo_video_super_resolution_stage_duration_seconds' not in text
