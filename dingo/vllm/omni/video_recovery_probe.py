# SPDX-License-Identifier: Apache-2.0
"""Opt-in single-attempt OOM recovery probe for the deployed H3 FL2VA profile.

Never submit through the public endpoint and never reopen admission until a
real small forward, file validation and all-rank device barriers have passed.
"""
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from aiohttp import ClientSession, ClientTimeout
from dingo.common.video_result_file import BINARY_RESULT_WRITER, BinaryResultWriter, validate_descriptor
from dingo.common.video_worker_error import failed_chunk_error
from dingo.vllm.omni.detached_tasks import _DetachedContext
from dingo.vllm.omni.seedvr2_media import probe
from dingo.vllm.omni.video_recovery import process_snapshot


async def probe_same_engine(recovery):
    manager, handler = recovery.manager, recovery.handler
    if manager._running or manager._direct_running:
        raise RuntimeError('accepted executions are not settled')
    if manager.execution_capacity != 1:
        raise RuntimeError('recovery probe requires one execution slot')
    engine = handler.engine_client
    pools = getattr(engine.engine, 'stage_pools', None)
    if not pools or len(pools) != 1:
        raise RuntimeError('recovery probe requires a single diffusion stage')
    clients = [c for c in pools[0].clients if c is not None]
    if len(clients) != 1 or not callable(getattr(clients[0], 'collective_rpc_async', None)):
        raise RuntimeError('all-rank synchronization interface unavailable')
    root = manager.root / '_recovery_probes' / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    report = {'pid': os.getpid(), 'before': process_snapshot(), 'phase': 'created', 'started_at': time.time()}

    def record(phase, **extra):
        report.update(phase=phase, **extra)
        (root / 'report.json').write_text(json.dumps(report, indent=2))

    async def barrier():
        # Native control RPC runs between model invocations and gathers every
        # rank's status; synchronize_device also waits for pending CUDA work.
        return await asyncio.wait_for(clients[0].collective_rpc_async(
            'synchronize_device', timeout=10, kwargs={'timeout': 10}), timeout=12)

    context = _DetachedContext('recovery-probe-' + root.name)
    token = None
    try:
        async with manager._execution_slots:
            await engine.check_health()
            record('before_barrier')
            result = await barrier()
            record('barrier_passed', barrier_result=result)
            payload = {'model': handler.config.served_model_name, 'prompt': '一只橘猫坐在窗边，缓缓转头看向镜头。',
                       'size': '1344x768', 'response_format': 'b64_json', 'output_format': 'mp4',
                       'nvext': {'fps': 24, 'num_frames': 96, 'num_inference_steps': 8, 'seed': 2026102605}}
            token = BINARY_RESULT_WRITER.set(BinaryResultWriter(root))
            record('generating')
            terminal = None
            async for chunk in handler.generate(payload, context):
                if chunk.get('status') == 'failed':
                    raise failed_chunk_error(chunk, stage='base_generation')
                if chunk.get('status') == 'completed':
                    if terminal is not None:
                        raise RuntimeError('probe returned multiple completions')
                    terminal = chunk
            if terminal is None:
                raise RuntimeError('probe did not complete')
            items = terminal.get('data')
            if not isinstance(items, list) or len(items) != 1:
                raise RuntimeError('probe output is not one video')
            descriptor = validate_descriptor(items[0].get('artifact'))
            path = root / descriptor['filename']
            info = await probe(path)
            if info['sha256'] != descriptor['sha256'] or (info['width'],info['height'],info['frames']) != (1344,768,107) or not info.get('audio'):
                raise RuntimeError('probe video validation failed')
            await barrier()
            after = process_snapshot()
            identity = lambda rows: sorted((v['pid'],v['start_ticks']) for v in rows)
            if identity(after) != identity(report['before']):
                raise RuntimeError('engine process identity changed during recovery probe')
            async with ClientSession(trust_env=False, timeout=ClientTimeout(total=3)) as session:
                async with session.get('http://127.0.0.1:8124/health') as response:
                    if response.status != 200:
                        raise RuntimeError('SR service is unavailable')
            record('passed', after=after, output=str(path), sha256=info['sha256'], frames=info['frames'], elapsed_s=time.time()-report['started_at'])
            return dict(report)
    except BaseException as exc:
        record('failed', error_type=type(exc).__name__, error=str(exc)[:2000], after=process_snapshot())
        context.stop_generating()
        raise
    finally:
        if token is not None:
            BINARY_RESULT_WRITER.reset(token)
        await context.close()
