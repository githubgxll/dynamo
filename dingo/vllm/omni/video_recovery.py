# SPDX-License-Identifier: Apache-2.0
"""Video-only supervision of owned engines and unresolved execution cleanup."""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _process_identity(path):
    stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': int(path.name), 'name': (path / 'comm').read_text().strip(),
            'state': stat[0], 'pgrp': int(stat[2]), 'start_ticks': stat[19]}


def process_snapshot():
    """Diagnostic evidence only; process discovery never uses command substrings."""
    result = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            command = (path / 'cmdline').read_bytes()
            if any(marker in command for marker in (b'dingo.vllm.omni', b'vLLM-Omni', b'vllm-omni')):
                result.append(_process_identity(path))
        except (OSError, IndexError, ValueError):
            continue
    return result


class OwnedSRRanks:
    """Pin required diffusion ranks inside the owned SR process group at readiness.

    Transient ffmpeg/media children, other process groups, and later processes
    with a reused PID cannot satisfy the captured rank identities.
    """
    def __init__(self, process, expected_ranks, *, proc_root=Path('/proc')):
        self.proc_root = proc_root
        self.ranks = []
        for path in proc_root.glob('[0-9]*'):
            try:
                identity = _process_identity(path)
            except FileNotFoundError:
                continue
            if identity['pgrp'] == process.pid and identity['name'].startswith('vLLM-Omni::Diff'):
                self.ranks.append(identity)
        if len(self.ranks) != expected_ranks:
            raise RuntimeError(f'expected {expected_ranks} owned SR ranks, found {len(self.ranks)}')

    def check(self):
        for rank in self.ranks:
            try:
                current = _process_identity(self.proc_root / str(rank['pid']))
            except FileNotFoundError:
                return False
            if (current['start_ticks'] != rank['start_ticks']
                    or current['pgrp'] != rank['pgrp'] or current['state'] in {'Z', 'X'}):
                return False
        return True


class CompositeWorkerRecovery:
    """Exit on engine death or unconfirmed settlement, never on OOM classification.

    The legacy recovery_probe argument remains accepted by the video startup
    hook but is never called. Request-scoped failures can continue indefinitely.
    """
    def __init__(self, manager, endpoint, handler, sr_process, *,
                 drain_timeout_s=30.0, cleanup_timeout_s=20.0, exit_fn=os._exit,
                 shutdown_event=None, recovery_probe=None, probe_timeout_s=90.0,
                 pause_timeout_s=35.0, health_timeout_s=2.0,
                 unhealthy_timeout_s=10.0, ranks_factory=OwnedSRRanks):
        if min(drain_timeout_s, cleanup_timeout_s, pause_timeout_s,
               health_timeout_s, unhealthy_timeout_s) <= 0:
            raise ValueError('video supervision timeouts must be positive')
        self.manager, self.endpoint = manager, endpoint
        self.handler, self.sr_process = handler, sr_process
        self.drain_timeout_s, self.cleanup_timeout_s = drain_timeout_s, cleanup_timeout_s
        self.pause_timeout_s = pause_timeout_s
        self.health_timeout_s, self.unhealthy_timeout_s = health_timeout_s, unhealthy_timeout_s
        self.ranks_factory = ranks_factory
        self.exit_fn, self.shutdown_event = exit_fn, shutdown_event
        self.started = False
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self.watch())

    async def watch(self):
        unhealthy_since = None
        stopped_since = {}
        admission_paused = False
        try:
            ranks = (self.ranks_factory(self.sr_process.process, self.sr_process.config['gpus'])
                     if self.sr_process is not None else None)
            while True:
                process = self.sr_process.process if self.sr_process is not None else None
                if self.sr_process is not None and (process is None or process.returncode is not None):
                    await self.recover('sr_process_exited')
                    return
                if ranks is not None and not ranks.check():
                    await self.recover('sr_rank_exited')
                    return
                if bool(getattr(self.manager.handler, 'quarantined', False)):
                    await self.recover('execution_not_settled')
                    return
                pause_started = getattr(self.manager.handler, 'pause_started', None)
                if pause_started is not None and time.monotonic() - pause_started > self.pause_timeout_s:
                    await self.recover('settlement_timeout')
                    return
                # Cancellation may get stuck inside the underlying generator's
                # abort/finally code before the video wrapper can set its own
                # pause flag. Supervise the owned detached executions too.
                now = time.monotonic()
                stopped = [item for item in list(getattr(self.manager, '_running', {}).values())
                           if not item.execution.done() and item.context.is_stopped()]
                stopped_since = {id(item.execution): stopped_since.get(id(item.execution), now)
                                 for item in stopped}
                if stopped and self.manager._accepting:
                    self.manager._accepting = False
                    self.manager.components_ready = False
                    admission_paused = True
                if any(now - started > self.pause_timeout_s for started in stopped_since.values()):
                    await self.recover('cancelled_execution_not_settled')
                    return
                if not stopped and admission_paused:
                    if self.shutdown_event is None or not self.shutdown_event.is_set():
                        self.manager._accepting = True
                        self.manager.components_ready = False
                    admission_paused = False
                try:
                    await asyncio.wait_for(self.handler.engine_client.check_health(), self.health_timeout_s)
                    unhealthy_since = None
                except Exception as exc:
                    if 'EngineDeadError' in {cls.__name__ for cls in type(exc).__mro__}:
                        await self.recover('h3_engine_dead')
                        return
                    unhealthy_since = unhealthy_since or time.monotonic()
                    if time.monotonic() - unhealthy_since >= self.unhealthy_timeout_s:
                        await self.recover('h3_health_unavailable')
                        return
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception('video_worker_supervisor_failed')
            await self.recover('supervisor_failed')

    async def recover(self, reason):
        if self.started:
            return
        self.started = True
        self.manager._accepting = False
        self.manager.components_ready = False
        if self.shutdown_event is not None:
            self.shutdown_event.set()
        watchdog = threading.Timer(self.drain_timeout_s + self.cleanup_timeout_s + 10,
                                   self.exit_fn, args=(70,))
        watchdog.daemon = True
        watchdog.start()
        logger.error('video_worker_recovery_started reason=%s policy=exit_and_restart processes=%s',
                     reason, process_snapshot())
        try:
            try:
                await asyncio.wait_for(self.manager.shutdown(), self.drain_timeout_s)
                logger.error('video_worker_recovery_tasks_settled')
            except Exception:
                logger.exception('video_worker_recovery_drain_incomplete')

            async def cleanup():
                try:
                    await self.endpoint.unregister_endpoint_instance()
                finally:
                    try:
                        if self.sr_process is not None:
                            await self.sr_process.close()
                    finally:
                        await asyncio.to_thread(self.handler.cleanup)

            try:
                await asyncio.wait_for(cleanup(), self.cleanup_timeout_s)
            except Exception:
                logger.exception('video_worker_recovery_cleanup_incomplete')
        finally:
            logger.error('video_worker_recovery_exit code=70 reason=%s', reason)
            self.exit_fn(70)
            watchdog.cancel()

    async def close(self):
        if self.task is not None and not self.started:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
