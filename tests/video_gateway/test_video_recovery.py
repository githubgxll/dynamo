# SPDX-License-Identifier: Apache-2.0
import asyncio
import runpy
import subprocess
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import AsyncMock

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'dingo/vllm/omni/video_recovery.py'))
Recovery, OwnedSRRanks = MODULE['CompositeWorkerRecovery'], MODULE['OwnedSRRanks']
Recovery.recover.__globals__['process_snapshot'] = lambda: []


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setup_recovery(self, drain=None, cleanup_error=False):
        calls = []
        manager = N(_accepting=True, components_ready=True,
                    handler=N(quarantined=False, pause_started=None))
        async def default_drain():
            self.assertFalse(manager._accepting)
            self.assertFalse(manager.components_ready)
            calls.append('settled')
        async def unregister(): calls.append('unregister')
        async def srclose(): calls.append('sr_close')
        def cleanup():
            calls.append('h3_cleanup')
            if cleanup_error: raise RuntimeError('cleanup failed')
        manager.shutdown = drain or default_drain
        engine = N(check_health=AsyncMock())
        ranks = N(check=lambda: True)
        sr = N(close=srclose, process=N(returncode=None), config={'gpus': 1})
        recovery = Recovery(manager, N(unregister_endpoint_instance=unregister),
                            N(cleanup=cleanup, engine_client=engine), sr,
                            drain_timeout_s=.03, cleanup_timeout_s=.03,
                            ranks_factory=lambda *args: ranks,
                            exit_fn=lambda code: calls.append(('exit', code)))
        return recovery, calls, ranks

    async def test_terminal_before_cleanup_and_exit_once(self):
        r, calls, _ = self.setup_recovery()
        r.shutdown_event = asyncio.Event()
        await r.recover('unsafe'); await r.recover('unsafe')
        self.assertTrue(r.shutdown_event.is_set())
        self.assertEqual(calls, ['settled', 'unregister', 'sr_close', 'h3_cleanup', ('exit', 70)])

    async def test_drain_timeout_still_exits(self):
        async def blocked(): await asyncio.sleep(10)
        r, calls, _ = self.setup_recovery(drain=blocked)
        await r.recover('unsafe')
        self.assertEqual(calls[-1], ('exit', 70))

    async def test_cleanup_error_still_exits(self):
        r, calls, _ = self.setup_recovery(cleanup_error=True)
        await r.recover('unsafe')
        self.assertEqual(calls[-1], ('exit', 70))

    async def test_busy_or_unready_alone_does_not_exit(self):
        r, calls, _ = self.setup_recovery()
        r.manager.components_ready = False
        r.start(); await asyncio.sleep(.02); await r.close()
        self.assertEqual(calls, [])

    async def test_oom_drains_then_unregisters_and_exits_without_probe(self):
        r, calls, _ = self.setup_recovery()
        r.manager.handler.last_failure = {'reason': 'resource_exhausted', 'stage': 'base_generation'}
        await r.watch()
        self.assertEqual(calls, ['settled', 'unregister', 'sr_close', ('exit', 70)])
        r.handler.engine_client.check_health.assert_not_awaited()

    async def test_dead_owned_service_exits(self):
        r, calls, _ = self.setup_recovery()
        r.sr_process.process.returncode = 1
        await r.watch()
        self.assertEqual(calls[-1], ('exit', 70))

    async def test_dead_owned_rank_exits_when_service_lives(self):
        r, calls, ranks = self.setup_recovery()
        ranks.check = lambda: False
        await r.watch()
        self.assertIsNone(r.sr_process.process.returncode)
        self.assertEqual(calls[-1], ('exit', 70))

    async def test_original_engine_monitor_owns_health_checks(self):
        r, calls, _ = self.setup_recovery()
        r.start(); await asyncio.sleep(.02); await r.close()
        r.handler.engine_client.check_health.assert_not_awaited()
        self.assertEqual(calls, [])

    async def test_unconfirmed_execution_exits(self):
        r, calls, _ = self.setup_recovery()
        r.manager.handler.quarantined = True
        await r.watch()
        self.assertEqual(calls[-1], ('exit', 70))

    async def test_settlement_pause_is_bounded(self):
        r, calls, _ = self.setup_recovery()
        r.manager.handler.pause_started = time.monotonic() - r.pause_timeout_s - 1
        await r.watch()
        self.assertEqual(calls[-1], ('exit', 70))

    async def test_transient_settlement_pause_does_not_exit(self):
        r, calls, _ = self.setup_recovery()
        r.manager.handler.pause_started = time.monotonic()
        r.start(); await asyncio.sleep(.02); await r.close()
        self.assertEqual(calls, [])

    async def test_stuck_cancelled_execution_is_bounded_before_wrapper_pause(self):
        r, calls, _ = self.setup_recovery()
        r.pause_timeout_s = .01
        r.manager._running = {'task': N(execution=N(done=lambda: False),
                                         context=N(is_stopped=lambda: True))}
        await asyncio.wait_for(r.watch(), 2)
        self.assertEqual(calls[-1], ('exit', 70))

    async def test_cancelled_execution_settles_and_readmission_waits_for_health(self):
        r, calls, _ = self.setup_recovery()
        done = False
        r.manager._running = {'task': N(execution=N(done=lambda: done),
                                         context=N(is_stopped=lambda: True))}
        r.start(); await asyncio.sleep(.02)
        self.assertFalse(r.manager._accepting)
        done = True
        await asyncio.sleep(.55); await r.close()
        self.assertTrue(r.manager._accepting)
        self.assertFalse(r.manager.components_ready)
        self.assertEqual(calls, [])

    async def test_live_busy_execution_does_not_start_cancellation_deadline(self):
        r, calls, _ = self.setup_recovery()
        r.pause_timeout_s = .01
        r.manager._running = {'task': N(execution=N(done=lambda: False),
                                         context=N(is_stopped=lambda: False))}
        r.start(); await asyncio.sleep(.03); await r.close()
        self.assertTrue(r.manager._accepting)
        self.assertEqual(calls, [])

    async def test_monitor_failure_cannot_silently_disable_supervision(self):
        r, calls, ranks = self.setup_recovery()
        def broken(): raise OSError('cannot read owned process')
        ranks.check = broken
        await r.watch()
        self.assertEqual(calls[-1], ('exit', 70))


class RankTrackingTests(unittest.TestCase):
    def test_real_rank_death_is_detected(self):
        code = "import ctypes,time;ctypes.CDLL(None).prctl(15,b'vLLM-Omni::Diff',0,0,0);print('READY',flush=True);time.sleep(30)"
        rank = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                                stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(rank.stdout.readline().strip(), 'READY')
            tracker = OwnedSRRanks(rank, 1)
            self.assertTrue(tracker.check())
            rank.terminate(); rank.wait(timeout=5)
            self.assertFalse(tracker.check())
        finally:
            if rank.poll() is None: rank.kill(); rank.wait()
            rank.stdout.close()

    def test_unrelated_process_group_and_transient_children_are_excluded(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for pid, name, group, ticks in [(10, 'vllm-omni', 10, 100),
                    (11, 'vLLM-Omni::Diff', 10, 101), (12, 'ffmpeg', 10, 102),
                    (20, 'vLLM-Omni::Diff', 20, 103)]:
                path = root / str(pid); path.mkdir()
                (path / 'comm').write_text(name)
                fields = ['S', '1', str(group)] + ['0'] * 16 + [str(ticks)]
                (path / 'stat').write_text(f'{pid} ({name}) ' + ' '.join(fields))
            tracker = OwnedSRRanks(N(pid=10), 1, proc_root=root)
            self.assertEqual([r['pid'] for r in tracker.ranks], [11])
            self.assertTrue(tracker.check())
            path = root / '11' / 'stat'
            path.write_text(path.read_text().replace('101', '999'))
            self.assertFalse(tracker.check(), 'PID reuse cannot hide rank death')


if __name__ == '__main__': unittest.main()

class OOMStatusOrder(unittest.IsolatedAsyncioTestCase):
    async def test_oom_error_persistence_precedes_unregister_and_exit(self):
        r,calls,_=RecoveryTests().setup_recovery()
        persisted=asyncio.Event()
        async def drain():
            await asyncio.sleep(.005)
            calls.append('failed_status_persisted')
            persisted.set()
        async def unregister():
            self.assertTrue(persisted.is_set())
            calls.append('unregister')
        r.manager.shutdown=drain
        r.endpoint.unregister_endpoint_instance=unregister
        await r.recover('oom')
        self.assertEqual(calls,['failed_status_persisted','unregister','sr_close',('exit',70)])
