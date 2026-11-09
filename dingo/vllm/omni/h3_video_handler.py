# SPDX-License-Identifier: Apache-2.0
"""H3 video failure settlement shared by base-only and composite video workers."""
import asyncio
import logging
import time
from dingo.common.video_worker_error import VideoWorkerFailure, failed_chunk_error, classify_worker_error

H3_CANCEL_SETTLE_TIMEOUT_S = 300

class H3VideoHandler:
    def __init__(self, handler, *, healthy=lambda: True, quarantine=lambda reason: None):
        self.handler = handler
        self.healthy = healthy
        self._outer_quarantine = quarantine
        self.quarantined = False
        self.quarantine_reason = None
        self.last_failure = None
        self.pause_started = None

    async def generate(self, payload, context):
        if not self.video_ready:
            raise VideoWorkerFailure("video Worker is unavailable", {
                "schema_version": 1, "reason": "engine_unavailable", "stage": "worker_execution"})
        async for chunk in self._generate_base(payload, context):
            yield chunk

    def pause(self):
        if self.pause_started is None:
            self.pause_started = time.monotonic()

    def resume(self):
        self.pause_started = None

    def quarantine(self, reason):
        if not self.quarantined:
            logging.getLogger(__name__).error("video_worker_quarantined reason=%s recovery=exit_and_restart", reason)
        self.quarantined = True
        self.quarantine_reason = reason
        self._outer_quarantine(reason)

    @property
    def video_ready(self):
        return not self.quarantined and self.pause_started is None and self.healthy()

    async def _confirm_failed_base_settled(self):
        """Cancellation fence only; OOM never attempts engine reuse.

        AsyncOmni aborts delivery before its synchronous GPU invocation finishes.
        The executor queues this RPC behind that invocation. Keep the video slot
        until every rank synchronizes; do not submit another forward as a probe.
        """
        engine = self.handler.engine_client
        clients = [client for pool in engine.engine.stage_pools
                   for client in pool.clients if client is not None]
        if not clients:
            raise RuntimeError("no live H3 diffusion clients after cancellation")
        async def synchronize(client):
            await client.collective_rpc_async(
                "synchronize_device", timeout=H3_CANCEL_SETTLE_TIMEOUT_S,
                kwargs={"timeout": H3_CANCEL_SETTLE_TIMEOUT_S},
            )
        await asyncio.wait_for(
            asyncio.gather(*(synchronize(client) for client in clients)),
            timeout=H3_CANCEL_SETTLE_TIMEOUT_S + 2,
        )

    async def _finish_cancellation(self):
        # The detached manager sends Task.cancel() after its abort grace. That
        # second cancellation must not cancel the GPU fence and falsely turn a
        # normal abort into engine corruption. The fence owns its own deadline.
        fence = asyncio.create_task(self._confirm_failed_base_settled())
        while True:
            try:
                return await asyncio.shield(fence)
            except asyncio.CancelledError:
                if fence.done():
                    return fence.result()

    def _record_failure(self, exc):
        failure = classify_worker_error(exc, stage="base_generation")
        # A local formatter may omit metadata; inspect only explicit OOM
        # signatures using the existing classifier, never generic error text.
        if failure["reason"] == "unknown":
            explicit = classify_worker_error(RuntimeError(str(exc)), stage="base_generation")
            if explicit["reason"] == "resource_exhausted":
                failure = explicit
        self.last_failure = failure
        if failure["reason"] == "resource_exhausted":
            self.quarantine("video OOM requires process restart")
        return failure

    async def _generate_base(self, payload, context):
        """OOM exits the video Worker; cancellation retains its GPU permit."""
        terminal = None
        settled = False
        try:
            async for chunk in self.handler.generate(payload, context):
                if chunk.get("status") in {"completed", "failed", "cancelled"}:
                    if terminal is not None:
                        raise RuntimeError("H3 returned multiple terminal results")
                    terminal = chunk
                if chunk.get("status") != "failed":
                    yield chunk
            if terminal is None:
                # AsyncOmni abort can end the stream without a terminal chunk.
                # A stopped caller must use the bounded cancellation fence,
                # rather than treating that normal abort shape as corruption.
                if context is not None and context.is_stopped():
                    raise asyncio.CancelledError
                raise RuntimeError("H3 stream ended without a terminal result")
            if terminal.get("status") == "failed":
                failure = failed_chunk_error(terminal, stage="base_generation")
                failure.internal_error = self._record_failure(failure)
                settled = True
                raise failure
            # A cancelled frontend stream does not prove GPU execution stopped.
            if terminal.get("status") == "cancelled":
                raise asyncio.CancelledError
            settled = True
        except BaseException as exc:
            if not settled and isinstance(exc, asyncio.CancelledError):
                self.pause()
                try:
                    await self._finish_cancellation()
                    settled = True
                except BaseException:
                    logging.getLogger(__name__).warning(
                        "H3 cancellation could not confirm rank settlement", exc_info=True
                    )
                finally:
                    self.resume()
            if not settled:
                self._record_failure(exc)
                if not self.quarantined:
                    self.quarantine("H3 execution ended without confirmed settlement")
            raise
