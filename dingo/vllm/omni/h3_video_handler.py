# SPDX-License-Identifier: Apache-2.0
"""H3 video failure settlement shared by base-only and composite video workers."""
import asyncio
import logging
import time
from dingo.common.video_worker_error import VideoWorkerFailure, failed_chunk_error, classify_worker_error

H3_FAILED_BASE_SETTLE_TIMEOUT_S = 30

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
        """Confirm all ranks left the failed invocation; never run a new inference."""
        engine = self.handler.engine_client
        await asyncio.wait_for(engine.check_health(), timeout=2)
        pools = engine.engine.stage_pools
        clients = [client for pool in pools for client in pool.clients if client is not None]
        if not clients:
            raise RuntimeError("no live H3 diffusion clients after failed request")
        async def synchronize(client):
            await client.collective_rpc_async(
                "synchronize_device",
                timeout=H3_FAILED_BASE_SETTLE_TIMEOUT_S,
                kwargs={"timeout": H3_FAILED_BASE_SETTLE_TIMEOUT_S},
            )
        await asyncio.wait_for(
            asyncio.gather(*(synchronize(client) for client in clients)),
            timeout=H3_FAILED_BASE_SETTLE_TIMEOUT_S + 2,
        )

    async def _generate_base(self, payload, context):
        """A settled request failure does not quarantine a living engine."""
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
                if failure.internal_error["reason"] != "input_processing_failed":
                    self.pause()
                    try:
                        await self._confirm_failed_base_settled()
                    except Exception as settlement_error:
                        # Keep the model failure as the task's primary cause.
                        # Settlement still failed: leave settled=False so the
                        # outer handler quarantines this Worker as before.
                        logging.getLogger(__name__).error(
                            "video_base_failure_settlement_failed reason=%s stage=%s settlement=%s",
                            failure.internal_error["reason"],
                            failure.internal_error["stage"],
                            "timeout" if isinstance(settlement_error, TimeoutError) else "error",
                            exc_info=True,
                        )
                        raise failure from settlement_error
                    finally:
                        self.resume()
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
                    await self._confirm_failed_base_settled()
                    settled = True
                except BaseException:
                    logging.getLogger(__name__).warning(
                        "H3 cancellation could not confirm rank settlement", exc_info=True
                    )
                finally:
                    self.resume()
            if not settled:
                self.last_failure = classify_worker_error(exc, stage="base_generation")
                self.quarantine("H3 execution ended without confirmed settlement")
            raise
