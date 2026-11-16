# SPDX-License-Identifier: Apache-2.0
"""Wakeup, shutdown and error supervision shared by finite-task dispatchers."""
import asyncio


class DispatchLoop:
    """Business adapters perform one scheduling turn and report more ready work."""

    def __init__(
        self, *, stop, advance, interval_s, on_error, wakeup=None, error_interval_s=None
    ):
        if interval_s <= 0:
            raise ValueError("dispatch interval must be positive")
        self.stop, self.advance = stop, advance
        self.wakeup = wakeup if wakeup is not None else asyncio.Event()
        self.interval_s = interval_s
        self.error_interval_s = (
            max(0.1, interval_s) if error_interval_s is None else error_interval_s
        )
        self.on_error = on_error

    async def _wait(self, delay):
        stopped = asyncio.create_task(self.stop.wait())
        ready = asyncio.create_task(self.wakeup.wait())
        try:
            await asyncio.wait(
                [stopped, ready], timeout=delay, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for waiter in (stopped, ready):
                waiter.cancel()
            await asyncio.gather(stopped, ready, return_exceptions=True)

    async def run(self):
        while not self.stop.is_set():
            # Clear before I/O so a submission during the turn isn't lost.
            self.wakeup.clear()
            delay = self.interval_s
            try:
                if await self.advance():
                    await asyncio.sleep(0)
                    continue
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                # The injected callback logs the active business exception.
                self.on_error()
                delay = self.error_interval_s
            await self._wait(delay)
