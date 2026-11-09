"""Bounded HTTP compiler transport. Compilation is never automatically retried."""
import asyncio
import json
import aiohttp
from dingo.video_gateway.file_io import opened_file, run_file_io


class UpstreamUnknown(RuntimeError):
    pass


class UpstreamRejected(RuntimeError):
    pass


class CompilerHTTP:
    def __init__(self, session, backend, max_bytes):
        self.session, self.backend, self.max_bytes = session, backend, max_bytes

    async def json(self, method, suffix, **kwargs):
        timeout = kwargs.pop("timeout", self.backend.prepare_timeout_s)
        async with self.session.request(method, self.backend.url.rstrip("/")+suffix,
                                        timeout=aiohttp.ClientTimeout(total=timeout, connect=self.backend.connect_timeout_s),
                                        allow_redirects=False, **kwargs) as response:
            chunks, size = [], 0
            async for chunk in response.content.iter_chunked(65536):
                size += len(chunk)
                if size > self.max_bytes:
                    raise UpstreamUnknown("compiler response exceeds configured limit")
                chunks.append(chunk)
            try:
                value = json.loads(b"".join(chunks))
            except (ValueError, UnicodeDecodeError) as exc:
                raise UpstreamUnknown("compiler returned invalid JSON") from exc
            return response.status, value

    async def upload(self, digest, path, lease):
        for attempt in range(3):
            try:
                async with opened_file(path.open, "rb") as stream:
                    async def body():
                        while chunk := await run_file_io(stream.read, 1024*1024):
                            yield chunk
                    status, response = await self.json("PUT", f"/v1/assets/{digest}", data=body(),
                        headers={"Content-Type": "application/octet-stream",
                                 "Content-Length": str(path.stat().st_size), "X-H3IR-Asset-Lease": lease})
                if 200 <= status < 300:
                    return response
                if status < 500:
                    raise UpstreamRejected("compiler rejected uploaded asset")
            except (aiohttp.ClientError, TimeoutError):
                if attempt == 2:
                    raise
            if attempt < 2:
                await asyncio.sleep((1, 3)[attempt])
        raise UpstreamRejected("compiler asset upload failed")

    async def compile_once(self, payload, lease, timeout):
        # The caller commits call intent before this method and persists the full
        # response before deciding whether a task can release its physical slot.
        return await self.json("POST", "/v1/briefs", json=payload,
                               headers={"X-H3IR-Asset-Lease": lease}, timeout=timeout)
