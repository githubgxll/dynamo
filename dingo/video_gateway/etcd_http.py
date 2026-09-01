# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small etcd v3 JSON gRPC-gateway client with explicit CAS transactions."""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import aiohttp

from dingo.video_gateway.errors import StoreUnavailable
from dingo.video_gateway.telemetry import GatewayTelemetry


def _b64(value: bytes | str) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return base64.b64encode(raw).decode("ascii")


def _prefix_end(prefix: bytes) -> bytes:
    value = bytearray(prefix)
    for index in range(len(value) - 1, -1, -1):
        if value[index] < 0xFF:
            value[index] += 1
            return bytes(value[: index + 1])
    return b"\0"


@dataclass(frozen=True, slots=True)
class EtcdValue:
    key: str
    value: bytes
    create_revision: int
    mod_revision: int
    version: int
    lease: int = 0


@dataclass(frozen=True, slots=True)
class EtcdRangePage:
    values: tuple[EtcdValue, ...]
    more: bool
    revision: int
    count: int = 0


@dataclass(frozen=True, slots=True)
class EtcdLease:
    lease_id: int
    ttl: int
    revision: int
    granted_ttl: int = 0
    keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EtcdWatchEvent:
    event_type: str
    value: EtcdValue
    previous: EtcdValue | None = None


@dataclass(frozen=True, slots=True)
class EtcdWatchResponse:
    watch_id: int
    revision: int
    events: tuple[EtcdWatchEvent, ...]
    created: bool = False


class EtcdWatchCompacted(StoreUnavailable):
    """Raised when a requested watch revision is no longer available."""

    def __init__(self, compact_revision: int, reason: str = "") -> None:
        self.compact_revision = compact_revision
        suffix = f": {reason}" if reason else ""
        super().__init__(
            f"etcd watch revision was compacted at {compact_revision}{suffix}"
        )


class EtcdHttpClient:
    def __init__(
        self,
        url: str | Sequence[str],
        *,
        timeout_s: float = 5.0,
        telemetry: GatewayTelemetry | None = None,
    ) -> None:
        raw_urls = (url,) if isinstance(url, str) else tuple(url)
        if not raw_urls:
            raise ValueError("at least one etcd endpoint is required")
        self.urls = tuple(item.rstrip("/") for item in raw_urls)
        if any(
            not item or not item.startswith(("http://", "https://"))
            for item in self.urls
        ):
            raise ValueError("etcd endpoints must be HTTP or HTTPS URLs")
        if len(set(self.urls)) != len(self.urls):
            raise ValueError("etcd endpoints must not contain duplicates")
        # Keep the original attribute for diagnostics and compatibility. New
        # request code uses the endpoint set below.
        self.url = self.urls[0]
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self.telemetry = telemetry
        self._session: aiohttp.ClientSession | None = None
        self._endpoint_index = 0
        self._endpoint_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=aiohttp.TCPConnector(limit=64, limit_per_host=64),
            )
        return self._session

    @staticmethod
    def prefix_end(prefix: str | bytes) -> bytes:
        raw = prefix.encode() if isinstance(prefix, str) else prefix
        return _prefix_end(raw)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _endpoint_order(self) -> tuple[tuple[int, str], ...]:
        async with self._endpoint_lock:
            start = self._endpoint_index
        return tuple(
            (
                (start + offset) % len(self.urls),
                self.urls[(start + offset) % len(self.urls)],
            )
            for offset in range(len(self.urls))
        )

    async def _mark_endpoint(self, index: int, *, succeeded: bool) -> None:
        async with self._endpoint_lock:
            if succeeded:
                self._endpoint_index = index
            elif self._endpoint_index == index:
                self._endpoint_index = (index + 1) % len(self.urls)

    async def _post_once(
        self, endpoint: str, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        session = await self._get_session()
        try:
            async with session.post(endpoint + path, json=payload) as response:
                body = await response.text()
                if response.status < 200 or response.status >= 300:
                    raise StoreUnavailable(
                        f"etcd {path} returned HTTP {response.status}: {body[:512]}"
                    )
                if not body:
                    return {}
                try:
                    value = await response.json()
                except (aiohttp.ContentTypeError, ValueError) as exc:
                    raise StoreUnavailable(
                        f"etcd {path} returned invalid JSON"
                    ) from exc
                if "error" in value:
                    raise StoreUnavailable(f"etcd {path} failed: {value['error']}")
                return value
        except StoreUnavailable:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise StoreUnavailable(
                f"etcd endpoint {endpoint} {path} request failed: {exc}"
            ) from exc

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        retry_safe: bool = False,
    ) -> dict[str, Any]:
        endpoints = await self._endpoint_order()
        last_error: StoreUnavailable | None = None
        for position, (index, endpoint) in enumerate(endpoints):
            started = time.monotonic()
            succeeded = False
            try:
                value = await self._post_once(endpoint, path, payload)
                succeeded = True
                await self._mark_endpoint(index, succeeded=True)
                return value
            except StoreUnavailable as exc:
                last_error = exc
                await self._mark_endpoint(index, succeeded=False)
                if not retry_safe or position + 1 == len(endpoints):
                    raise
            finally:
                if self.telemetry is not None:
                    self.telemetry.record_etcd_request(
                        path.removeprefix("/v3/"),
                        time.monotonic() - started,
                        succeeded=succeeded,
                    )
        assert last_error is not None
        raise last_error

    async def _stream_post(
        self, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield newline-delimited responses from an etcd streaming RPC."""

        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=self.timeout.total,
            sock_read=None,
        )
        endpoints = await self._endpoint_order()
        for position, (index, endpoint) in enumerate(endpoints):
            session = await self._get_session()
            yielded = False
            try:
                async with session.post(
                    endpoint + path,
                    json=payload,
                    timeout=timeout,
                ) as response:
                    if response.status < 200 or response.status >= 300:
                        body = await response.text()
                        raise StoreUnavailable(
                            f"etcd {path} returned HTTP {response.status}: {body[:512]}"
                        )
                    while True:
                        line = await response.content.readline()
                        if not line:
                            if response.content.at_eof():
                                if yielded:
                                    return
                                raise StoreUnavailable(
                                    f"etcd {path} stream ended before a response"
                                )
                            continue
                        try:
                            value = json.loads(line)
                        except (UnicodeDecodeError, ValueError) as exc:
                            raise StoreUnavailable(
                                f"etcd {path} returned invalid streaming JSON"
                            ) from exc
                        if "error" in value:
                            raise StoreUnavailable(
                                f"etcd {path} failed: {value['error']}"
                            )
                        await self._mark_endpoint(index, succeeded=True)
                        yielded = True
                        yield value
                return
            except asyncio.CancelledError:
                raise
            except StoreUnavailable:
                await self._mark_endpoint(index, succeeded=False)
                if yielded or position + 1 == len(endpoints):
                    raise
            except (aiohttp.ClientError, TimeoutError) as exc:
                await self._mark_endpoint(index, succeeded=False)
                if yielded or position + 1 == len(endpoints):
                    raise StoreUnavailable(
                        f"etcd endpoint {endpoint} {path} stream failed: {exc}"
                    ) from exc

    @staticmethod
    def _decode_value(item: dict[str, Any]) -> EtcdValue:
        return EtcdValue(
            key=base64.b64decode(item["key"]).decode(),
            value=base64.b64decode(item.get("value", "")),
            create_revision=int(item.get("create_revision", 0)),
            mod_revision=int(item.get("mod_revision", 0)),
            version=int(item.get("version", 0)),
            lease=int(item.get("lease", 0)),
        )

    async def range_page(
        self,
        key: str | bytes,
        *,
        prefix: bool = False,
        limit: int = 0,
        keys_only: bool = False,
        count_only: bool = False,
        descending: bool = False,
        revision: int = 0,
        range_end: bytes | None = None,
    ) -> EtcdRangePage:
        if prefix and range_end is not None:
            raise ValueError("prefix and range_end are mutually exclusive")
        key_bytes = key.encode() if isinstance(key, str) else key
        payload: dict[str, Any] = {"key": _b64(key_bytes)}
        if prefix:
            range_end = _prefix_end(key_bytes)
        if range_end is not None:
            payload.update(
                {
                    "range_end": _b64(range_end),
                    "sort_order": "DESCEND" if descending else "ASCEND",
                    "sort_target": "KEY",
                }
            )
        if limit:
            payload["limit"] = str(limit)
        if keys_only:
            payload["keys_only"] = True
        if count_only:
            payload["count_only"] = True
        if revision:
            payload["revision"] = str(revision)
        response = await self._post("/v3/kv/range", payload, retry_safe=True)
        result = [self._decode_value(item) for item in response.get("kvs", [])]
        raw_more = response.get("more", False)
        more = raw_more is True or str(raw_more).lower() == "true"
        return EtcdRangePage(
            values=tuple(result),
            more=more,
            revision=int(response.get("header", {}).get("revision", 0)),
            count=int(response.get("count", len(result))),
        )

    async def range(
        self,
        key: str,
        *,
        prefix: bool = False,
        limit: int = 0,
        keys_only: bool = False,
        descending: bool = False,
    ) -> list[EtcdValue]:
        page = await self.range_page(
            key,
            prefix=prefix,
            limit=limit,
            keys_only=keys_only,
            descending=descending,
        )
        return list(page.values)

    async def count_prefix(self, prefix: str, *, revision: int = 0) -> int:
        page = await self.range_page(
            prefix,
            prefix=True,
            count_only=True,
            revision=revision,
        )
        return page.count

    async def range_all(
        self,
        key: str,
        *,
        prefix: bool = False,
        page_size: int = 512,
        keys_only: bool = False,
        revision: int = 0,
    ) -> tuple[list[EtcdValue], int]:
        """Read a complete key range from one fixed MVCC snapshot."""

        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if not prefix:
            page = await self.range_page(
                key,
                limit=page_size,
                keys_only=keys_only,
                revision=revision,
            )
            if page.more:
                raise StoreUnavailable("bounded exact-key etcd range was truncated")
            return list(page.values), revision or page.revision

        prefix_bytes = key.encode()
        range_end = _prefix_end(prefix_bytes)
        cursor = prefix_bytes
        snapshot_revision = revision
        result: list[EtcdValue] = []
        while True:
            page = await self.range_page(
                cursor,
                range_end=range_end,
                limit=page_size,
                keys_only=keys_only,
                revision=snapshot_revision,
            )
            if snapshot_revision == 0:
                snapshot_revision = page.revision
                if snapshot_revision <= 0:
                    raise StoreUnavailable("etcd range response omitted its revision")
            result.extend(page.values)
            if not page.more:
                return result, snapshot_revision
            if not page.values:
                raise StoreUnavailable("etcd range returned more=true without values")
            next_cursor = page.values[-1].key.encode() + b"\0"
            if next_cursor <= cursor or next_cursor >= range_end:
                raise StoreUnavailable("etcd range pagination cursor did not advance")
            cursor = next_cursor

    async def get(self, key: str) -> EtcdValue | None:
        values = await self.range(key, limit=1)
        return values[0] if values else None

    async def get_many(
        self,
        keys: Sequence[str],
        *,
        revision: int = 0,
    ) -> tuple[list[EtcdValue | None], int]:
        """Read exact keys in one transaction and one MVCC snapshot."""

        if len(keys) > 128:
            raise ValueError("get_many accepts at most 128 keys")
        if not keys:
            return [], revision
        success: list[dict[str, Any]] = []
        for key in keys:
            request: dict[str, Any] = {"key": _b64(key), "limit": "1"}
            if revision:
                request["revision"] = str(revision)
            success.append({"request_range": request})
        response = await self._post(
            "/v3/kv/txn",
            {"compare": [], "success": success, "failure": []},
            retry_safe=True,
        )
        responses = response.get("responses", [])
        if len(responses) != len(keys):
            raise StoreUnavailable("etcd get_many returned an invalid response count")
        result: list[EtcdValue | None] = []
        for item in responses:
            values = item.get("response_range", {}).get("kvs", [])
            result.append(self._decode_value(values[0]) if values else None)
        return result, int(response.get("header", {}).get("revision", 0))

    async def lease_grant(self, ttl: int, *, lease_id: int = 0) -> EtcdLease:
        if ttl <= 0:
            raise ValueError("lease ttl must be positive")
        if lease_id < 0:
            raise ValueError("lease id must not be negative")
        requested_id = lease_id or max(1, secrets.randbits(63))
        try:
            response = await self._post(
                "/v3/lease/grant",
                {"TTL": str(ttl), "ID": str(requested_id)},
                retry_safe=True,
            )
        except StoreUnavailable as exc:
            # A response can be lost after etcd committed the grant. Because
            # the client chooses one stable random ID, confirm that lease
            # instead of allocating a second one during endpoint failover.
            try:
                existing = await self.lease_time_to_live(requested_id)
            except StoreUnavailable:
                raise exc
            if existing.ttl > 0:
                return EtcdLease(
                    lease_id=requested_id,
                    ttl=existing.ttl,
                    granted_ttl=existing.granted_ttl or ttl,
                    revision=existing.revision,
                )
            raise exc
        granted_id = int(response.get("ID", 0))
        granted_ttl = int(response.get("TTL", 0))
        if granted_id != requested_id or granted_ttl <= 0:
            raise StoreUnavailable("etcd returned an invalid lease grant response")
        return EtcdLease(
            lease_id=granted_id,
            ttl=granted_ttl,
            granted_ttl=granted_ttl,
            revision=int(response.get("header", {}).get("revision", 0)),
        )

    async def lease_revoke(self, lease_id: int) -> int:
        if lease_id == 0:
            raise ValueError("lease id must not be zero")
        response = await self._post(
            "/v3/lease/revoke",
            {"ID": str(lease_id)},
            retry_safe=True,
        )
        return int(response.get("header", {}).get("revision", 0))

    async def lease_keepalive(self, lease_id: int) -> EtcdLease:
        if lease_id == 0:
            raise ValueError("lease id must not be zero")
        stream = self._stream_post(
            "/v3/lease/keepalive",
            {"ID": str(lease_id)},
        )
        try:
            response = await anext(stream)
        except StopAsyncIteration as exc:
            raise StoreUnavailable("etcd lease keepalive returned no response") from exc
        finally:
            await stream.aclose()
        result = response.get("result", response)
        if "error" in result:
            raise StoreUnavailable(
                f"etcd /v3/lease/keepalive failed: {result['error']}"
            )
        response_id = int(result.get("ID", 0))
        ttl = int(result.get("TTL", 0))
        if response_id != lease_id or ttl <= 0:
            raise StoreUnavailable("etcd lease keepalive reported an expired lease")
        return EtcdLease(
            lease_id=response_id,
            ttl=ttl,
            revision=int(result.get("header", {}).get("revision", 0)),
        )

    async def lease_time_to_live(
        self, lease_id: int, *, keys: bool = False
    ) -> EtcdLease:
        if lease_id == 0:
            raise ValueError("lease id must not be zero")
        response = await self._post(
            "/v3/lease/timetolive",
            {"ID": str(lease_id), "keys": keys},
            retry_safe=True,
        )
        return EtcdLease(
            lease_id=int(response.get("ID", 0)),
            ttl=int(response.get("TTL", 0)),
            granted_ttl=int(response.get("grantedTTL", 0)),
            revision=int(response.get("header", {}).get("revision", 0)),
            keys=tuple(
                base64.b64decode(key).decode() for key in response.get("keys", [])
            ),
        )

    async def watch_prefix(
        self,
        prefix: str,
        *,
        start_revision: int = 0,
        previous: bool = False,
        progress_notify: bool = True,
    ) -> AsyncIterator[EtcdWatchResponse]:
        if start_revision < 0:
            raise ValueError("watch start_revision must not be negative")
        prefix_bytes = prefix.encode()
        create_request: dict[str, Any] = {
            "key": _b64(prefix_bytes),
            "range_end": _b64(_prefix_end(prefix_bytes)),
            "progress_notify": progress_notify,
            "prev_kv": previous,
        }
        if start_revision:
            create_request["start_revision"] = str(start_revision)
        async for envelope in self._stream_post(
            "/v3/watch",
            {"create_request": create_request},
        ):
            result = envelope.get("result", envelope)
            if "error" in result:
                raise StoreUnavailable(f"etcd /v3/watch failed: {result['error']}")
            compact_revision = int(result.get("compact_revision", 0))
            if compact_revision:
                raise EtcdWatchCompacted(
                    compact_revision,
                    str(result.get("cancel_reason", "")),
                )
            if result.get("canceled"):
                raise StoreUnavailable(
                    "etcd watch was canceled"
                    + (
                        f": {result['cancel_reason']}"
                        if result.get("cancel_reason")
                        else ""
                    )
                )
            events: list[EtcdWatchEvent] = []
            for event in result.get("events", []):
                value = self._decode_value(event["kv"])
                previous_value = event.get("prev_kv")
                events.append(
                    EtcdWatchEvent(
                        event_type=str(event.get("type", "PUT")),
                        value=value,
                        previous=(
                            self._decode_value(previous_value)
                            if previous_value is not None
                            else None
                        ),
                    )
                )
            yield EtcdWatchResponse(
                watch_id=int(result.get("watch_id", 0)),
                revision=int(result.get("header", {}).get("revision", 0)),
                events=tuple(events),
                created=bool(result.get("created", False)),
            )

    async def txn(
        self,
        compare: Sequence[dict[str, Any]],
        success: Sequence[dict[str, Any]],
        failure: Sequence[dict[str, Any]] = (),
    ) -> tuple[bool, int]:
        response = await self._post(
            "/v3/kv/txn",
            {
                "compare": list(compare),
                "success": list(success),
                "failure": list(failure),
            },
        )
        revision = int(response.get("header", {}).get("revision", 0))
        return bool(response.get("succeeded")), revision

    @staticmethod
    def compare_version(key: str, version: int, *, result: str = "EQUAL") -> dict:
        return {
            "key": _b64(key),
            "target": "VERSION",
            "result": result,
            "version": str(version),
        }

    @staticmethod
    def compare_mod(key: str, revision: int) -> dict:
        return {
            "key": _b64(key),
            "target": "MOD",
            "result": "EQUAL",
            "mod_revision": str(revision),
        }

    @staticmethod
    def compare_value(key: str, value: bytes | str) -> dict:
        return {
            "key": _b64(key),
            "target": "VALUE",
            "result": "EQUAL",
            "value": _b64(value),
        }

    @staticmethod
    def put(
        key: str,
        value: bytes | str,
        *,
        lease_id: int = 0,
    ) -> dict:
        request = {"key": _b64(key), "value": _b64(value)}
        if lease_id:
            request["lease"] = str(lease_id)
        return {"request_put": request}

    @staticmethod
    def delete(key: str) -> dict:
        return {"request_delete_range": {"key": _b64(key)}}
