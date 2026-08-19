# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small etcd v3 JSON gRPC-gateway client with explicit CAS transactions."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import aiohttp

from dingo.video_gateway.errors import StoreUnavailable


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


class EtcdHttpClient:
    def __init__(self, url: str, *, timeout_s: float = 5.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=aiohttp.TCPConnector(limit=64, limit_per_host=64),
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = await self._get_session()
        try:
            async with session.post(self.url + path, json=payload) as response:
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
            raise StoreUnavailable(f"etcd {path} request failed: {exc}") from exc

    async def range(
        self,
        key: str,
        *,
        prefix: bool = False,
        limit: int = 0,
        keys_only: bool = False,
    ) -> list[EtcdValue]:
        key_bytes = key.encode()
        payload: dict[str, Any] = {"key": _b64(key_bytes)}
        if prefix:
            payload.update(
                {
                    "range_end": _b64(_prefix_end(key_bytes)),
                    "sort_order": "ASCEND",
                    "sort_target": "KEY",
                }
            )
        if limit:
            payload["limit"] = str(limit)
        if keys_only:
            payload["keys_only"] = True
        response = await self._post("/v3/kv/range", payload)
        result: list[EtcdValue] = []
        for item in response.get("kvs", []):
            result.append(
                EtcdValue(
                    key=base64.b64decode(item["key"]).decode(),
                    value=base64.b64decode(item.get("value", "")),
                    create_revision=int(item.get("create_revision", 0)),
                    mod_revision=int(item.get("mod_revision", 0)),
                    version=int(item.get("version", 0)),
                )
            )
        return result

    async def get(self, key: str) -> EtcdValue | None:
        values = await self.range(key, limit=1)
        return values[0] if values else None

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
    def put(key: str, value: bytes | str) -> dict:
        return {"request_put": {"key": _b64(key), "value": _b64(value)}}

    @staticmethod
    def delete(key: str) -> dict:
        return {"request_delete_range": {"key": _b64(key)}}
