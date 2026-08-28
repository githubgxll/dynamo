# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64

import pytest

from dingo.video_gateway.errors import StoreUnavailable
from dingo.video_gateway.etcd_http import (
    EtcdHttpClient,
    EtcdWatchCompacted,
)


class PagedEtcd(EtcdHttpClient):
    def __init__(self) -> None:
        super().__init__("http://unused")
        self.requests: list[dict] = []
        self.values = {
            f"/tasks/video-{index:02d}": f"value-{index}".encode()
            for index in range(5)
        }

    async def _post(self, path, payload):
        assert path == "/v3/kv/range"
        self.requests.append(payload)
        start = base64.b64decode(payload["key"]).decode()
        end = base64.b64decode(payload["range_end"]).decode()
        limit = int(payload.get("limit", 0))
        matches = [
            (key, value)
            for key, value in sorted(self.values.items())
            if start <= key < end
        ]
        selected = matches[:limit] if limit else matches
        return {
            "header": {"revision": "123"},
            "more": len(matches) > len(selected),
            "kvs": [
                {
                    "key": base64.b64encode(key.encode()).decode(),
                    "value": base64.b64encode(value).decode(),
                    "create_revision": "1",
                    "mod_revision": "2",
                    "version": "1",
                }
                for key, value in selected
            ],
        }


async def test_range_all_uses_advancing_cursor_and_fixed_revision():
    client = PagedEtcd()

    values, revision = await client.range_all(
        "/tasks/",
        prefix=True,
        page_size=2,
    )

    assert [value.key for value in values] == sorted(client.values)
    assert revision == 123
    assert len(client.requests) == 3
    assert "revision" not in client.requests[0]
    assert [request["revision"] for request in client.requests[1:]] == ["123", "123"]
    second_cursor = base64.b64decode(client.requests[1]["key"])
    assert second_cursor == b"/tasks/video-01\0"


class BatchEtcd(EtcdHttpClient):
    def __init__(self) -> None:
        super().__init__("http://unused")
        self.payload: dict | None = None

    async def _post(self, path, payload):
        assert path == "/v3/kv/txn"
        self.payload = payload
        responses = []
        for operation in payload["success"]:
            key = base64.b64decode(operation["request_range"]["key"])
            if key == b"/tasks/missing":
                responses.append({"response_range": {}})
            else:
                responses.append(
                    {
                        "response_range": {
                            "kvs": [
                                {
                                    "key": base64.b64encode(key).decode(),
                                    "value": base64.b64encode(b"value").decode(),
                                    "create_revision": "1",
                                    "mod_revision": "40",
                                    "version": "1",
                                }
                            ]
                        }
                    }
                )
        return {"header": {"revision": "44"}, "responses": responses}


async def test_get_many_uses_one_fixed_revision_transaction():
    client = BatchEtcd()

    values, revision = await client.get_many(
        ["/tasks/a", "/tasks/missing"], revision=40
    )

    assert values[0] is not None and values[0].key == "/tasks/a"
    assert values[1] is None
    assert revision == 44
    assert client.payload is not None
    assert [
        operation["request_range"]["revision"]
        for operation in client.payload["success"]
    ] == ["40", "40"]


async def test_count_prefix_and_descending_are_sent_to_etcd():
    client = PagedEtcd()

    count = await client.count_prefix("/tasks/")
    page = await client.range_page(
        "/tasks/", prefix=True, limit=2, descending=True
    )

    assert count == 5
    assert page.values
    assert client.requests[-1]["sort_order"] == "DESCEND"


class LeaseEtcd(EtcdHttpClient):
    def __init__(self) -> None:
        super().__init__("http://unused")
        self.requests: list[tuple[str, dict]] = []

    async def _post(self, path, payload):
        self.requests.append((path, payload))
        if path == "/v3/lease/grant":
            return {"header": {"revision": "7"}, "ID": "42", "TTL": "9"}
        if path == "/v3/lease/timetolive":
            return {
                "header": {"revision": "8"},
                "ID": "42",
                "TTL": "8",
                "grantedTTL": "9",
                "keys": [base64.b64encode(b"/owner/one").decode()],
            }
        if path == "/v3/lease/revoke":
            return {"header": {"revision": "9"}}
        raise AssertionError(path)

    async def _stream_post(self, path, payload):
        self.requests.append((path, payload))
        yield {
            "result": {
                "header": {"revision": "10"},
                "ID": "42",
                "TTL": "9",
            }
        }


async def test_lease_contract_and_attached_put():
    client = LeaseEtcd()

    granted = await client.lease_grant(9)
    alive = await client.lease_keepalive(granted.lease_id)
    remaining = await client.lease_time_to_live(granted.lease_id, keys=True)
    revoked_revision = await client.lease_revoke(granted.lease_id)

    assert (granted.lease_id, granted.ttl, granted.revision) == (42, 9, 7)
    assert (alive.lease_id, alive.ttl, alive.revision) == (42, 9, 10)
    assert (remaining.ttl, remaining.granted_ttl, remaining.keys) == (
        8,
        9,
        ("/owner/one",),
    )
    assert revoked_revision == 9
    put = client.put("/owner/one", "value", lease_id=42)
    assert put["request_put"]["lease"] == "42"


@pytest.mark.parametrize("ttl", [0, -1])
async def test_lease_grant_rejects_invalid_ttl(ttl):
    with pytest.raises(ValueError, match="ttl"):
        await LeaseEtcd().lease_grant(ttl)


class WatchEtcd(EtcdHttpClient):
    def __init__(self, responses) -> None:
        super().__init__("http://unused")
        self.responses = responses
        self.request: tuple[str, dict] | None = None

    async def _stream_post(self, path, payload):
        self.request = (path, payload)
        for response in self.responses:
            yield response


def _watch_kv(key: str, value: bytes, *, revision: int) -> dict:
    return {
        "key": base64.b64encode(key.encode()).decode(),
        "value": base64.b64encode(value).decode(),
        "create_revision": str(revision),
        "mod_revision": str(revision),
        "version": "1",
        "lease": "42",
    }


async def test_watch_prefix_decodes_create_progress_and_events():
    client = WatchEtcd(
        [
            {
                "result": {
                    "header": {"revision": "20"},
                    "watch_id": "7",
                    "created": True,
                }
            },
            {
                "result": {
                    "header": {"revision": "21"},
                    "watch_id": "7",
                    "events": [
                        {
                            "type": "PUT",
                            "kv": _watch_kv("/workers/a", b"ready", revision=21),
                            "prev_kv": _watch_kv(
                                "/workers/a", b"busy", revision=20
                            ),
                        }
                    ],
                }
            },
        ]
    )

    responses = [
        response
        async for response in client.watch_prefix(
            "/workers/", start_revision=20, previous=True
        )
    ]

    assert responses[0].created is True
    assert responses[1].events[0].event_type == "PUT"
    assert responses[1].events[0].value.key == "/workers/a"
    assert responses[1].events[0].value.value == b"ready"
    assert responses[1].events[0].value.lease == 42
    assert responses[1].events[0].previous.value == b"busy"
    assert client.request is not None
    path, payload = client.request
    assert path == "/v3/watch"
    assert payload["create_request"]["start_revision"] == "20"
    assert payload["create_request"]["prev_kv"] is True


async def test_watch_prefix_reports_compaction_separately():
    client = WatchEtcd(
        [
            {
                "result": {
                    "header": {"revision": "30"},
                    "watch_id": "8",
                    "canceled": True,
                    "compact_revision": "29",
                    "cancel_reason": "etcdserver: mvcc: required revision has been compacted",
                }
            }
        ]
    )

    with pytest.raises(EtcdWatchCompacted) as captured:
        async for _response in client.watch_prefix("/workers/", start_revision=2):
            pass

    assert captured.value.compact_revision == 29


async def test_watch_prefix_reports_server_cancel():
    client = WatchEtcd(
        [
            {
                "result": {
                    "header": {"revision": "30"},
                    "watch_id": "8",
                    "canceled": True,
                    "cancel_reason": "permission denied",
                }
            }
        ]
    )

    with pytest.raises(StoreUnavailable, match="permission denied"):
        async for _response in client.watch_prefix("/workers/"):
            pass
