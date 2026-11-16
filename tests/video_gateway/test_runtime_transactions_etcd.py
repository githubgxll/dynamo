# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local integration tests. Set ETCD_TEST_BINARY to an etcd executable.

Always starts a fresh loopback-only process; never takes a remote endpoint.
"""

import asyncio
import os
import socket

import pytest

from dingo.video_gateway.errors import StoreUnavailable
from dingo.video_gateway.etcd_http import EtcdHttpClient
from dingo.video_gateway.task_runtime.repository import (
    CasStatus,
    OwnerSnapshot,
    RuntimeKeys,
    ValidatedOutcome,
)
from dingo.video_gateway.task_runtime.transactions import (
    execute_runtime_transaction,
    settle_transaction,
)
from tests.video_gateway.test_runtime_transactions import (
    attempt_snapshot,
    plan,
    put,
    setup,
)


@pytest.fixture
async def runtime_etcd(tmp_path):
    binary = os.environ.get("ETCD_TEST_BINARY")
    if not binary:
        pytest.skip("set ETCD_TEST_BINARY to run isolated local etcd tests")
    sockets = [socket.socket(), socket.socket()]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        client_url, peer_url = [
            f"http://127.0.0.1:{s.getsockname()[1]}" for s in sockets
        ]
    finally:
        for sock in sockets:
            sock.close()
    with (tmp_path / "etcd.log").open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            binary,
            "--name",
            "runtime-test",
            "--data-dir",
            str(tmp_path / "data"),
            "--listen-client-urls",
            client_url,
            "--advertise-client-urls",
            client_url,
            "--listen-peer-urls",
            peer_url,
            "--initial-advertise-peer-urls",
            peer_url,
            "--initial-cluster",
            f"runtime-test={peer_url}",
            "--log-level",
            "error",
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
        )
        client = EtcdHttpClient(client_url, timeout_s=1)
        try:
            for _ in range(80):
                if process.returncode is not None:
                    pytest.fail("local etcd exited; inspect test etcd.log")
                try:
                    await client.get("/startup-probe")
                    break
                except StoreUnavailable:
                    await asyncio.sleep(0.1)
            else:
                pytest.fail("local etcd did not become ready")
            yield client
        finally:
            await client.close()
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=5)


async def test_real_etcd_competing_gateways_and_capacity(runtime_etcd):
    client = runtime_etcd
    route, owner, worker = await setup(client, 17)
    second = OwnerSnapshot("second", await put(client, "/second-owner", b"second"))
    keys = RuntimeKeys("deployment")
    for i in range(16):
        a, b = await asyncio.gather(
            plan(client, route, owner, worker, i, i),
            plan(client, route, second, worker, i, i),
        )
        results = await asyncio.gather(
            execute_runtime_transaction(client, *a[1:]),
            execute_runtime_transaction(client, *b[1:]),
        )
        assert {r.status for r in results} == {CasStatus.APPLIED, CasStatus.CONFLICT}
    assert (await client.get("/queued")).value == b"1"
    assert len(await client.range(keys.root + "/slots/", prefix=True)) == 16
    _, compare, writes = await plan(client, route, owner, worker, 16, 0)
    assert (
        await execute_runtime_transaction(client, compare, writes)
    ).status is CasStatus.CONFLICT
    assert await client.get("/queue/16") is not None


async def test_real_etcd_settle_and_stale_release(runtime_etcd):
    client = runtime_etcd
    route, owner, worker = await setup(client)
    fence, compare, writes = await plan(client, route, owner, worker)
    await execute_runtime_transaction(client, compare, writes)
    attempt = await attempt_snapshot(client, fence, owner)
    compare, writes = settle_transaction(
        client,
        attempt=attempt,
        outcome=ValidatedOutcome("failed", None, b'{"complete":false}'),
        task_json={"status": "failed"},
    )
    assert (
        await execute_runtime_transaction(client, compare, writes)
    ).status is CasStatus.APPLIED
    assert (
        await execute_runtime_transaction(client, compare, writes)
    ).status is CasStatus.CONFLICT
    assert await client.get(attempt.slot.key) is None
    assert b'"complete":false' in (await client.get(attempt.record.key)).value
