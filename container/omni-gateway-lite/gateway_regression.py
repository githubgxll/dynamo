# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed Gateway capacity implementation with mocked filesystem inputs."""
from __future__ import annotations

import asyncio
from collections import namedtuple
import errno
from pathlib import Path
from unittest.mock import patch


def check_capacity(module):
    mount = Path("/mock/pvc")
    inner = mount / "deployment"
    root = inner / "artifacts"
    usage = namedtuple("Usage", "total used free")(2000000, 500000, 1500000)
    # Avoid creating any model/PVC directories. Only capacity() runs, using its
    # real installed run_file_io helper and mocked xattr/statfs observations.
    store = object.__new__(module.FileArtifactStore)
    store.root = root
    cases = []

    def check(name, attributes, expected, *, denied=None, expect_fallback=False):
        reads = []

        def getxattr(path, attribute):
            path = Path(path)
            reads.append((path, attribute))
            if denied is not None and path == denied:
                raise OSError(errno.EACCES, "mock inaccessible path")
            try:
                return attributes[path][attribute]
            except KeyError:
                raise OSError(errno.ENODATA, "mock missing xattr") from None

        with patch.object(module.os, "getxattr", getxattr, create=True), \
             patch.object(Path, "is_mount", lambda path: path == mount), \
             patch.object(module.shutil, "disk_usage", return_value=usage) as fallback:
            actual = asyncio.run(store.capacity())
        observed = (actual.total_bytes, actual.used_bytes, actual.free_bytes)
        if observed != expected or fallback.called != expect_fallback:
            raise AssertionError(f"Gateway capacity regression {name}: {observed}, fallback={fallback.called}")
        if any(path != mount and mount not in path.parents for path, _ in reads):
            raise AssertionError("Gateway quota lookup crossed the mount boundary")
        cases.append({"name": name, "status": "PASS", "observed": list(observed),
                      "fallback": fallback.called, "reads": [[str(path), attr] for path, attr in reads]})
        return reads

    for used in (250000, 300000):
        reads = check(f"ancestor-quota-refresh-{used}", {
            mount: {"ceph.quota.max_bytes": b"1000000", "ceph.dir.rbytes": str(used).encode()},
            root: {"ceph.dir.rbytes": b"1000"}}, (1000000, used, 1000000 - used))
        if (root, "ceph.dir.rbytes") in reads:
            raise AssertionError("Gateway combined parent quota with child's usage")
    check("no-quota-mount-fallback", {}, tuple(usage), expect_fallback=True)
    reads = check("inaccessible-child-fallback", {
        mount: {"ceph.quota.max_bytes": b"1000000", "ceph.dir.rbytes": b"0"}},
        tuple(usage), denied=root, expect_fallback=True)
    if any(path != root for path, _ in reads):
        raise AssertionError("Gateway skipped an inaccessible directory")
    reads = check("nearest-quota", {
        inner: {"ceph.quota.max_bytes": b"1000", "ceph.dir.rbytes": b"200"},
        mount: {"ceph.quota.max_bytes": b"1000000", "ceph.dir.rbytes": b"900000"}}, (1000, 200, 800))
    if any(path == mount for path, _ in reads):
        raise AssertionError("Gateway ignored a nearer quota")
    check("over-quota-clamp", {root: {"ceph.quota.max_bytes": b"100", "ceph.dir.rbytes": b"120"}}, (100, 120, 0))
    for used in (None, b"invalid", b"-1"):
        local = {"ceph.quota.max_bytes": b"1000"}
        if used is not None:
            local["ceph.dir.rbytes"] = used
        reads = check("unreadable-usage-" + str(used), {
            root: local, mount: {"ceph.quota.max_bytes": b"1000000", "ceph.dir.rbytes": b"0"}},
            tuple(usage), expect_fallback=True)
        if any(path != root for path, _ in reads):
            raise AssertionError("Gateway skipped a quota with unusable usage")
    return {"status": "PASS", "case_count": len(cases), "cases": cases,
            "scope": "Actual imported Gateway capacity method and async I/O helper; mocked xattr/statfs, no real CephFS or deployment acceptance"}
