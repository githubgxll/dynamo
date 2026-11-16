# SPDX-License-Identifier: Apache-2.0
"""Deployment-independent descriptions of finite task workers."""
from dataclasses import dataclass

from dingo.common.task_protocol import TASK_KINDS, identifier


@dataclass(frozen=True)
class WorkerProfile:
    """Business declarations injected by an entry point, never inferred by the core."""

    adapter: str
    task_kind: str
    model_adapters: tuple[tuple[str, int], ...]
    protocol_adapters: tuple[tuple[str, int], ...]
    preparation_error_code: str = "worker_preparation_failed"
    recovery_error_code: str = "worker_execution_unknown"

    def __post_init__(self):
        for name in ("adapter", "preparation_error_code", "recovery_error_code"):
            identifier(getattr(self, name), name)
        if self.task_kind not in TASK_KINDS:
            raise ValueError("unsupported task kind")
        for entries in (self.model_adapters, self.protocol_adapters):
            if not entries or len(dict(entries)) != len(entries):
                raise ValueError("adapter declarations must be nonempty and unique")
            for name, version in entries:
                identifier(name, "adapter name")
                if type(version) is not int or version < 1:
                    raise ValueError("adapter version must be positive")


class RecoveryPolicy:
    """Hooks for backend ownership and recovery metadata, not task scheduling.

    The manager still requires an independent engine-idle proof before using
    recovered metadata. The default never installs a filesystem/domain lock.
    """

    async def acquire(self, manager):
        pass

    def outcome(self, manager, root):
        return {"failed": True, "error_code": manager.profile.recovery_error_code}

    async def close(self):
        pass
