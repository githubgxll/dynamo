# SPDX-License-Identifier: Apache-2.0
"""Compiler policy bindings for the shared finite-task execution manager."""
import fcntl

from dingo.worker.contracts import RecoveryPolicy, WorkerProfile
from dingo.worker.task_manager import DetachedTaskManager

CONTEXT_IR_PROFILE = WorkerProfile(
    adapter="context_ir_compile",
    task_kind="context_ir",
    model_adapters=(("minimax_h3", 1),),
    protocol_adapters=(("open_h3_ir", 1), ("intent_resolution", 1)),
    preparation_error_code="context_ir_preparation_failed",
    recovery_error_code="context_ir_execution_unknown",
)


class ContextIRRecovery(RecoveryPolicy):
    """One compiler owns a dedicated Qwen; a CPU restart does not stop Qwen."""

    def __init__(self):
        self.lock = None

    async def acquire(self, manager):
        path = (
            manager.root
            / manager.deployment_id
            / "worker-locks"
            / (manager.domain + ".lock")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.open("a+b")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            raise RuntimeError("dedicated Qwen already has a compiler Worker") from None
        self.lock = lock

    def outcome(self, manager, root):
        usage = (
            manager.handler.recover_usage(root)
            if hasattr(manager.handler, "recover_usage")
            else None
        )
        return {**super().outcome(manager, root), "usage": usage}

    async def close(self):
        if self.lock is not None:
            self.lock.close()
            self.lock = None


def create_task_manager(handler, artifact_root, *, capacity=16, **options):
    """Bind compiler-specific declarations without putting defaults in the core."""
    return DetachedTaskManager(
        handler,
        artifact_root,
        profile=CONTEXT_IR_PROFILE,
        recovery=ContextIRRecovery(),
        capacity=capacity,
        **options,
    )
