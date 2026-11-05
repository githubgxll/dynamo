"""Job-owned authorization; native pool windows use an explicit copy_context."""
from contextvars import ContextVar

_OPERATION = ContextVar("h3sr_owned_operation", default=None)


def bind_operation(action_id, token):
    return _OPERATION.set({"x-h3sr-action-id": action_id, "x-h3sr-operation-token": token})


def reset_operation(token):
    _OPERATION.reset(token)


def internal_window_headers(authorization):
    result = {"Authorization": authorization} if authorization else {}
    operation = _OPERATION.get()
    if operation is not None:
        result.update(operation)
    return result
