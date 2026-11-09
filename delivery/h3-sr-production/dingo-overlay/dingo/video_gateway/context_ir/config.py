"""Strict compiler routing and capacity configuration, disabled by default."""
from dataclasses import asdict, dataclass, field
import hashlib
import math
import re
from urllib.parse import urlsplit
from dingo.video_gateway.task_runtime.transactions import encode


@dataclass(frozen=True)
class BackendConfig:
    url: str
    adapter: str = "open_h3_ir"
    epoch: str = "initial"
    max_inflight: int = 2
    queue_limit: int = 32
    queue_timeout_s: float = 600
    prepare_timeout_s: float = 120
    connect_timeout_s: float = 5
    compile_timeout_s: float = 600
    execution_timeout_s: float = 900
    storage_budget_bytes: int = 8 * 1024**3

    @property
    def revision(self):
        return hashlib.sha256(encode(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True)
class ContextIRConfig:
    enabled: bool = False
    backends: dict[str, BackendConfig] = field(default_factory=dict)
    model_bindings: dict[str, str] = field(default_factory=dict)
    upload_timeout_s: float = 300
    owner_lease_ttl_s: int = 30
    owner_heartbeat_s: float = 10
    max_pre_call_attempts: int = 2
    max_compile_attempts: int = 1
    input_terminal_retention_s: float = 3600
    result_retention_s: float = 86400
    tombstone_retention_s: float = 86400
    sweeper_interval_s: float = 30
    managed_storage_budget_bytes: int = 16 * 1024**3
    derived_bytes_per_task: int = 64 * 1024**2
    response_max_bytes: int = 2 * 1024**2
    result_max_bytes: int = 512 * 1024


def parse_context_ir(raw, pools):
    if raw is None:
        return ContextIRConfig()
    if not isinstance(raw, dict) or set(raw) - set(ContextIRConfig.__dataclass_fields__):
        raise ValueError("invalid context_ir configuration keys")
    data = dict(raw)
    if not isinstance(data.get("enabled", False), bool):
        raise ValueError("context_ir.enabled must be boolean")
    backends = {}
    urls = set()
    for name, values in data.get("backends", {}).items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
            raise ValueError("invalid ContextIR backend identifier")
        if not isinstance(values, dict) or set(values) - set(BackendConfig.__dataclass_fields__):
            raise ValueError("unknown ContextIR backend setting")
        b = BackendConfig(**values)
        url = urlsplit(b.url)
        if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError("ContextIR backend requires a credential-free HTTP URL")
        if b.adapter != "open_h3_ir":
            raise ValueError("unregistered ContextIR protocol adapter")
        if not isinstance(b.epoch, str) or not b.epoch or len(b.epoch) > 128:
            raise ValueError("invalid ContextIR backend epoch")
        canonical_url = b.url.rstrip("/").lower()
        if canonical_url in urls:
            raise ValueError("one compiler capacity domain cannot have duplicate backends")
        urls.add(canonical_url)
        for key in ("max_inflight", "queue_limit", "storage_budget_bytes"):
            if type(getattr(b, key)) is not int or getattr(b, key) <= 0:
                raise ValueError(f"context_ir backend {key} must be positive integer")
        for key in ("queue_timeout_s", "prepare_timeout_s", "connect_timeout_s", "compile_timeout_s", "execution_timeout_s"):
            v = getattr(b, key)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"invalid context_ir backend {key}")
        backends[name] = b
    data["backends"] = backends
    result = ContextIRConfig(**data)
    for key in set(ContextIRConfig.__dataclass_fields__) - {"enabled", "backends", "model_bindings"}:
        v = getattr(result, key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"invalid context_ir.{key}")
    if result.owner_heartbeat_s * 2 >= result.owner_lease_ttl_s:
        raise ValueError("ContextIR owner heartbeat must be less than half lease TTL")
    if result.max_compile_attempts != 1 or result.max_pre_call_attempts != 2:
        raise ValueError("ContextIR v1 permits one compile and at most two prepare attempts")
    for model, backend in result.model_bindings.items():
        if model not in pools or backend not in backends:
            raise ValueError("ContextIR binding references unknown model or backend")
        if pools[model].adapter.name != "minimax_h3":
            raise ValueError("unregistered ContextIR model semantic adapter")
    if result.enabled and (not backends or not result.model_bindings):
        raise ValueError("enabled ContextIR needs explicit backends and model bindings")
    return result
