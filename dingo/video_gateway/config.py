# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, namespace-agnostic configuration for the video gateway."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_KNOWN_ADAPTERS = {"minimax_h3"}
_MIB = 1024 * 1024
_MAX_BODY_HARD_CAP = 512 * _MIB
_MEDIA_HARD_CAPS = {
    "max_total_file_bytes": 256 * _MIB,
    "max_single_file_bytes": 50 * _MIB,
    "max_text_field_bytes": 1024 * 1024,
    "max_parts": 32,
    "max_encoded_reference_bytes": 384 * _MIB,
    "max_result_bytes": 128 * _MIB,
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _only(data: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown keys: {', '.join(unknown)}")


def _required_string(data: Mapping[str, Any], key: str, name: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}.{key} must be a non-empty string")
    return value.strip()


def _safe_identifier(value: str, name: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{name} must match {_SAFE_ID.pattern!r}; received {value!r}")
    return value


def _boolean(data: Mapping[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def canonical_backend_target(value: str) -> str:
    raw = value.strip()
    raw = raw.removeprefix("dyn://")
    parts = raw.split(".")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            "backend_target must be dyn://namespace.component.endpoint or "
            "namespace.component.endpoint"
        )
    for index, part in enumerate(parts):
        _safe_identifier(part, f"backend_target component {index}")
    return "dyn://" + ".".join(parts)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    discovery_backend: str = "etcd"
    request_plane: str = "tcp"
    event_plane: str | None = "zmq"


@dataclass(frozen=True, slots=True)
class HttpConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    upstream_url: str | None = None
    max_body_bytes: int = 256 * 1024 * 1024
    sync_timeout_s: float = 1800.0
    default_model: str | None = None
    async_submit_status_code: int = 202


@dataclass(frozen=True, slots=True)
class MediaConfig:
    max_total_file_bytes: int = 256 * _MIB
    max_single_file_bytes: int = 50 * _MIB
    max_text_field_bytes: int = 64 * 1024
    max_parts: int = 32
    max_encoded_reference_bytes: int = 384 * _MIB
    max_result_bytes: int = 128 * _MIB
    task_memory_overhead_bytes: int = 64 * _MIB
    inflight_memory_budget_bytes: int = 2 * 1024 * _MIB

    @property
    def max_result_encoded_bytes(self) -> int:
        return ((self.max_result_bytes + 2) // 3) * 4

    @property
    def max_task_memory_bytes(self) -> int:
        # Mixed-reference construction briefly holds both the individual data
        # URLs and their final JSON envelope.
        return (
            2 * self.max_encoded_reference_bytes
            + self.max_result_encoded_bytes
            + self.task_memory_overhead_bytes
        )


@dataclass(frozen=True, slots=True)
class TaskStoreConfig:
    kind: str = "etcd_http"
    url: str | None = None
    prefix: str = "/dingo/video-gateway/v1"
    request_timeout_s: float = 5.0


@dataclass(frozen=True, slots=True)
class ArtifactStoreConfig:
    kind: str = "filesystem"
    root: Path = Path("/tmp/dingo-video-gateway")


@dataclass(frozen=True, slots=True)
class SchedulingConfig:
    worker_capacity: int = 1
    queue_limit: int = 32
    accept_without_workers: bool = False
    execution_timeout_s: float = 1800.0
    abort_grace_s: float = 15.0
    discovery_interval_s: float = 1.0
    dispatch_interval_s: float = 0.25


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    name: str
    workflow: str
    compatibility_version: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PoolConfig:
    pool_id: str
    served_models: tuple[str, ...]
    backend_model: str
    backend_target: str
    adapter: AdapterConfig
    scheduling: SchedulingConfig
    configuration_revision: str

    @property
    def endpoint_path(self) -> str:
        return self.backend_target.removeprefix("dyn://")


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    schema_version: int
    deployment_id: str
    runtime: RuntimeConfig
    http: HttpConfig
    media: MediaConfig
    task_store: TaskStoreConfig
    artifact_store: ArtifactStoreConfig
    pools: tuple[PoolConfig, ...]
    pools_by_model: Mapping[str, PoolConfig] = field(repr=False)
    pools_by_id: Mapping[str, PoolConfig] = field(repr=False)

    def pool_for_model(self, model: str) -> PoolConfig | None:
        return self.pools_by_model.get(model)


def _runtime_config(raw: Any) -> RuntimeConfig:
    data = _mapping(raw or {}, "runtime")
    _only(data, {"discovery_backend", "request_plane", "event_plane"}, "runtime")
    event_plane = data.get("event_plane", "zmq")
    if event_plane is not None and not isinstance(event_plane, str):
        raise ValueError("runtime.event_plane must be a string or null")
    config = RuntimeConfig(
        discovery_backend=str(data.get("discovery_backend", "etcd")),
        request_plane=str(data.get("request_plane", "tcp")),
        event_plane=event_plane,
    )
    if config.discovery_backend not in {"etcd", "kubernetes", "file", "mem"}:
        raise ValueError("runtime.discovery_backend is not supported")
    if config.request_plane not in {"tcp", "nats"}:
        raise ValueError("runtime.request_plane is not supported")
    if config.event_plane not in {None, "zmq", "nats"}:
        raise ValueError("runtime.event_plane is not supported")
    return config


def _http_config(raw: Any) -> HttpConfig:
    data = _mapping(raw or {}, "http")
    _only(
        data,
        {
            "host",
            "port",
            "upstream_url",
            "max_body_bytes",
            "sync_timeout_s",
            "default_model",
            "async_submit_status_code",
        },
        "http",
    )
    port = int(data.get("port", 8000))
    max_body_bytes = int(data.get("max_body_bytes", 256 * 1024 * 1024))
    sync_timeout_s = float(data.get("sync_timeout_s", 1800.0))
    async_submit_status_code = int(data.get("async_submit_status_code", 202))
    if not 1 <= port <= 65535:
        raise ValueError("http.port must be between 1 and 65535")
    if max_body_bytes <= 0:
        raise ValueError("http.max_body_bytes must be positive")
    if max_body_bytes > _MAX_BODY_HARD_CAP:
        raise ValueError(
            f"http.max_body_bytes exceeds the hard cap of {_MAX_BODY_HARD_CAP}"
        )
    if sync_timeout_s <= 0:
        raise ValueError("http.sync_timeout_s must be positive")
    if async_submit_status_code not in {200, 202}:
        raise ValueError("http.async_submit_status_code must be 200 or 202")
    default_model = data.get("default_model")
    if default_model is not None:
        if not isinstance(default_model, str) or not default_model.strip():
            raise ValueError("http.default_model must be a non-empty string or null")
        default_model = default_model.strip()
    upstream = data.get("upstream_url")
    if upstream is not None:
        if not isinstance(upstream, str) or not upstream.startswith(
            ("http://", "https://")
        ):
            raise ValueError("http.upstream_url must be an HTTP(S) URL or null")
        upstream = upstream.rstrip("/")
    host = str(data.get("host", "0.0.0.0")).strip()
    if not host:
        raise ValueError("http.host must be non-empty")
    return HttpConfig(
        host=host,
        port=port,
        upstream_url=upstream,
        max_body_bytes=max_body_bytes,
        sync_timeout_s=sync_timeout_s,
        default_model=default_model,
        async_submit_status_code=async_submit_status_code,
    )


def _media_config(raw: Any, http: HttpConfig) -> MediaConfig:
    data = _mapping(raw or {}, "media")
    allowed = set(_MEDIA_HARD_CAPS) | {
        "task_memory_overhead_bytes",
        "inflight_memory_budget_bytes",
    }
    _only(data, allowed, "media")
    defaults = MediaConfig()
    values: dict[str, int] = {}
    for name in allowed:
        value = data.get(name, getattr(defaults, name))
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"media.{name} must be an integer")
        values[name] = value
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"media.{name} must be positive")
    for name, hard_cap in _MEDIA_HARD_CAPS.items():
        if values[name] > hard_cap:
            raise ValueError(
                f"media.{name} exceeds the protocol hard cap of {hard_cap}"
            )
    config = MediaConfig(**values)
    if config.max_single_file_bytes > config.max_total_file_bytes:
        raise ValueError(
            "media.max_single_file_bytes must not exceed max_total_file_bytes"
        )
    if config.max_total_file_bytes > http.max_body_bytes:
        raise ValueError(
            "media.max_total_file_bytes must not exceed http.max_body_bytes"
        )
    if config.inflight_memory_budget_bytes < config.max_task_memory_bytes:
        raise ValueError(
            "media.inflight_memory_budget_bytes must fit one maximum-size task"
        )
    return config


def _task_store_config(raw: Any) -> TaskStoreConfig:
    data = _mapping(raw or {}, "task_store")
    _only(data, {"kind", "url", "prefix", "request_timeout_s"}, "task_store")
    kind = str(data.get("kind", "etcd_http"))
    if kind not in {"etcd_http", "memory"}:
        raise ValueError("task_store.kind must be 'etcd_http' or 'memory'")
    url = data.get("url")
    if kind == "etcd_http" and (not isinstance(url, str) or not url):
        raise ValueError("task_store.url is required for etcd_http")
    prefix = str(data.get("prefix", "/dingo/video-gateway/v1"))
    if prefix == "/" or not prefix.startswith("/") or ".." in prefix:
        raise ValueError("task_store.prefix must be an absolute safe key prefix")
    timeout = float(data.get("request_timeout_s", 5.0))
    if timeout <= 0:
        raise ValueError("task_store.request_timeout_s must be positive")
    return TaskStoreConfig(
        kind=kind,
        url=url.rstrip("/") if isinstance(url, str) else None,
        prefix=prefix.rstrip("/"),
        request_timeout_s=timeout,
    )


def _artifact_store_config(raw: Any) -> ArtifactStoreConfig:
    data = _mapping(raw or {}, "artifact_store")
    _only(data, {"kind", "root"}, "artifact_store")
    kind = str(data.get("kind", "filesystem"))
    if kind != "filesystem":
        raise ValueError("artifact_store.kind must be 'filesystem' in version 1")
    root = Path(_required_string(data, "root", "artifact_store")).expanduser()
    if not root.is_absolute():
        raise ValueError("artifact_store.root must be an absolute path")
    return ArtifactStoreConfig(kind=kind, root=root)


def _scheduling_config(raw: Any, pool_name: str) -> SchedulingConfig:
    data = _mapping(raw or {}, f"pools[{pool_name}].scheduling")
    _only(
        data,
        {
            "worker_capacity",
            "queue_limit",
            "accept_without_workers",
            "execution_timeout_s",
            "abort_grace_s",
            "discovery_interval_s",
            "dispatch_interval_s",
        },
        f"pools[{pool_name}].scheduling",
    )
    config = SchedulingConfig(
        worker_capacity=int(data.get("worker_capacity", 1)),
        queue_limit=int(data.get("queue_limit", 32)),
        accept_without_workers=_boolean(data, "accept_without_workers", False),
        execution_timeout_s=float(data.get("execution_timeout_s", 1800.0)),
        abort_grace_s=float(data.get("abort_grace_s", 15.0)),
        discovery_interval_s=float(data.get("discovery_interval_s", 1.0)),
        dispatch_interval_s=float(data.get("dispatch_interval_s", 0.25)),
    )
    if config.worker_capacity != 1:
        raise ValueError("version 1 requires scheduling.worker_capacity=1")
    if config.queue_limit < 1:
        raise ValueError("scheduling.queue_limit must be positive")
    if (
        min(
            config.execution_timeout_s,
            config.abort_grace_s,
            config.discovery_interval_s,
            config.dispatch_interval_s,
        )
        <= 0
    ):
        raise ValueError("scheduling timeout and interval values must be positive")
    return config


def _pool_config(raw: Any, index: int) -> PoolConfig:
    name = f"pools[{index}]"
    data = _mapping(raw, name)
    _only(
        data,
        {
            "pool_id",
            "served_models",
            "backend_model",
            "backend_target",
            "adapter",
            "scheduling",
        },
        name,
    )
    pool_id = _safe_identifier(
        _required_string(data, "pool_id", name), f"{name}.pool_id"
    )
    served = data.get("served_models")
    if not isinstance(served, list) or not served:
        raise ValueError(f"{name}.served_models must be a non-empty list")
    if any(not isinstance(value, str) or not value.strip() for value in served):
        raise ValueError(f"{name}.served_models must contain only non-empty strings")
    served_models = tuple(value.strip() for value in served)
    if len(set(served_models)) != len(served_models):
        raise ValueError(f"{name}.served_models must contain unique non-empty strings")
    backend_model = _required_string(data, "backend_model", name)
    backend_target = canonical_backend_target(
        _required_string(data, "backend_target", name)
    )

    adapter_data = _mapping(data.get("adapter"), f"{name}.adapter")
    required_adapter = {"name", "workflow", "compatibility_version"}
    missing = sorted(required_adapter - set(adapter_data))
    if missing:
        raise ValueError(f"{name}.adapter missing keys: {', '.join(missing)}")
    adapter = AdapterConfig(
        name=_required_string(adapter_data, "name", f"{name}.adapter"),
        workflow=_required_string(adapter_data, "workflow", f"{name}.adapter"),
        compatibility_version=_required_string(
            adapter_data, "compatibility_version", f"{name}.adapter"
        ),
        options={
            key: value
            for key, value in adapter_data.items()
            if key not in required_adapter
        },
    )
    if adapter.name not in _KNOWN_ADAPTERS:
        raise ValueError(f"unknown video adapter: {adapter.name}")
    scheduling = _scheduling_config(data.get("scheduling"), pool_id)
    revision_payload = {
        "pool_id": pool_id,
        "served_models": served_models,
        "backend_model": backend_model,
        "backend_target": backend_target,
        "adapter": {
            "name": adapter.name,
            "workflow": adapter.workflow,
            "compatibility_version": adapter.compatibility_version,
            "options": adapter.options,
        },
        "scheduling": {
            "worker_capacity": scheduling.worker_capacity,
            "execution_timeout_s": scheduling.execution_timeout_s,
        },
    }
    revision = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                revision_payload, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        ).hexdigest()
    )
    return PoolConfig(
        pool_id=pool_id,
        served_models=served_models,
        backend_model=backend_model,
        backend_target=backend_target,
        adapter=adapter,
        scheduling=scheduling,
        configuration_revision=revision,
    )


def parse_config(raw: Any) -> GatewayConfig:
    data = _mapping(raw, "config")
    _only(
        data,
        {
            "schema_version",
            "deployment_id",
            "runtime",
            "http",
            "media",
            "task_store",
            "artifact_store",
            "pools",
        },
        "config",
    )
    schema_version = int(data.get("schema_version", 0))
    if schema_version != 1:
        raise ValueError("schema_version must be 1")
    deployment_id = _safe_identifier(
        _required_string(data, "deployment_id", "config"), "deployment_id"
    )
    raw_pools = data.get("pools")
    if not isinstance(raw_pools, list) or not raw_pools:
        raise ValueError("pools must be a non-empty list")
    pools = tuple(_pool_config(value, index) for index, value in enumerate(raw_pools))
    by_id: dict[str, PoolConfig] = {}
    by_model: dict[str, PoolConfig] = {}
    for pool in pools:
        if pool.pool_id in by_id:
            raise ValueError(f"duplicate pool_id: {pool.pool_id}")
        by_id[pool.pool_id] = pool
        for model in pool.served_models:
            if model in by_model:
                raise ValueError(f"served model {model!r} maps to multiple pools")
            by_model[model] = pool

    http = _http_config(data.get("http"))
    media = _media_config(data.get("media"), http)
    if http.default_model is not None and http.default_model not in by_model:
        raise ValueError(
            f"http.default_model {http.default_model!r} is not a configured served model"
        )

    return GatewayConfig(
        schema_version=schema_version,
        deployment_id=deployment_id,
        runtime=_runtime_config(data.get("runtime")),
        http=http,
        media=media,
        task_store=_task_store_config(data.get("task_store")),
        artifact_store=_artifact_store_config(data.get("artifact_store")),
        pools=pools,
        pools_by_model=by_model,
        pools_by_id=by_id,
    )


def load_config(path: str | Path) -> GatewayConfig:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - packaging guard
            raise RuntimeError(
                "YAML configuration requires the ai-dingo[video-gateway] extra"
            ) from exc
        raw = yaml.safe_load(text)
    return parse_config(raw)
