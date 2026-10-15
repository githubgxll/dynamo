# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regional Gateway log collector. No Worker, etcd, artifact or inference access."""

import json
import math
import os
import re
import signal
import ssl
import threading
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULTS = {
    "namespaces": ["token-factory"],
    "poll_interval_s": 15,
    "initial_lookback_s": 86400,
    "max_read_bytes": 4194304,
    "max_line_bytes": 262144,
    "storage_budget_bytes": 10737418240,
    "storage_reserve_bytes": 536870912,
    "warning_fraction": 0.8,
    "pause_fraction": 0.9,
}
KEYS = ("cluster_id", "namespace", "deployment", "task_id", "attempt")
SOURCE_KEYS = ("cluster_id", "namespace", "pod_uid", "container", "restart")
ANNOTATION = "dingo.datacanvas.com/performance-container"
MODULE = re.compile(r"(?<![\w.])dingo\.video_gateway(?![\w.])")
STAMP = re.compile(
    r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.(\d{1,9}))?(?:Z|[+-]\d\d:\d\d)$"
)


def configuration(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("unknown configuration keys or invalid object")
    result = {**DEFAULTS, **value}
    namespaces = result["namespaces"]
    if not isinstance(namespaces, list) or len(namespaces) > 50:
        raise ValueError("namespaces must be a list of at most 50 names")
    for name in namespaces:
        if not isinstance(name, str) or not re.fullmatch(
            r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", name
        ):
            raise ValueError("invalid namespace")
    result["namespaces"] = sorted(set(namespaces))
    bounds = {
        "poll_interval_s": (5, 3600),
        "initial_lookback_s": (60, 604800),
        "max_read_bytes": (1024, 67108864),
        "max_line_bytes": (1024, 1048576),
        "storage_budget_bytes": (1048576, 1099511627776),
        "storage_reserve_bytes": (0, 1099511627776),
    }
    for key, (lower, upper) in bounds.items():
        if type(result[key]) is not int or not lower <= result[key] <= upper:
            raise ValueError(f"invalid {key}")
    for key in ("warning_fraction", "pause_fraction"):
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]):
            raise ValueError(f"invalid {key}")
    if not 0 < result["warning_fraction"] < result["pause_fraction"] < 1:
        raise ValueError("thresholds must satisfy 0 < warning < pause < 1")
    if result["storage_reserve_bytes"] >= result["storage_budget_bytes"]:
        raise ValueError("reserve must be smaller than budget")
    if result["max_line_bytes"] > result["max_read_bytes"]:
        raise ValueError("line budget exceeds read budget")
    return result


def load_configuration(path, previous):
    try:
        return configuration(json.loads(Path(path).read_text())), None
    except (OSError, ValueError, TypeError):
        # No raw configuration in logs, and no implicit fallback to token-factory.
        return previous, "invalid_config_using_last_good_or_paused"


def discover(pod, cluster):
    metadata = pod["metadata"]
    statuses = {
        s["name"]: s for s in pod.get("status", {}).get("containerStatuses", [])
    }
    explicit = metadata.get("annotations", {}).get(ANNOTATION)
    for container in pod.get("spec", {}).get("containers", []):
        command = " ".join(container.get("command", []) + container.get("args", []))
        if not (
            container["name"] == explicit or (not explicit and MODULE.search(command))
        ):
            continue
        state = statuses.get(container["name"], {})
        if not state.get("state", {}).get("running") and not state.get("state", {}).get(
            "terminated"
        ):
            continue
        yield {
            "cluster_id": cluster,
            "namespace": metadata["namespace"],
            "pod_uid": metadata["uid"],
            "pod": metadata["name"],
            "container": container["name"],
            "restart": state.get("restartCount", 0),
            "image": state.get("imageID") or container.get("image", "unknown"),
        }


def timestamp_key(value):
    if not isinstance(value, str) or not STAMP.fullmatch(value):
        raise ValueError("invalid Kubernetes timestamp")
    parsed = datetime.fromisoformat(value).astimezone(UTC)
    fraction = STAMP.fullmatch(value).group(1) or ""
    return parsed.strftime("%Y-%m-%dT%H:%M:%S") + "." + fraction.ljust(9, "0")


def number(value, integer=False):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid numeric field")
    if integer and (value != int(value) or value > 2147483647):
        raise ValueError("invalid integer field")
    if value > 9e15:
        raise ValueError("numeric field too large")
    return int(value) if integer else value


def short_text(value):
    if not isinstance(value, str) or not value or len(value) > 512 or "\x00" in value:
        raise ValueError("invalid text field")
    return value


def worker_identity(value):
    if value is None or value == "":
        return None
    # Dynamo serializes its u64 instance IDs as JSON integers. Preserve all
    # digits without converting through float or relaxing other text fields.
    if type(value) is int and 0 <= value <= 18446744073709551615:
        return str(value)
    return short_text(value)


def event_row(event, source):
    if not isinstance(event, dict) or event.get("log_type") != "video_task_lifecycle":
        return None
    if event.get("event") not in {"completed", "failed", "cancelled"} or not event.get(
        "performance"
    ):
        return None
    p = event["performance"]
    if p.get("schema_version", 1) != 1:
        raise ValueError("unsupported performance schema")
    g, counts, timings, execution = (
        p.get(k) or {} for k in ("generation", "counts", "timings", "model_execution")
    )
    row = {k: source[k] for k in ("cluster_id", "namespace")}
    for column, field in (
        ("deployment", "deployment_id"),
        ("task_id", "task_id"),
        ("model", "model"),
        ("pool", "pool_id"),
    ):
        row[column] = short_text(event[field])
    row["status"] = event["event"]
    row["attempt"] = number(event.get("attempt", 1), True)
    if row["attempt"] is None:
        raise ValueError("invalid attempt")
    row["completed_at"] = datetime.fromtimestamp(
        number(event["timestamp_ms"]) / 1000, UTC
    )
    row.update(
        configuration_revision=short_text(p.get("configuration_revision") or "unknown"),
        environment_revision="unknown",
        schema_id=short_text(execution.get("schema") or "unavailable"),
        source="gateway_log",
        input_type=short_text(p.get("input_type") or "unknown"),
    )
    for column, field in (
        ("seconds", "seconds"),
        ("width", "width"),
        ("height", "height"),
        ("fps", "fps"),
        ("steps", "num_inference_steps"),
    ):
        row[column] = number(g.get(field), column in {"width", "height", "steps"})
    row["sound"] = g.get("generate_sound")
    if row["sound"] is not None and type(row["sound"]) is not bool:
        raise ValueError("invalid generate_sound")
    for kind in ("image", "video", "audio"):
        row[kind + "_count"] = number(counts.get(kind), True)
    for column in (
        "image_pixels",
        "image_bytes",
        "video_seconds",
        "video_bytes",
        "video_frames",
        "audio_seconds",
        "audio_bytes",
    ):
        kind, quantity = column.split("_")
        value = number(p.get(f"{kind}_total_{quantity}"))
        if value is not None and quantity != "seconds" and value != int(value):
            raise ValueError("invalid media integer")
        row[column] = value if quantity == "seconds" or value is None else int(value)
    for column, field in (
        ("e2e_s", "end_to_end_s"),
        ("queue_s", "queue_s"),
        ("worker_queue_s", "worker_queue_s"),
        ("execution_s", "execution_s"),
        ("finalize_s", "finalize_s"),
    ):
        row[column] = number(timings.get(field))
    stages = execution.get("stages") or {}
    if not isinstance(stages, dict) or len(stages) > 128:
        raise ValueError("invalid stages")
    row["stages"] = {
        short_text(k): number(v) for k, v in stages.items() if v is not None
    }
    # Strict allowlist: never store prompts, URLs, entire environments or raw log lines.
    row["details"] = {
        k: source[k] for k in ("pod", "pod_uid", "container", "restart", "image")
    }
    row["details"]["worker_id"] = worker_identity(event.get("worker_instance_id"))
    references = p.get("references") or []
    if not isinstance(references, list) or len(references) > 32:
        raise ValueError("invalid references")
    safe_references = []
    for reference in references:
        kind = reference.get("kind")
        if kind not in {"image", "audio", "video", "unknown"}:
            raise ValueError("invalid reference kind")
        safe_references.append(
            {
                "kind": kind,
                **{
                    field: number(reference.get(field))
                    for field in (
                        "ordinal",
                        "width",
                        "height",
                        "fps",
                        "duration_s",
                        "bytes",
                        "frame_count",
                        "used_frame_count",
                    )
                },
            }
        )
    row["details"]["performance"] = {"schema_version": 1, "references": safe_references}
    return row


def read_batch(stream, source, cursor_time, cursor_count, config):
    rows, invalid, read_bytes = [], 0, 0
    latest, count = cursor_time, cursor_count
    old_key = timestamp_key(cursor_time) if cursor_time else None
    same_seen = 0
    truncated = False
    while read_bytes < config["max_read_bytes"]:
        raw = stream.readline(config["max_line_bytes"] + 1)
        if not raw:
            break
        read_bytes += len(raw)
        if not raw.endswith(b"\n"):
            # An oversized line is not checkpointed: explicitly report blockage.
            raise ValueError("oversized_or_partial_log_line")
        stamp, separator, message = raw.decode("utf-8", errors="replace").partition(" ")
        if not separator:
            raise ValueError("missing_log_timestamp")
        key = timestamp_key(stamp)
        if old_key and key <= old_key:
            if key == old_key:
                same_seen += 1
            if key < old_key or same_seen <= cursor_count:
                continue
        if latest and key < timestamp_key(latest):
            raise ValueError("non_monotonic_log_timestamp")
        count = count + 1 if latest and key == timestamp_key(latest) else 1
        latest = stamp
        start = message.find('{"')
        if start < 0:
            continue
        try:
            value = event_row(json.loads(message[start:]), source)
            if value:
                rows.append(value)
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            invalid += 1
    else:
        truncated = True
    if truncated and (latest, count) == (cursor_time, cursor_count):
        raise ValueError("log_budget_exhausted_without_progress")
    return rows, latest, count, invalid, truncated, read_bytes


class Kubernetes:
    def __init__(self):
        self.account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
        context = ssl.create_default_context(cafile=str(self.account / "ca.crt"))
        self.http = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=context),
        )

    def open(self, path, params):
        request = urllib.request.Request(
            "https://kubernetes.default.svc"
            + path
            + "?"
            + urllib.parse.urlencode(params),
            headers={
                "Authorization": "Bearer "
                + (self.account / "token").read_text().strip()
            },
        )
        return self.http.open(request, timeout=10)

    def pods(self, namespace):
        continuation = ""
        while True:
            with self.open(
                f"/api/v1/namespaces/{namespace}/pods",
                {"limit": 200, "continue": continuation},
            ) as response:
                page = json.load(response)
            yield from page["items"]
            continuation = page.get("metadata", {}).get("continue", "")
            if not continuation:
                break


def insert_row(db, row):
    from psycopg.types.json import Jsonb

    columns = list(row)
    assignments = ",".join(f"{c}=EXCLUDED.{c}" for c in columns if c not in KEYS)
    query = (
        f"INSERT INTO video_performance ({','.join(columns)}) VALUES ({','.join(['%s'] * len(columns))}) "
        f"ON CONFLICT ({','.join(KEYS)}) DO UPDATE SET {assignments} "
        "WHERE EXCLUDED.completed_at > video_performance.completed_at"
    )
    return db.execute(
        query,
        [Jsonb(row[c]) if c in {"stages", "details"} else row[c] for c in columns],
    ).rowcount


def source_key(source):
    return tuple(source[k] for k in SOURCE_KEYS)


SOURCE_WHERE = " AND ".join(f"{k}=%s" for k in SOURCE_KEYS)


def collect_source(db, kube, source, config, previous=False):
    key = source_key(source)
    db.execute(
        "INSERT INTO collector_sources(cluster_id,namespace,pod_uid,container,restart,pod) "
        "VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        (*key, source["pod"]),
    )
    cursor_time, cursor_count = db.execute(
        "SELECT cursor_time,cursor_count FROM collector_sources WHERE " + SOURCE_WHERE,
        key,
    ).fetchone()
    since = cursor_time or (
        datetime.now(UTC) - timedelta(seconds=config["initial_lookback_s"])
    ).isoformat().replace("+00:00", "Z")
    params = {
        "container": source["container"],
        "timestamps": "true",
        "sinceTime": since,
        "previous": str(previous).lower(),
    }
    path = f"/api/v1/namespaces/{source['namespace']}/pods/{source['pod']}/log"
    with kube.open(path, params) as stream:
        rows, stamp, count, invalid, truncated, read_bytes = read_batch(
            stream, source, cursor_time, cursor_count, config
        )
    # Do not commit a cursor under the wrong generation if a container restarted
    # between discovery and reading logs. The next poll can read previous logs.
    with kube.open(
        f"/api/v1/namespaces/{source['namespace']}/pods/{source['pod']}", {}
    ) as response:
        pod = json.load(response)
    statuses = {
        s["name"]: s for s in pod.get("status", {}).get("containerStatuses", [])
    }
    actual_restart = statuses.get(source["container"], {}).get("restartCount")
    if pod["metadata"]["uid"] != source["pod_uid"] or actual_restart != source[
        "restart"
    ] + int(previous):
        raise ValueError("source_generation_changed_retry_next_poll")
    with db.transaction():
        inserted = sum(insert_row(db, row) for row in rows)
        db.execute(
            "UPDATE collector_sources SET cursor_time=%s,cursor_count=%s,last_poll=now(),last_error=NULL,"
            "inserted_events=inserted_events+%s,invalid_events=invalid_events+%s,truncated=%s WHERE "
            + SOURCE_WHERE,
            (stamp, count, inserted, invalid, truncated, *key),
        )
    return inserted, invalid, read_bytes


def capacity(db, config):
    database, relation, wal, rows = db.execute(
        "SELECT pg_database_size(current_database()),pg_total_relation_size('video_performance'),"
        "(SELECT COALESCE(sum(size),0)::bigint FROM pg_ls_waldir()),"
        "(SELECT count(*) FROM video_performance)"
    ).fetchone()
    # Mounted read-only in the collector: only statvfs, never inspect table files.
    local_database = os.environ.get("PGHOST", "127.0.0.1") in {
        "127.0.0.1",
        "localhost",
        "::1",
    }
    data_path = os.environ.get("DATA_PATH") or (
        "/var/lib/postgresql/data" if local_database else None
    )
    # A standalone puller does not mount PostgreSQL storage. Do not report its
    # own root filesystem as database free space.
    filesystem = (
        os.statvfs(data_path) if data_path and Path(data_path).exists() else None
    )
    free = filesystem.f_bavail * filesystem.f_frsize if filesystem else None
    usage = database + wal + config["storage_reserve_bytes"]
    fraction = usage / config["storage_budget_bytes"]
    pause = fraction >= config["pause_fraction"] or (
        free is not None and free < config["storage_reserve_bytes"]
    )
    return {
        "database_bytes": database,
        "table_and_indexes_bytes": relation,
        "wal_bytes": wal,
        "rows": rows,
        "node_available_bytes": free,
        "estimated_budget_usage_bytes": usage,
        "budget_bytes": config["storage_budget_bytes"],
        "paused": pause,
        "warning": pause or fraction >= config["warning_fraction"],
    }


def connect():
    import psycopg
    from psycopg import sql

    db = psycopg.connect(
        host=os.environ.get("PGHOST", "127.0.0.1"),
        port=os.environ.get("PGPORT", "5432"),
        dbname="performance",
        user="postgres",
        password=os.environ["DB_PASSWORD"],
        autocommit=True,
        connect_timeout=10,
        options="-c statement_timeout=10000",
    )
    if not db.execute("SELECT pg_try_advisory_lock(749132087)").fetchone()[0]:
        db.close()
        raise RuntimeError("another_collector_owns_database")
    # Refuse legacy databases rather than silently run an incompatible migration.
    exists = db.execute(
        "SELECT to_regclass('public.video_performance'),to_regclass('public.collector_schema')"
    ).fetchone()
    if exists[0] and not exists[1]:
        db.close()
        raise RuntimeError("legacy_database_requires_explicit_migration")
    if exists[1] and db.execute(
        "SELECT array_agg(version) FROM collector_schema"
    ).fetchone()[0] != [2]:
        db.close()
        raise RuntimeError("unsupported_database_schema")
    db.execute(Path(__file__).with_name("schema.sql").read_text())
    if not db.execute(
        "SELECT 1 FROM pg_roles WHERE rolname='grafanareader'"
    ).fetchone():
        db.execute(
            sql.SQL("CREATE ROLE grafanareader LOGIN PASSWORD {}").format(
                sql.Literal(os.environ["READ_PASSWORD"])
            )
        )
    # Older PostgreSQL versions grant schema CREATE and temporary-table rights
    # through PUBLIC. A SELECT-only Grafana role must not inherit those writes.
    db.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC, grafanareader")
    db.execute(
        "REVOKE CREATE, TEMPORARY ON DATABASE performance FROM PUBLIC, grafanareader"
    )
    db.execute("GRANT USAGE ON SCHEMA public TO grafanareader")
    db.execute(
        "GRANT SELECT ON video_performance,video_performance_stages,performance_environments,performance_collector_health,collector_sources,performance_http_sources,performance_http_rejections TO grafanareader"
    )
    db.execute("ALTER ROLE grafanareader SET statement_timeout='10s'")
    return db


def run():
    from psycopg.types.json import Jsonb

    cluster = short_text(os.environ["CLUSTER_ID"])
    kube, db = Kubernetes(), None
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    config = configuration({"namespaces": []})  # fail closed until valid mounted config
    while not stopping.is_set():
        config, config_error = load_configuration(
            os.environ.get("CONFIG_PATH", "/opt/config/config.json"), config
        )
        errors = [config_error] if config_error else []
        inserted, invalid, read_bytes, discovered = 0, 0, 0, []
        try:
            if db is None or db.closed:
                db = connect()
            storage = capacity(db, config)
            if not storage["paused"]:
                for namespace in config["namespaces"]:
                    try:
                        sources = [
                            source
                            for pod in kube.pods(namespace)
                            for source in discover(pod, cluster)
                        ]
                    except Exception as exc:  # noqa: BLE001
                        # Isolate namespace failures without logging raw payloads.
                        errors.append(f"{namespace}:discovery:{type(exc).__name__}")
                        continue
                    for current in sources:
                        discovered.append(
                            {
                                k: current[k]
                                for k in ("namespace", "pod", "container", "restart")
                            }
                        )
                        generations = (
                            [({**current, "restart": current["restart"] - 1}, True)]
                            if current["restart"]
                            else []
                        )
                        for source, previous in generations + [(current, False)]:
                            try:
                                added, bad, size = collect_source(
                                    db, kube, source, config, previous
                                )
                                inserted += added
                                invalid += bad
                                read_bytes += size
                            except Exception as exc:  # noqa: BLE001
                                # Isolate source failures without logging raw payloads.
                                # Error class only: SQL errors can echo bound user values.
                                error = f"{source['namespace']}/{source['pod']}/{source['container']}/{source['restart']}:{type(exc).__name__}"
                                errors.append(error)
                                db.execute(
                                    "UPDATE collector_sources SET last_poll=now(),last_error=%s WHERE "
                                    + SOURCE_WHERE,
                                    (error, *source_key(source)),
                                )
            health = {
                "namespaces": config["namespaces"],
                "poll_interval_s": config["poll_interval_s"],
                "sources": discovered,
                "storage": storage,
                "read_bytes": read_bytes,
                "invalid_events": invalid,
                "config_error": config_error,
            }
            db.execute(
                "UPDATE performance_collector_health SET last_poll=now(),last_event=(SELECT max(completed_at) FROM video_performance),"
                "last_error=%s,inserted_events=inserted_events+%s,details=%s WHERE id=1",
                (";".join(errors)[:2000] or None, inserted, Jsonb(health)),
            )
            print(
                json.dumps(
                    {
                        "collector": "v2",
                        "inserted": inserted,
                        "errors": errors[:10],
                        **health,
                    }
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 -- reconnect without logging SQL values
            print(json.dumps({"collector_error": type(exc).__name__}), flush=True)
            if db is not None:
                db.close()
            db = None
        stopping.wait(config["poll_interval_s"])
    if db is not None:
        db.close()


if __name__ == "__main__":
    run()
