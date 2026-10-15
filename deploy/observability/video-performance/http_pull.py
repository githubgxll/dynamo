# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP performance puller with explicit targets; no Kubernetes credentials."""

import concurrent.futures
import hashlib
import json
import math
import os
import re
import signal
import ssl
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import collector


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a performance credential to another origin.
        return None


def configuration(raw):
    if not isinstance(raw, dict) or set(raw) - {"targets", "storage", "max_workers"}:
        raise ValueError("invalid HTTP pull configuration")
    targets = raw.get("targets", [])
    if not isinstance(targets, list) or len(targets) > 50:
        raise ValueError("targets must be a list of at most 50 entries")
    result, names = [], set()
    for value in targets:
        if not isinstance(value, dict) or set(value) - {
            "name",
            "cluster_id",
            "namespace",
            "deployment_id",
            "url",
            "token_env",
            "poll_interval_s",
            "timeout_s",
            "allow_insecure_http",
        }:
            raise ValueError("invalid target fields")
        target = {**value}
        for key in (
            "name",
            "cluster_id",
            "namespace",
            "deployment_id",
            "url",
            "token_env",
        ):
            collector.short_text(target.get(key))
        if target["name"] in names:
            raise ValueError("duplicate target name")
        names.add(target["name"])
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target["token_env"]):
            raise ValueError("invalid credential environment reference")
        if target["token_env"] in {
            "DB_PASSWORD",
            "READ_PASSWORD",
            "PGHOST",
            "PGPORT",
            "CONFIG_PATH",
            "DATA_PATH",
            "CLUSTER_ID",
        }:
            raise ValueError(
                "target credential must not reuse database or runtime environment variables"
            )
        url = urllib.parse.urlsplit(target["url"])
        insecure = target.get("allow_insecure_http", False)
        if type(insecure) is not bool or url.scheme not in {"http", "https"}:
            raise ValueError("invalid URL scheme")
        if url.scheme == "http" and not insecure:
            raise ValueError(
                "HTTP requires explicit allow_insecure_http for isolated tests"
            )
        if (
            not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("URL must not contain credentials, query or fragment")
        for key, default, lower, upper in (
            ("poll_interval_s", 5, 5, 60),
            ("timeout_s", 3, 0.1, 10),
        ):
            number = target.get(key, default)
            if (
                type(number) not in (int, float)
                or not math.isfinite(number)
                or not lower <= number <= upper
            ):
                raise ValueError("invalid target timing")
            target[key] = number
        target["binding"] = hashlib.sha256(
            json.dumps(
                {
                    k: target[k]
                    for k in ("cluster_id", "namespace", "deployment_id", "url")
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        result.append(target)
    workers = raw.get("max_workers", 4)
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("invalid max_workers")
    return {
        "targets": result,
        "max_workers": workers,
        "storage": collector.configuration(
            {"namespaces": [], **raw.get("storage", {})}
        ),
    }


def fetch(target, cursor):
    token = os.environ.get(target["token_env"], "")
    if len(token) < 16 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("missing or invalid performance credential")
    query = {"limit": 500}
    if cursor:
        query["cursor"] = cursor
    url = (
        target["url"].rstrip("/")
        + "/internal/v1/performance/records?"
        + urllib.parse.urlencode(query)
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    request = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + token, "Accept-Encoding": "identity"}
    )
    deadline = time.monotonic() + target["timeout_s"]
    with opener.open(request, timeout=target["timeout_s"]) as response:
        chunks, size = [], 0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("performance response exceeded read budget")
            chunk = response.read1(min(65536, 2 * 1024 * 1024 + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > 2 * 1024 * 1024:
                raise ValueError("performance response exceeds byte budget")
        raw = b"".join(chunks)
        if (
            len(raw) > 2 * 1024 * 1024
            or response.headers.get_content_type() != "application/json"
        ):
            raise ValueError("invalid performance response size or content type")
    return json.loads(raw)


def validate_page(target, page):
    if (
        not isinstance(page, dict)
        or type(page.get("schema_version")) is not int
        or page["schema_version"] != 1
    ):
        raise ValueError("unsupported performance protocol")
    collector.short_text(page.get("instance_id"))
    cursor = page.get("next_cursor")
    if not isinstance(cursor, str) or not 1 <= len(cursor) <= 1024:
        raise ValueError("invalid next cursor")
    for key in ("has_more", "gap_detected", "instance_changed"):
        if type(page.get(key)) is not bool:
            raise ValueError("invalid page flags")
    if type(page.get("next_seq")) is not int or page["next_seq"] < 0:
        raise ValueError("invalid page progress")
    rows = page.get("records")
    if not isinstance(rows, list) or len(rows) > 500 or (page["has_more"] and not rows):
        raise ValueError("invalid page records")
    source = page.get("source")
    if (
        not isinstance(source, dict)
        or source.get("deployment_id") != target["deployment_id"]
    ):
        raise ValueError("unexpected deployment identity")
    if source.get("namespace") not in (None, target["namespace"]):
        raise ValueError("unexpected namespace identity")
    source = {
        "cluster_id": target["cluster_id"],
        "namespace": target["namespace"],
        "pod": collector.short_text(source.get("pod")),
        "pod_uid": collector.short_text(source.get("pod_uid")),
        "container": collector.short_text(source.get("container")),
        "image": collector.short_text(source.get("image")),
        "restart": 0,
    }
    last = 0
    for item in rows:
        if (
            not isinstance(item, dict)
            or type(item.get("seq")) is not int
            or not last < item["seq"] <= page["next_seq"]
        ):
            raise ValueError("unordered performance records")
        last = item["seq"]
    return source


def insert_row(db, row):
    from psycopg.types.json import Jsonb

    columns = list(row)
    values = [Jsonb(row[c]) if c in {"details", "stages"} else row[c] for c in columns]
    inserted = db.execute(
        f"INSERT INTO video_performance ({','.join(columns)}) VALUES ({','.join(['%s'] * len(columns))}) ON CONFLICT DO NOTHING",
        values,
    ).rowcount
    if inserted:
        return "inserted"
    # Provenance differs between log and HTTP paths; business measurements must agree.
    compare = [c for c in columns if c not in {*collector.KEYS, "details", "source"}]
    same = db.execute(
        "SELECT "
        + " AND ".join(f"{c} IS NOT DISTINCT FROM %s" for c in compare)
        + " FROM video_performance WHERE "
        + " AND ".join(f"{c}=%s" for c in collector.KEYS),
        [Jsonb(row[c]) if c == "stages" else row[c] for c in compare]
        + [row[c] for c in collector.KEYS],
    ).fetchone()
    return "duplicate" if same and same[0] else "conflict"


def apply_page(db, target, page):
    source = validate_page(target, page)
    inserted, invalid = 0, 0
    with db.transaction():
        previous = db.execute(
            "SELECT binding,instance_id,cursor FROM performance_http_sources WHERE target_id=%s FOR UPDATE",
            (target["name"],),
        ).fetchone()
        changed = bool(
            previous
            and previous[0] == target["binding"]
            and previous[1]
            and previous[1] != page["instance_id"]
        )
        for item in page["records"]:
            reason = None
            try:
                event = item.get("event")
                if (
                    not isinstance(event, dict)
                    or event.get("deployment_id") != target["deployment_id"]
                ):
                    raise ValueError("unexpected event identity")
                row = collector.event_row(event, source)
                if row is None:
                    raise ValueError("missing performance event")
                row["source"] = "gateway_http"
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                reason = "invalid_event"
            if reason is None:
                outcome = insert_row(db, row)
                inserted += outcome == "inserted"
                reason = "conflict" if outcome == "conflict" else None
            if reason:
                invalid += 1
                db.execute(
                    "INSERT INTO performance_http_rejections(target_id,instance_id,seq,reason) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (target["name"], page["instance_id"], item["seq"], reason),
                )
        db.execute(
            "INSERT INTO performance_http_sources(target_id,binding,instance_id,cursor,last_poll,last_success,inserted_events,invalid_events,gap_events,instance_changes) "
            "VALUES(%s,%s,%s,%s,now(),now(),%s,%s,%s,%s) ON CONFLICT(target_id) DO UPDATE SET "
            "binding=EXCLUDED.binding,instance_id=EXCLUDED.instance_id,cursor=EXCLUDED.cursor,last_poll=now(),last_success=now(),last_error=NULL,"
            "inserted_events=performance_http_sources.inserted_events+EXCLUDED.inserted_events,invalid_events=performance_http_sources.invalid_events+EXCLUDED.invalid_events,"
            "gap_events=performance_http_sources.gap_events+EXCLUDED.gap_events,instance_changes=performance_http_sources.instance_changes+EXCLUDED.instance_changes",
            (
                target["name"],
                target["binding"],
                page["instance_id"],
                page["next_cursor"],
                inserted,
                invalid,
                int(page["gap_detected"]),
                int(changed),
            ),
        )
    return inserted, invalid


def run():
    from psycopg.types.json import Jsonb

    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    config, db, pending, due = configuration({}), None, {}, {}
    storage, storage_at, health_at = None, 0.0, 0.0
    # Bounded network workers; all PostgreSQL transactions stay on the main thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        while not stopping.is_set():
            try:
                config_error = None
                try:
                    config = configuration(
                        json.loads(
                            Path(
                                os.environ.get(
                                    "CONFIG_PATH", "/opt/config/http-pull.json"
                                )
                            ).read_text()
                        )
                    )
                except (OSError, ValueError, TypeError):
                    config_error = "invalid_config_using_last_good_or_paused"
                if db is None or db.closed:
                    db = collector.connect()
                    storage_at = health_at = 0.0
                targets = {t["name"]: t for t in config["targets"]}
                due = {name: value for name, value in due.items() if name in targets}
                if storage is None or time.monotonic() - storage_at >= 5:
                    storage = collector.capacity(db, config["storage"])
                    storage_at = time.monotonic()
                for name, (future, sent_target, sent_cursor) in list(pending.items()):
                    if not future.done():
                        continue
                    del pending[name]
                    current = targets.get(name)
                    if (
                        not current
                        or current["binding"] != sent_target["binding"]
                        or storage["paused"]
                    ):
                        continue
                    try:
                        page = future.result()
                        if (
                            page.get("has_more")
                            and page.get("next_cursor") == sent_cursor
                        ):
                            raise ValueError("performance page made no progress")
                        apply_page(db, current, page)
                        due[name] = time.monotonic() + (
                            0 if page["has_more"] else current["poll_interval_s"]
                        )
                    except Exception as exc:  # noqa: BLE001
                        # Never log remote payloads, SQL values or credentials.
                        error = type(exc).__name__
                        db.execute(
                            "INSERT INTO performance_http_sources(target_id,binding,last_poll,last_error) VALUES(%s,%s,now(),%s) "
                            "ON CONFLICT(target_id) DO UPDATE SET last_poll=now(),last_error=EXCLUDED.last_error",
                            (name, current["binding"], error),
                        )
                        due[name] = time.monotonic() + current["poll_interval_s"]
                if not storage["paused"]:
                    for name, target in sorted(
                        targets.items(), key=lambda item: (due.get(item[0], 0), item[0])
                    ):
                        if (
                            name in pending
                            or due.get(name, 0) > time.monotonic()
                            or len(pending) >= config["max_workers"]
                        ):
                            continue
                        state = db.execute(
                            "SELECT binding,cursor FROM performance_http_sources WHERE target_id=%s",
                            (name,),
                        ).fetchone()
                        cursor = (
                            state[1]
                            if state and state[0] == target["binding"]
                            else None
                        )
                        pending[name] = (
                            pool.submit(fetch, target, cursor),
                            target,
                            cursor,
                        )
                if time.monotonic() - health_at < 5:
                    stopping.wait(0.5)
                    continue
                health_at = time.monotonic()
                sources = []
                errors = [config_error] if config_error else []
                for name, target in targets.items():
                    state = db.execute(
                        "SELECT last_success,last_error,gap_events,instance_changes,invalid_events FROM performance_http_sources WHERE target_id=%s",
                        (name,),
                    ).fetchone()
                    sources.append(
                        {
                            "target": name,
                            "namespace": target["namespace"],
                            "last_success": str(state[0]) if state else None,
                            "gap_events": state[2] if state else 0,
                            "instance_changes": state[3] if state else 0,
                            "invalid_events": state[4] if state else 0,
                        }
                    )
                    if state and state[1]:
                        errors.append(name + ":" + state[1])
                health = {
                    "mode": "http_pull",
                    "namespaces": sorted({t["namespace"] for t in targets.values()}),
                    "sources": sources,
                    "storage": storage,
                    "config_error": config_error,
                    "poll_interval_s": 5,
                }
                db.execute(
                    "UPDATE performance_collector_health SET last_poll=now(),last_event=(SELECT max(completed_at) FROM video_performance),"
                    "last_error=%s,inserted_events=(SELECT coalesce(sum(inserted_events),0) FROM performance_http_sources),details=%s WHERE id=1",
                    (";".join(errors)[:2000] or None, Jsonb(health)),
                )
            except Exception as exc:  # noqa: BLE001
                # Reconnect without leaking SQL values.
                print(json.dumps({"http_pull_error": type(exc).__name__}), flush=True)
                if db is not None:
                    db.close()
                db = None
            stopping.wait(0.5)
    if db is not None:
        db.close()


if __name__ == "__main__":
    run()
