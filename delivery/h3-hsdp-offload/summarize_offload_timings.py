"""Summarize local H3 offload timing evidence; no cluster or inference actions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile

SCHEMA = "h3-offload-timing/1"
MARKER = "H3_OFFLOAD_TIMING "
WRITE_FAILURE = "H3_OFFLOAD_TIMING_WRITE_FAILED"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def timestamp(value: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ValueError("Timestamp must be an ISO string with timezone")
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must include its timezone: " + value)
    return parsed.astimezone(dt.timezone.utc)


def _number(value, label, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Invalid numeric timing field: " + label)
    if integer and not isinstance(value, int):
        raise ValueError("Counter must be an integer: " + label)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Timing value must be finite and nonnegative: " + label)


def validate_record(record: dict, source: str) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("Unexpected timing schema: " + source)
    for key in ("scope_id", "label"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError("Missing timing " + key + ": " + source)
    rank = record.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < -1:
        raise ValueError("Invalid rank in " + source)
    if record.get("status") not in ("succeeded", "failed", "running"):
        raise ValueError("Unsupported scope status in " + source)
    timestamp(record["started_at"])
    if record["status"] == "running":
        return  # Explicitly excluded; a partial scope is not a zero-duration run.
    if timestamp(record["finished_at"]) < timestamp(record["started_at"]):
        raise ValueError("Scope finished before it started: " + source)
    _number(record.get("wall_seconds"), "wall_seconds")
    for field, integer in (("counts", True), ("seconds", False)):
        values = record.get(field)
        if not isinstance(values, dict):
            raise ValueError("Missing " + field + " in " + source)
        for name, value in values.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Invalid metric name in " + source)
            _number(value, field + "." + name, integer=integer)


def collect(paths):
    """Return records and provenance without extracting any ZIP member."""
    records, ignored, write_failures = [], [], []
    absolute = []

    def accept(record, source):
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            ignored.append({"source": source, "reason": "not_offload_timing_schema"})
            return
        validate_record(record, source)
        records.append((record, source))

    def read_json(stream, source):
        raw = stream.read()
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            marker = SCHEMA.encode() if isinstance(raw, bytes) else SCHEMA
            if marker in raw:
                raise ValueError("Malformed offload timing JSON: " + source) from error
            ignored.append({"source": source, "reason": "unrelated_or_invalid_json"})
            return
        accept(value, source)

    def read_log(stream, source):
        for lineno, raw in enumerate(stream, 1):
            line = ANSI.sub("", raw)
            location = f"{source}:{lineno}"
            if WRITE_FAILURE in line:
                write_failures.append({"source": location, "line": line.strip()})
            if MARKER not in line:
                continue
            payload = line.split(MARKER, 1)[1].strip()
            try:
                value, consumed = json.JSONDecoder().raw_decode(payload)
            except json.JSONDecodeError as error:
                raise ValueError("Malformed offload timing log line: " + location) from error
            if payload[consumed:].strip():
                raise ValueError("Unexpected text after offload timing JSON: " + location)
            accept(value, location)

    def read_file(path):
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as bundle:
                for item in bundle.infolist():
                    if item.is_dir():
                        continue
                    suffix = Path(item.filename).suffix.lower()
                    source = str(path) + "!" + item.filename
                    if suffix == ".json":
                        with bundle.open(item) as stream:
                            read_json(stream, source)
                    elif suffix in (".log", ".txt"):
                        with bundle.open(item) as binary, io.TextIOWrapper(binary, encoding="utf-8", errors="replace") as stream:
                            read_log(stream, source)
        elif path.suffix.lower() == ".json":
            with path.open("rb") as stream:
                read_json(stream, str(path))
        elif path.suffix.lower() in (".log", ".txt"):
            with path.open(encoding="utf-8", errors="replace") as stream:
                read_log(stream, str(path))
        else:
            ignored.append({"source": str(path), "reason": "unsupported_file_type"})

    for raw in paths:
        path = Path(raw).resolve(strict=True)
        absolute.append(str(path))
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_symlink():
                    ignored.append({"source": str(child), "reason": "symlink_not_followed"})
                elif child.is_file() and child.suffix.lower() in (".json", ".zip", ".log", ".txt"):
                    read_file(child)
        else:
            read_file(path)
    return records, absolute, ignored, write_failures


def build_report(paths, *, started_after=None, finished_before=None):
    lower = timestamp(started_after) if started_after else None
    upper = timestamp(finished_before) if finished_before else None
    if lower and upper and lower > upper:
        raise ValueError("--started-after must not exceed --finished-before")
    found, absolute, ignored, write_failures = collect(paths)
    unique = {}
    duplicates = 0
    for record, source in found:
        scope = record["scope_id"]
        if scope in unique:
            if unique[scope]["record"] != record:
                raise ValueError("Conflicting duplicate scope_id " + scope + ": " +
                                 unique[scope]["sources"][0] + " <> " + source)
            unique[scope]["sources"].append(source)
            duplicates += 1
        else:
            unique[scope] = {"record": record, "sources": [source]}
    selected, excluded = [], []
    for scope in sorted(unique):
        item = unique[scope]
        record = item["record"]
        if record["status"] == "running":
            reason = "incomplete_running_scope"
        elif lower and timestamp(record["started_at"]) < lower:
            reason = "starts_before_interval"
        elif upper and timestamp(record["finished_at"]) > upper:
            reason = "finishes_after_interval"
        else:
            selected.append(item)
            continue
        excluded.append({"scope_id": scope, "reason": reason, "sources": item["sources"]})
    if not selected:
        raise ValueError(f"No complete offload timing scopes selected; matched={len(unique)}, excluded={len(excluded)}. No zero-filled report was written.")

    grouped = defaultdict(list)
    for item in selected:
        r = item["record"]
        grouped[(r["rank"], str(r.get("fastpath_env", "unknown")), r["label"])].append(r)

    def aggregate(records):
        counts, seconds = defaultdict(int), defaultdict(float)
        for r in records:
            for k, value in r["counts"].items():
                counts[k] += value
            for k, value in r["seconds"].items():
                seconds[k] += value
        return {"scope_count": len(records),
                "wall_seconds_sum": sum(r["wall_seconds"] for r in records) if records else None,
                "counts_sum": dict(sorted(counts.items())), "seconds_sum": dict(sorted(seconds.items()))}

    groups = []
    for (rank, mode, label), records in sorted(grouped.items()):
        groups.append({"rank": rank, "rank_status": "unknown" if rank < 0 else "known",
                       "fastpath_env": mode, "label": label,
                       "pids": sorted({r.get("pid") for r in records if isinstance(r.get("pid"), int)}),
                       "succeeded": aggregate([r for r in records if r["status"] == "succeeded"]),
                       "failed": aggregate([r for r in records if r["status"] == "failed"])})
    failed = sum(item["record"]["status"] == "failed" for item in selected)
    unknown_rank = sum(item["record"]["rank"] < 0 for item in selected)
    return {"schema": "h3-offload-timing-report/1",
            "status": "COMPLETE_WITH_FLAGS" if failed or unknown_rank or write_failures else "COMPLETE_LOCAL_SUMMARY",
            "inputs": absolute, "filters": {"started_after": started_after, "finished_before": finished_before,
                "rule": "Scope must be complete and entirely inside inclusive ISO time bounds; no case/action identity is inferred"},
            "selected_scope_count": len(selected), "unique_scope_count_before_filter": len(unique),
            "duplicate_copies_removed": duplicates, "failed_scope_count": failed,
            "unknown_rank_scope_count": unknown_rank, "write_failure_messages": write_failures,
            "groups": groups, "records": selected, "excluded_scopes": excluded, "ignored_sources": ignored,
            "interpretation": "Inclusive CPU wall intervals, not GPU kernel time. Never sum rank times into E2E or add nested phase times. Failed scopes are separate from successful scopes. Unknown-rank groups cannot be interpreted as a single known GPU rank."}


def markdown(report):
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    def table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"] +
                         ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows])
    lines = ["# H3 offload 分项计时（本地证据汇总）", "", f"状态：`{report['status']}`。有效完整scope {report['selected_scope_count']}，去重副本 {report['duplicate_copies_removed']}，失败scope {report['failed_scope_count']}，未知rank scope {report['unknown_rank_scope_count']}。", "",
             "计时单位为秒，均为CPU墙钟，未额外同步CUDA。**不要将不同rank的耗时加成端到端时间，也不要将父子计时相加。** `native_apply`包含`tensor_to`及`pin_memory`；`migration_total`包含迁移子阶段，方向计时还包含no-op检查。", "",
             "失败scope单独汇总，不混入成功性能样本。未知rank可能包含多个进程，不能当作已知单rank。缺失指标显示“—”，不填零。没有据时间窗口猜测case或action。", "",
             f"时间过滤：started-after=`{report['filters']['started_after']}`；finished-before=`{report['filters']['finished_before']}`。scope必须完整落在包含边界的区间内。", "",
             table(["rank", "fastpath", "label", "成功scope", "失败scope", "成功wall合计", "失败wall合计"],
                   [["unknown" if g['rank']<0 else g['rank'],g['fastpath_env'],g['label'],g['succeeded']['scope_count'],g['failed']['scope_count'],g['succeeded']['wall_seconds_sum'],g['failed']['wall_seconds_sum']] for g in report['groups']]), "",
             "以上wall为各组所选作用域的累计值，不是整体请求E2E。相同rank跨进程累计也不等于连续墙钟。", ""]
    for g in report['groups']:
        lines += [f"## rank={g['rank']}／fastpath={g['fastpath_env']}／{cell(g['label'])}", "", f"PID：{g['pids']}。", ""]
        for field, title in (("counts_sum", "计数"), ("seconds_sum", "分项秒数（包含关系，不加总）")):
            names = sorted(set(g['succeeded'][field]) | set(g['failed'][field]))
            lines += [title, "", table(["指标", "成功scope累计", "失败scope累计"],
                      [[name,g['succeeded'][field].get(name),g['failed'][field].get(name)] for name in names]), ""]
    lines += ["## 来源与排除", ""] + [f"- `{path}`" for path in report['inputs']]
    lines += ["", f"时间范围外/不完整scope：{len(report['excluded_scopes'])}；非本schema或跳过来源：{len(report['ignored_sources'])}；计时落盘失败日志：{len(report['write_failure_messages'])}。完整来源、重复副本、排除原因及scope记录见同目录JSON。", ""]
    for item in report['excluded_scopes']:
        lines.append(f"- 排除scope `{item['scope_id']}`：`{item['reason']}`。")
    for item in report['write_failure_messages']:
        lines.append(f"- **计时写入失败**：`{item['source']}`。这部分计时可能缺失。")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path,
                        help="Repeat for local JSON, log, ZIP or directory; media are never extracted")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--started-after")
    parser.add_argument("--finished-before")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.input, started_after=args.started_after, finished_before=args.finished_before)
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        result = output / "OFFLOAD-TIMINGS.json"
        result.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        document = output / "OFFLOAD-TIMINGS.md"
        document.write_text(markdown(report), encoding="utf-8")
        print("OFFLOAD_TIMING_REPORT_READY=" + str(document))
        print(f"SCOPES={report['selected_scope_count']}; FAILED={report['failed_scope_count']}; UNKNOWN_RANK={report['unknown_rank_scope_count']}; DUPLICATES_REMOVED={report['duplicate_copies_removed']}")
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"STOP: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
