#!/usr/bin/env python3
"""Summarize actual command exit codes and test reports, never infer success from zero failures."""

import json
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


def summarize(directory, kind):
    directory = Path(directory)
    rows = []
    for line in (directory / "commands.jsonl").read_text().splitlines():
        row = json.loads(line)
        counts = dict(passed=0, failed=0, errors=0, ignored=0, skipped=0, deselected=0)
        log = Path(row["log"]).read_text(errors="replace")
        if kind == "rust":
            for match in re.finditer(
                r"^test result: .*? (\d+) passed; (\d+) failed; (\d+) ignored;",
                log,
                re.M,
            ):
                for key, number in zip(("passed", "failed", "ignored"), match.groups()):
                    counts[key] += int(number)
        elif not row.get("omitted"):
            row["report_complete"] = False
            if row.get("xml") and Path(row["xml"]).exists():
                try:
                    root = ET.parse(row["xml"]).getroot()
                    for case in root.iter("testcase"):
                        if case.find("error") is not None:
                            counts["errors"] += 1
                        elif case.find("failure") is not None:
                            counts["failed"] += 1
                        elif case.find("skipped") is not None:
                            counts["skipped"] += 1
                        else:
                            counts["passed"] += 1
                    row["report_complete"] = True
                except ET.ParseError:
                    # A crash can leave truncated XML; preserve observed outcomes below.
                    counts = dict.fromkeys(counts, 0)
            if not row["report_complete"]:
                clean_log = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", log)
                observed = {}
                for node, outcome in re.findall(
                    r"^(\S+::.*?) (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s+\[|\s*$)",
                    clean_log,
                    re.M,
                ):
                    observed[node] = outcome
                keys = dict(
                    PASSED="passed",
                    FAILED="failed",
                    ERROR="errors",
                    SKIPPED="skipped",
                    XFAIL="skipped",
                    XPASS="passed",
                )
                for outcome in observed.values():
                    counts[keys[outcome]] += 1
                row["reason"] = (
                    "Missing or truncated JUnit XML; counts are partial verbose-log observations"
                )
                row["observed_failures"] = [
                    node
                    for node, outcome in observed.items()
                    if outcome in ("FAILED", "ERROR")
                ]
            matches = re.findall(r"(\d+) deselected", log)
            counts["deselected"] = int(matches[-1]) if matches else 0
        row.update(counts)
        if row.get("omitted"):
            row["status"] = "NOT_RUN"
        elif (
            row["exit_code"] != 0
            or counts["failed"]
            or counts["errors"]
            or row.get("report_complete") is False
        ):
            row["status"] = "FAIL"
        elif (
            sum(counts[k] for k in ("passed", "failed", "errors", "ignored", "skipped"))
            == 0
        ):
            row["status"] = "EMPTY"
        else:
            row["status"] = "PASS"
        rows.append(row)
    totals = {
        k: sum(r[k] for r in rows)
        for k in ("passed", "failed", "errors", "ignored", "skipped", "deselected")
    }
    totals["command_failures"] = sum(
        r["exit_code"] != 0 and not r.get("omitted") for r in rows
    )
    totals["partial_groups"] = sum(r.get("report_complete") is False for r in rows)
    totals["empty_groups"] = sum(r["status"] == "EMPTY" for r in rows)
    totals["not_run_groups"] = sum(r["status"] == "NOT_RUN" for r in rows)
    failed = (
        any(
            r["status"] == "FAIL" or (kind == "python" and r["status"] == "EMPTY")
            for r in rows
        )
        or not rows
    )
    # Empty feature-gated Rust targets remain visible. A completely empty run fails.
    failed |= totals["passed"] + totals["ignored"] + totals["skipped"] == 0
    text = "\n".join(
        f"{r['name']} | {r['status']} | exit={r['exit_code']} | passed={r['passed']} failed={r['failed']} errors={r['errors']} ignored={r['ignored']} skipped={r['skipped']} | {r.get('reason', '')} | {r['log']}"
        for r in rows
    )
    (directory / "SUMMARY.txt").write_text(text + "\n")
    (directory / "summary.json").write_text(
        json.dumps(dict(totals=totals, commands=rows), indent=2) + "\n"
    )
    print(text)
    print("GROUP_TOTAL " + " ".join(f"{k}={v}" for k, v in totals.items()))
    return 1 if failed else (2 if totals["not_run_groups"] else 0)


if __name__ == "__main__":
    sys.exit(summarize(*sys.argv[1:]))
