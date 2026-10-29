"""Exercise offline timing provenance, deduplication and comparison boundaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


TOOL = Path(__file__).resolve().parents[1] / "summarize_offload_timings.py"
SPEC = importlib.util.spec_from_file_location("offload_timing_report_test", TOOL)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def event(scope="a", rank=0, *, status="succeeded", mode="1", start="2026-10-29T01:00:00+00:00", finish="2026-10-29T01:00:02+00:00"):
    return {"schema": "h3-offload-timing/1", "scope_id": scope, "rank": rank, "pid": 100 + rank,
            "label": "component:DiT", "fastpath_env": mode, "started_at": start, "finished_at": finish,
            "status": status, "wall_seconds": 2.0, "counts": {"hook_fastpath_hits": 50, "h2d_moves": 1},
            "seconds": {"migration_total": 1.2, "native_apply": 1.0, "tensor_to": 0.9}}


class TimingReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_json_log_zip_sources_deduplicate_without_extracting_media(self):
        a, b = event(), event("b", 1)
        path = self.put("scope.json", a)
        log = self.root / "h3.log"
        log.write_text("prefix\x1b[32m H3_OFFLOAD_TIMING " + json.dumps(a) + "\x1b[0m\n", encoding="utf-8")
        archive = self.root / "download.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("actions/h3.log", "worker H3_OFFLOAD_TIMING " + json.dumps(b) + "\n")
            z.writestr("offload/rank1.json", json.dumps(b))
            z.writestr("unrelated.json", json.dumps({"schema": "other/1"}))
            z.writestr("job/base.mp4", b"not opened as media")
        before = sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*"))
        report = reporter.build_report([path, log, archive])
        self.assertEqual(report["selected_scope_count"], 2)
        self.assertEqual(report["duplicate_copies_removed"], 2)
        self.assertEqual([g["rank"] for g in report["groups"]], [0, 1])
        self.assertEqual(before, sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*")))
        self.assertTrue(all(Path(x).is_absolute() for x in report["inputs"]))
        self.assertTrue(any("download.zip!" in source for item in report["records"] for source in item["sources"]))

    def test_duplicate_conflict_is_rejected_not_averaged(self):
        first = self.put("a.json", event())
        changed = event()
        changed["seconds"]["tensor_to"] = 0.1
        second = self.put("b.json", changed)
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate scope_id a"):
            reporter.build_report([first, second])

    def test_unknown_rank_failed_and_modes_are_separate(self):
        self.put("a.json", event("a", -1, status="failed"))
        self.put("b.json", event("b", 0, mode="0"))
        self.put("c.json", event("c", 0, mode="1"))
        report = reporter.build_report([self.root])
        self.assertEqual(report["status"], "COMPLETE_WITH_FLAGS")
        self.assertEqual(report["unknown_rank_scope_count"], 1)
        self.assertEqual(report["failed_scope_count"], 1)
        self.assertEqual(len(report["groups"]), 3)
        unknown = next(g for g in report["groups"] if g["rank"] < 0)
        self.assertIsNone(unknown["succeeded"]["wall_seconds_sum"])
        self.assertEqual(unknown["failed"]["wall_seconds_sum"], 2.0)
        self.assertEqual(unknown["rank_status"], "unknown")
        self.assertIn("unknown", reporter.markdown(report))

    def test_nested_metrics_remain_separate_without_rank_e2e_sum(self):
        self.put("a.json", event("a", 0))
        self.put("b.json", event("b", 1))
        report = reporter.build_report([self.root])
        self.assertNotIn("wall_seconds_sum", report)
        for g in report["groups"]:
            aggregate = g["succeeded"]
            self.assertEqual(aggregate["wall_seconds_sum"], 2.0)
            self.assertEqual(aggregate["seconds_sum"]["migration_total"], 1.2)
            self.assertEqual(aggregate["seconds_sum"]["native_apply"], 1.0)
            self.assertEqual(aggregate["seconds_sum"]["tensor_to"], 0.9)
        self.assertIn("不要将不同rank", reporter.markdown(report))

    def test_time_filter_requires_complete_scope_inside_inclusive_bounds(self):
        self.put("a.json", event("inside"))
        self.put("b.json", event("starts-early", start="2026-10-29T00:59:59Z"))
        self.put("c.json", event("ends-late", finish="2026-10-29T01:00:03Z"))
        running = event("running", status="running")
        running.pop("finished_at")
        running.pop("wall_seconds")
        self.put("d.json", running)
        report = reporter.build_report([self.root], started_after="2026-10-29T09:00:00+08:00", finished_before="2026-10-29T01:00:02Z")
        self.assertEqual(report["selected_scope_count"], 1)
        self.assertEqual(report["records"][0]["record"]["scope_id"], "inside")
        self.assertEqual({x["reason"] for x in report["excluded_scopes"]},
                         {"starts_before_interval", "finishes_after_interval", "incomplete_running_scope"})

    def test_empty_or_fully_filtered_input_does_not_write_zero_report(self):
        irrelevant = self.put("other.json", {"schema": "other/1"})
        with self.assertRaisesRegex(ValueError, "No complete offload timing scopes"):
            reporter.build_report([irrelevant])
        scope = self.put("scope.json", event())
        with self.assertRaisesRegex(ValueError, "No complete offload timing scopes"):
            reporter.build_report([scope], started_after="2026-10-30T00:00:00Z")
        output = self.root / "report"
        self.assertEqual(reporter.main(["--input", str(irrelevant), "--output-dir", str(output)]), 2)
        self.assertFalse(output.exists())

    def test_invalid_timing_values_and_naive_timestamps_are_rejected(self):
        for bad in (float("nan"), -1.0, True):
            value = event()
            value["seconds"]["tensor_to"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                reporter.build_report([self.put("invalid.json", value)])
        with self.assertRaisesRegex(ValueError, "timezone"):
            reporter.timestamp("2026-10-29T01:00:00")
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            reporter.build_report([], started_after="2026-10-30T00:00:00Z", finished_before="2026-10-29T00:00:00Z")

    def test_malformed_timing_log_fails_without_hiding_truncation(self):
        log = self.root / "h3.log"
        log.write_text('prefix H3_OFFLOAD_TIMING {"schema":"h3-offload-timing/1"\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Malformed offload timing log"):
            reporter.build_report([log])

    def test_write_failure_log_is_flagged_as_missing_evidence(self):
        log = self.root / "h3.log"
        log.write_text("H3_OFFLOAD_TIMING_WRITE_FAILED scope=z type=OSError\nH3_OFFLOAD_TIMING " + json.dumps(event()), encoding="utf-8")
        report = reporter.build_report([log])
        self.assertEqual(report["status"], "COMPLETE_WITH_FLAGS")
        self.assertEqual(len(report["write_failure_messages"]), 1)
        self.assertIn("计时写入失败", reporter.markdown(report))

    def test_cli_outputs_markdown_json_with_absolute_sources(self):
        source = self.put("scope.json", event())
        output = self.root / "report"
        self.assertEqual(reporter.main(["--input", str(source), "--output-dir", str(output)]), 0)
        report = json.loads((output / "OFFLOAD-TIMINGS.json").read_text(encoding="utf-8"))
        self.assertEqual(report["inputs"], [str(source.resolve())])
        self.assertEqual(report["groups"][0]["succeeded"]["counts_sum"]["hook_fastpath_hits"], 50)
        self.assertIn(str(source.resolve()), (output / "OFFLOAD-TIMINGS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
