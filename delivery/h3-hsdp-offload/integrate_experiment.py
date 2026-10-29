"""Produce an offline, non-runnable integration patch for a frozen experiment.

The source package and existing runs are never modified. The output is a patch
bundle, not a deployment package: published image/source/gate/seal bindings must
be refreshed separately before preparing any candidate Pod.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


TEST_SOURCE = r'''"""Offline tests only; image binding is mocked during manifest preparation."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import resident
import workflow as w


def base_config():
    return {
        "variant": "ref2va", "gpus": 8, "h3_model": "/models/h3/Ref2VA",
        "sr_model": "/opt/sr/seedvr2-3b", "work_root": "/work",
        "image": resident.IMAGE, "source_hashes": {
            "diffusion/models/seedvr2/" + name: "a" * 64
            for name in ("pipeline_seedvr2.py", "long_video.py", "nadit.py", "vae.py")},
        "package_hashes": {name: "a" * 64 for name in
                           ("resident.py", "resident_media.py", "convert_seedvr2.py")},
        "owner": "offline-test", "token": "offline-token",
        "offload_profile": "a", "sr_capable": True,
    }


class OffloadControlTests(unittest.TestCase):
    def test_old_configs_keep_controls_absent(self):
        config = resident.validate_config(base_config())
        self.assertNotIn("offload_fastpath", config)
        self.assertNotIn("offload_timing", config)

    def test_controls_accept_exact_integers_only(self):
        for key in ("offload_fastpath", "offload_timing"):
            for value in (0, 1):
                self.assertEqual(resident.validate_config({**base_config(), key: value})[key], value)
            for value in (True, False, "1", "0", None, 2, -1, 1.0):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    resident.validate_config({**base_config(), key: value})

    def test_environment_injects_only_frozen_h3_controls(self):
        config = resident.validate_config({**base_config(), "offload_fastpath": 0, "offload_timing": 1})
        ambient = {"H3_OFFLOAD_FASTPATH": "1", "H3_OFFLOAD_TIMING": "0",
                   "H3_OFFLOAD_TIMING_DIR": "/unexpected", "H3_UNRELATED": "1"}
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, ambient):
            h3 = resident.environment(folder, "h3", config)
            sr = resident.environment(folder, "sr", config)
            old = resident.environment(folder, "h3", resident.validate_config(base_config()))
        self.assertEqual({key: value for key, value in h3.items() if key.startswith("H3_")},
                         {"H3_OFFLOAD_FASTPATH": "0", "H3_OFFLOAD_TIMING": "1"})
        self.assertFalse(any(key.startswith("H3_") for key in sr))
        self.assertFalse(any(key.startswith("H3_") for key in old))

    def test_controls_do_not_change_sampling_or_sr_argv(self):
        before = resident.validate_config(base_config())
        after = resident.validate_config({**base_config(), "offload_fastpath": 1, "offload_timing": 1})
        self.assertEqual(resident.build_h3(before), resident.build_h3(after))
        self.assertEqual(resident.build_sr(before), resident.build_sr(after))

    def test_prepare_parser_explicit_only(self):
        parser = w.build_parser()
        old = parser.parse_args(["prepare", "--id", "old"])
        self.assertIsNone(old.offload_fastpath)
        self.assertIsNone(old.offload_timing)
        explicit = parser.parse_args(["prepare", "--id", "candidate", "--offload-fastpath", "0", "--offload-timing", "1"])
        self.assertEqual((explicit.offload_fastpath, explicit.offload_timing), (0, 1))
        for value in ("2", "true", "-1"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["prepare", "--id", "bad", "--offload-fastpath", value])

    def test_prepare_freezes_controls_without_pod_or_converter_env(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in w.PAYLOAD:
                (root / name).write_text("# offline test fixture\n", encoding="utf-8")
            with patch.object(w, "ROOT", root), patch.object(w, "require_bound", return_value={}), \
                 patch.object(w, "kubectl", side_effect=AssertionError("No cluster calls")), \
                 contextlib.redirect_stdout(io.StringIO()):
                for explicit in (False, True):
                    run = "explicit" if explicit else "legacy"
                    args = ["prepare", "--id", run, "--sr-capable"]
                    if explicit:
                        args += ["--offload-fastpath", "0", "--offload-timing", "1"]
                    w.prepare(w.build_parser().parse_args(args))
                    directory, deployment = w.load_run(run)
                    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
                    cm = json.loads((directory / "configmap.yaml").read_text(encoding="utf-8"))
                    pod = json.loads((directory / "pod.yaml").read_text(encoding="utf-8"))
                    launch = json.loads((directory / "launch-argv.json").read_text(encoding="utf-8"))
                    self.assertEqual(json.loads(cm["data"]["config.json"]), config)
                    for filename, digest in deployment["files"].items():
                        self.assertEqual(w.file_hash(directory / filename), digest)
                    for container in pod["spec"]["containers"] + pod["spec"]["initContainers"]:
                        self.assertFalse(any(row["name"].startswith("H3_OFFLOAD_") for row in container["env"]))
                    if explicit:
                        self.assertEqual((config["offload_fastpath"], config["offload_timing"]), (0, 1))
                        self.assertEqual(launch["h3_offload_env"], {"H3_OFFLOAD_FASTPATH": "0", "H3_OFFLOAD_TIMING": "1"})
                    else:
                        self.assertNotIn("offload_fastpath", config)
                        self.assertNotIn("offload_timing", config)
                        self.assertNotIn("h3_offload_env", launch)

    def test_programmatic_invalid_controls_fail_before_run_creation(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(w, "ROOT", Path(folder)), \
             patch.object(w, "require_bound", return_value={}):
            for key in ("offload_fastpath", "offload_timing"):
                for value in (True, "1", 2):
                    args = w.build_parser().parse_args(["prepare", "--id", "bad"])
                    setattr(args, key, value)
                    with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                        w.prepare(args)
                    self.assertFalse((Path(folder) / "runs").exists())


if __name__ == "__main__":
    unittest.main()
'''


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Source anchor is missing or ambiguous; refusing an approximate patch: " + before[:100])
    return text.replace(before, after, 1)


def transform(resident, workflow):
    resident = replace_once(resident,
        'allowed = required | {"sr_window_frames", "sr_frame_pixels", "tiers"}',
        'allowed = required | {"sr_window_frames", "sr_frame_pixels", "tiers", "offload_fastpath", "offload_timing"}')
    resident = replace_once(resident,
        '    require(type(config["sr_capable"]) is bool, "sr_capable must be boolean")\n',
        '    require(type(config["sr_capable"]) is bool, "sr_capable must be boolean")\n'
        '    for name in ("offload_fastpath", "offload_timing"):\n'
        '        if name in config:\n'
        '            require(type(config[name]) is int and config[name] in (0, 1), f"{name} must be integer 0 or 1")\n')
    resident = replace_once(resident,
        '    if stage == "sr":\n        env.update(VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS=',
        '    if stage == "h3":\n'
        '        # Only explicitly frozen controls reach H3; inherited H3_* was removed above.\n'
        '        for key, name in (("offload_fastpath", "H3_OFFLOAD_FASTPATH"), ("offload_timing", "H3_OFFLOAD_TIMING")):\n'
        '            if key in config:\n'
        '                env[name] = str(config[key])\n'
        '    if stage == "sr":\n        env.update(VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS=')
    workflow = replace_once(workflow,
        "    run_id = identifier(args.id)\n    identifier(args.namespace, 63)",
        '    controls = {}\n'
        '    for key in ("offload_fastpath", "offload_timing"):\n'
        '        value = getattr(args, key, None)\n'
        '        if value is not None:\n'
        '            if type(value) is not int or value not in (0, 1):\n'
        '                raise ValueError(key + " must be integer 0 or 1")\n'
        '            controls[key] = value\n'
        "    run_id = identifier(args.id)\n    identifier(args.namespace, 63)")
    workflow = replace_once(workflow,
        "    payload['config.json'] = json.dumps(runtime, indent=2)",
        "    runtime.update(controls)  # Omitted controls remain absent in old-compatible configs.\n"
        "    payload['config.json'] = json.dumps(runtime, indent=2)")
    workflow = replace_once(workflow,
        "    save(directory / 'launch-argv.json', launch)",
        '    if controls:\n'
        '        launch["h3_offload_env"] = {"H3_" + key.upper(): str(value) for key, value in controls.items()}\n'
        "    save(directory / 'launch-argv.json', launch)")
    workflow = replace_once(workflow,
        "    p.add_argument('--sr-capable', action='store_true', help='Mount and CPU-convert SR weights; still load SR only on explicit sr-start')",
        "    p.add_argument('--offload-fastpath', type=int, choices=[0, 1], default=None, help='Freeze an explicit H3-only fastpath control; requires the matching new image')\n"
        "    p.add_argument('--offload-timing', type=int, choices=[0, 1], default=None, help='Freeze H3-only CPU-wall offload telemetry; no added CUDA synchronization')\n"
        "    p.add_argument('--sr-capable', action='store_true', help='Mount and CPU-convert SR weights; still load SR only on explicit sr-start')")
    return {"resident.py": resident, "workflow.py": workflow, "tests/test_offload_controls.py": TEST_SOURCE}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def function_ast(source, name):
    return ast.dump(next(node for node in ast.parse(source).body
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name),
                    include_attributes=False)


def verify_unchanged_contracts(original, candidate):
    contracts = {"resident.py": ("build_h3", "build_sr"),
                 "workflow.py": ("load_run", "environment_contract", "validate_pod_spec")}
    for filename, names in contracts.items():
        for name in names:
            if function_ast(original[filename], name) != function_ast(candidate[filename], name):
                raise ValueError("Unexpected change to locked contract: " + filename + ":" + name)


def generate(source, output):
    source = Path(source).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output must not exist; never write over any package or runs")
    if source == output.resolve() or source in output.resolve().parents:
        raise ValueError("Output must be outside the immutable source package")
    if not output.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    for name in ("resident.py", "workflow.py", "image_binding.py", "image-repair-lock.json", "console_output.py"):
        if not (source / name).is_file() or (source / name).is_symlink():
            raise ValueError("Missing or symlinked source input: " + name)
    raw = {name: (source / name).read_bytes() for name in ("resident.py", "workflow.py")}
    original = {name: value.decode("utf-8-sig").replace("\r\n", "\n") for name, value in raw.items()}
    candidate = transform(original["resident.py"], original["workflow.py"])
    for name, text in candidate.items():
        compile(text, name, "exec")
    verify_unchanged_contracts(original, candidate)
    with tempfile.TemporaryDirectory(prefix="h3-offload-integration-", dir=output.parent) as temporary:
        temp = Path(temporary)
        # Code-only, disposable fixture: no runs, inputs, artifacts, or published seal are copied.
        for path in source.iterdir():
            if path.is_file() and not path.is_symlink() and path.suffix == ".py":
                shutil.copyfile(path, temp / path.name)
        shutil.copyfile(source / "image-repair-lock.json", temp / "image-repair-lock.json")
        for name, text in candidate.items():
            target = temp / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="\n")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "TMP": str(temp), "TEMP": str(temp)}
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-p", "test_offload_controls.py", "-v"],
                                cwd=temp, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    if result.returncode:
        raise RuntimeError("Offline integration tests failed; no output published:\n" + result.stdout + result.stderr)
    if any((source / name).read_bytes() != value for name, value in raw.items()):
        raise RuntimeError("Source changed while preparing patch; no output published")
    patch = "".join("".join(difflib.unified_diff(original.get(name, "").splitlines(keepends=True),
              text.splitlines(keepends=True), fromfile="a/" + name if name in original else "/dev/null",
              tofile="b/" + name, n=0)) for name, text in candidate.items())
    output.mkdir()
    (output / "experiment.patch").write_text(patch, encoding="utf-8", newline="\n")
    (output / "LOCAL-VALIDATION.log").write_text(result.stdout + result.stderr, encoding="utf-8", newline="\n")
    report = {"schema": "h3-offload-experiment-integration/1", "status": "LOCAL_PATCH_TESTED_NOT_DEPLOYABLE",
              "source_package": str(source), "source_sha256": {name: digest(value) for name, value in raw.items()},
              "candidate_sha256": {name: digest(value.encode("utf-8")) for name, value in candidate.items()},
              "patch_sha256": digest(patch.encode("utf-8")), "cluster_calls": 0,
              "source_modified": False, "new_image_bound": False,
              "unchanged_contracts": ["H3 and SR argv", "load_run", "Pod environment contract", "admission validation"],
              "required_before_deployment": ["Publish the matching repaired image and bind its full digest",
                  "Refresh installed-source locks, repair-gate hashes and image recipe identity from the published build",
                  "Apply this patch only to a new code-only package and regenerate its package seal",
                  "Use new RUN/BATCH IDs; never reuse or edit existing frozen configs"]}
    (output / "integration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "README.md").write_text(
        "# Offload experiment integration patch\n\n"
        "Status: **LOCAL_PATCH_TESTED_NOT_DEPLOYABLE**. This directory is a reviewable patch bundle, not a runnable experiment.\n\n"
        "`experiment.patch` is zero-context unified diff. Verify the source hashes from integration.json before using `git apply --unidiff-zero` in a new package. "
        "It changes only the controller/runtime integration and adds seven local contract tests. "
        "The input package, its existing RUNs, image bindings and frozen configs are unchanged.\n\n"
        "After publication, create a new code-only experiment package, apply this diff, refresh the new image binding, "
        "source locks, repair gate hashes and package seal using the actual build evidence. The old image identity "
        "is not runnable as a new candidate. No image digest or commit has been fabricated here.\n\n"
        "Prepare controls (available after integration): `--offload-fastpath 0|1` and `--offload-timing 0|1`. "
        "They default to absent, preserving compatibility with old configs. For a controlled A/B, explicitly freeze "
        "both flags. Only the H3 child receives them; SR and the converter receive neither. Telemetry goes to the "
        "existing H3 log as `H3_OFFLOAD_TIMING` records; `H3_OFFLOAD_TIMING_DIR` is deliberately not passed through.\n\n"
        "The new `launch-argv.json` records explicit `h3_offload_env`; `config.json` and ConfigMap include the same "
        "frozen controls. Admission checks and launch/sampling/parallelism arguments are unchanged.\n\n"
        "Local tests mock image binding only while generating disposable manifests. Their success does not confirm "
        "GPU correctness, performance or quality. See `LOCAL-VALIDATION.log` and `integration.json`.\n",
        encoding="utf-8", newline="\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", required=True, type=Path)
    parser.add_argument("--output-package", required=True, type=Path, help="New, nonexistent patch-bundle directory; not a runnable package")
    args = parser.parse_args(argv)
    report = generate(args.source_package, args.output_package)
    print(report["status"] + "=" + str(args.output_package.absolute()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
