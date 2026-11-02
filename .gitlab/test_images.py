"""Offline CI contract tests; no build, registry writes or pipeline creation."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("prepare", Path(__file__).with_name("prepare_images.py"))
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
ROOT = prepare.ROOT
HOST = "registry.xb-01.alayanew.com:8443"
DIGEST = "sha256:" + "a" * 64


class ImageTests(unittest.TestCase):
    def test_configured_and_optional_external_images_are_mirrored(self):
        original_loader = prepare.load_module
        for selection in ("configured", "all"):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as temporary:
                out = Path(temporary)

                def load(name, path):
                    module = original_loader(name, path)
                    if name == "image_matrix" and selection == "all":
                        config = module.load_config(ROOT / ".github/dingo-images.json")
                        config = copy.deepcopy(config)
                        for image in list(config["images"]):
                            image["enabled"] = True
                            if image["framework"] in {"vllm", "sglang"}:
                                alternate = copy.deepcopy(image)
                                alternate["docker_target"] = (
                                    "pre_runtime" if image.get("docker_target", "runtime") == "runtime" else "runtime"
                                )
                                config["images"].append(alternate)
                        module.load_config = lambda _: config
                    return module

                with patch.object(prepare, "OUT", out), patch.object(prepare, "load_module", load), patch.dict(
                    os.environ, CI_IMAGE_REGISTRY_HOST=HOST, CI_COMMIT_SHA="b" * 40
                ):
                    prepare.main()
                rows = (out / "images.tsv").read_text().splitlines()
                self.assertEqual(len(rows), 1 if selection == "configured" else 6)
                self.assertEqual(len({row.split("\t")[0] for row in rows}), len(rows))
                policy = json.loads((out / "source-policy.json").read_text())
                mapped = {rule["selector"]["identifier"].removeprefix("docker-image://") for rule in policy["rules"]}
                for rule in policy["rules"]:
                    source = rule["selector"]["identifier"]
                    destination = rule["updates"]["identifier"]
                    if source.startswith(("docker-image://docker.io/lmsysorg/sglang:",
                                          "docker-image://docker.io/vllm/vllm-openai:")):
                        self.assertEqual(destination, f"docker-image://{HOST}/openclaw/{source.rsplit('/', 1)[1]}")
                    else:
                        self.assertTrue(destination.startswith(f"docker-image://{HOST}/openclaw/ci/"))
                for row in rows:
                    image, filename, target, platform, frontend = row.split("\t")
                    self.assertIn(f"/{filename.split('-')[0]}-", f"/{image.rsplit(':', 1)[1]}")
                    self.assertIn(f"-{target}-", image)
                    self.assertTrue(frontend.startswith(f"{HOST}/openclaw/ci/dockerfile:"))
                    self.assertTrue(image.startswith(f"{HOST}/openclaw/"))
                    self.assertEqual(platform, "linux/amd64")
                    text = (out / filename).read_text()
                    stages = set(re.findall(r"^FROM .*? AS (\S+)", text, re.M | re.I)) | {"scratch"}
                    args = dict(re.findall(r"^ARG (\w+)=(.*)$", text, re.M))
                    refs = re.findall(r"^FROM (?:--platform=\S+ )?(\S+)", text, re.M)
                    refs += re.findall(r"^COPY .*?--from=(\S+)", text, re.M)
                    refs += re.findall(r"^# syntax=(\S+)", text, re.M)
                    for ref in refs:
                        for _ in range(4):
                            ref = re.sub(r"\$\{(\w+)\}", lambda match: args[match[1]], ref)
                        if ref in stages:
                            continue
                        if "/" not in ref:
                            ref = "docker.io/library/" + ref
                        elif "." not in ref.split("/")[0]:
                            ref = "docker.io/" + ref
                        self.assertIn(ref, mapped, f"Unmirrored source in {filename}: {ref}")

    def test_build_records_exporter_digest_and_cleans_auth(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / ".gitlab-render").mkdir()
            image = f"{HOST}/openclaw/ai-dingo-sglang:v0.5.21-bbbbbbbbbbbb"
            (work / ".gitlab-render/images.tsv").write_text(
                f"{image}\ttest.Dockerfile\tpre_runtime\tlinux/amd64\t{HOST}/openclaw/ci/dockerfile:1.10.0-labs\n"
            )
            stub = work / "buildctl"
            stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" > "$TEST_LOG"\n'
                            'printf "%s\\n" "$DOCKER_CONFIG" > "$TEST_AUTH_PATH"\n'
                            'while [ "$#" -gt 0 ]; do\n'
                            '  if [ "$1" = --metadata-file ]; then shift; '
                            'printf \'{"containerimage.digest": "%s"}\\n\' "$TEST_DIGEST" > "$1"; fi\n'
                            '  shift\ndone\n')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=f"{work}:{os.environ['PATH']}", CI_IMAGE_REGISTRY_HOST=HOST,
                       CI_IMAGE_REGISTRY_USER_XB01="offline-test", CI_IMAGE_REGISTRY_PASSWORD_XB01="not-a-real-secret",
                       CI_BUILD_PROXY_URL="http://proxy.invalid:1080", CI_BUILD_NO_PROXY="localhost",
                       CI_BUILD_PIP_INDEX="https://index.invalid/simple", CI_COMMIT_SHA="b" * 40,
                       TEST_LOG=str(work / "calls"), TEST_AUTH_PATH=str(work / "auth-path"), TEST_DIGEST=DIGEST)
            result = subprocess.run(["sh", str(ROOT / ".gitlab/build-images.sh")], cwd=work,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((work / ".gitlab-published/images.tsv").read_text(), f"{image}\t{DIGEST}\n")
            calls = (work / "calls").read_text()
            self.assertIn("--addr unix:///run/buildkit/buildkitd.sock", calls)
            self.assertIn("--source-policy-file .gitlab-render/source-policy.json", calls)
            self.assertNotIn("--export-cache", calls)
            self.assertNotIn("not-a-real-secret", calls + result.stdout + result.stderr)
            self.assertFalse(Path((work / "auth-path").read_text().strip()).exists())

    def test_sync_uses_artifact_digest_and_fails_on_mismatch(self):
        for region in ("HD04", "BJ09"):
            for mismatch in (False, True):
                with self.subTest(region=region, mismatch=mismatch), tempfile.TemporaryDirectory() as temporary:
                    work = Path(temporary)
                    (work / ".gitlab-published").mkdir()
                    image = f"{HOST}/openclaw/ai-dingo-sglang:v0.5.21-bbbbbbbbbbbb"
                    (work / ".gitlab-published/images.tsv").write_text(f"{image}\t{DIGEST}\n")
                    stub = work / "skopeo"
                    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_LOG"\n'
                                    'printf "%s\\n" "$REGISTRY_AUTH_FILE" > "$TEST_AUTH_PATH"\n'
                                    'case "$1" in login) read -r password || true ;; '
                                    'inspect) printf "%s\\n" "$TEST_DIGEST" ;; esac\n')
                    stub.chmod(0o755)
                    env = dict(os.environ, PATH=f"{work}:{os.environ['PATH']}", CI_SYNC_REGION=region,
                               CI_IMAGE_REGISTRY_HOST=HOST, TEST_LOG=str(work / "calls"),
                               TEST_AUTH_PATH=str(work / "auth-path"),
                               TEST_DIGEST="sha256:" + "c" * 64 if mismatch else DIGEST)
                    for name in ("XB01", "HD04", "BJ09"):
                        env[f"CI_IMAGE_REGISTRY_USER_{name}"] = "offline-test"
                        env[f"CI_IMAGE_REGISTRY_PASSWORD_{name}"] = "not-a-real-secret"
                    result = subprocess.run(["sh", str(ROOT / ".gitlab/registry-images.sh")], cwd=work,
                                            env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode == 0, not mismatch, result.stderr)
                    calls = (work / "calls").read_text()
                    self.assertIn(f"copy --all --preserve-digests --retry-times 3 docker://{image.rsplit(':', 1)[0]}@{DIGEST}", calls)
                    self.assertNotIn("not-a-real-secret", calls + result.stdout + result.stderr)
                    self.assertFalse(Path((work / "auth-path").read_text().strip()).parent.exists())
                    (work / ".gitlab-render").mkdir()
                    base = f"{HOST}/openclaw/ci/manylinux_2_28_x86_64"
                    (work / ".gitlab-render/bases.tsv").write_text(f"public-source\t{base}:2026.09.05-1@{DIGEST}\n")
                    checked = subprocess.run(["sh", str(ROOT / ".gitlab/registry-images.sh"), "check-bases"],
                                             cwd=work, env=env, capture_output=True, text=True)
                    self.assertEqual(checked.returncode, 0, checked.stderr)
                    calls = (work / "calls").read_text()
                    self.assertIn(f"inspect --raw --retry-times 3 docker://{base}@{DIGEST}", calls)
                    self.assertNotIn(f"docker://{base}:2026.09.05-1@", calls)


if __name__ == "__main__":
    unittest.main()
