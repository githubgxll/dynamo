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
                mapped = {line.split("\t")[0] for line in (out / "bases.tsv").read_text().splitlines()}
                self.assertFalse(any("docker.io/registry." in source for source in mapped))
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
                    if 'maturin build --release --locked ${KVBM_FEATURES}' in text:
                        kvbm = text.split('FROM reusable_builder_base AS wheel_builder\n', 1)[1]
                        self.assertLess(kvbm.index('COPY lib/'), kvbm.index('--rust-source-only'))
                        self.assertLess(kvbm.index('COPY dingo/'), kvbm.index('--rust-source-only'))
                        self.assertLess(kvbm.index('--rust-source-only'), kvbm.index('maturin build'))
                        self.assertEqual(kvbm.count('--rust-source-only'), 1)
                        with self.assertRaises(ValueError):
                            prepare.stage_kvbm_aisimulate(text.replace('--release --locked ${KVBM_FEATURES}',
                                                                     '--release ${KVBM_FEATURES}').replace(
                                                                         'COPY dingo/ /opt/dynamo/dingo/',
                                                                         'COPY changed/ /opt/dynamo/dingo/'))
                    elif 'KVBM_FEATURES' in text:
                        self.fail('KVBM build lost its locked source staging')
                    self.assertNotIn("--extra-index-url https://pypi.org/simple", text)
                    self.assertNotIn("https://static.rust-lang.org/rustup/archive/", text)
                    if "rustup-init" in text:
                        self.assertIn("https://rsproxy.cn/rustup/archive/1.28.1/${RUSTARCH}/rustup-init", text)
                        self.assertIn("RUST_VERSION=1.96.1", text)
                    for instruction in re.split(r"(?=^[A-Z]+\b)", text, flags=re.M):
                        if not instruction.startswith("RUN "):
                            continue
                        code = re.sub(r"(?m)^[ \t]*#[^\n]*$", "", instruction)
                        if re.search(r"\bapt(?:-get)?\s", code):
                            self.assertIn("for source in /etc/apt/sources.list", instruction)
                        if re.search(r"\b(?:cargo|maturin)\b", code):
                            self.assertIn("id=cargo-config,target=/usr/local/cargo/config.toml,required=true", instruction)
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

    def test_kvbm_source_patch_resolves_missing_nightly_with_real_cargo(self):
        # Real offline Cargo resolution, not a string-only test. The empty local
        # replacement registry models a mirror (and upstream) missing the pin.
        staging = prepare.load_module("stage_aisimulate", ROOT / "container/deps/stage_aisimulate.py")
        version = "0.13.0-dev.202610060000000067"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkout"
            home = Path(temporary) / "cargo-home"
            vendor = Path(temporary) / "empty-registry"
            for directory in (root / ".cargo", home, vendor):
                directory.mkdir(parents=True)
            (root / ".cargo/config.toml").write_text('[build]\nrustflags = []\n')
            (home / "config.toml").write_text(
                '[source.crates-io]\nreplace-with = "build-mirror"\n'
                '[source.build-mirror]\ndirectory = ' + json.dumps(str(vendor)) + '\n')
            source = Path(temporary) / "reviewed-source"
            core = source / "crates/core"
            (core / "src").mkdir(parents=True)
            (core / "Cargo.toml").write_text(
                f'[package]\nname = "aisimulate-core"\nversion = "{version}"\nedition = "2021"\n')
            (core / "src/lib.rs").write_text('pub fn fixture() {}\n')
            workspaces = [root, root / "lib/bindings/python", root / "lib/bindings/kvbm"]
            for index, workspace in enumerate(workspaces):
                (workspace / "src").mkdir(parents=True)
                (workspace / "src/lib.rs").write_text('pub fn fixture() {}\n')
                (workspace / "Cargo.toml").write_text(
                    f'[package]\nname = "consumer-{index}"\nversion = "1.0.0"\nedition = "2021"\n'
                    f'[workspace]\n[workspace.dependencies]\naisimulate-core = {{ version = "={version}" }}\n'
                    '[dependencies]\naisimulate-core = { workspace = true }\n')
                (workspace / "Cargo.lock").write_text(
                    f'version = 3\n\n[[package]]\nname = "aisimulate-core"\nversion = "{version}"\n'
                    'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
                    f'checksum = "{"a" * 64}"\n\n[[package]]\nname = "consumer-{index}"\n'
                    'version = "1.0.0"\ndependencies = ["aisimulate-core"]\n')
            env = dict(os.environ, CARGO_HOME=str(home), CARGO_NET_OFFLINE="true")

            def metadata(workspace):
                return subprocess.run(['cargo', 'metadata', '--offline', '--locked', '--format-version', '1'],
                                      cwd=workspace, env=env, text=True, capture_output=True)

            failure = metadata(workspaces[2])
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn('aisimulate-core', failure.stderr)
            # Isolate network/stamping only; exercise the production patch and
            # lockfile rewrite for all three independent Cargo workspaces.
            with patch.object(staging, 'prepare_source', return_value=source):
                staging.stage_rust_source(root, Path(temporary) / 'output')
            for workspace in workspaces:
                lock = (workspace / 'Cargo.lock').read_bytes()
                result = metadata(workspace)
                self.assertEqual(result.returncode, 0, result.stderr)
                package = next(p for p in json.loads(result.stdout)['packages'] if p['name'] == 'aisimulate-core')
                self.assertEqual(package['version'], version)
                self.assertIsNone(package['source'])
                self.assertEqual(Path(package['manifest_path']), core / 'Cargo.toml')
                self.assertEqual((workspace / 'Cargo.lock').read_bytes(), lock)

    def test_package_policy_preserves_specialized_sources_and_shell_syntax(self):
        original = ('RUN --mount=type=cache,target=/tmp/cache \\\n    '
                    'apt-get update && printf done\n'
                    'RUN uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu130\n'
                    'RUN wget https://github.com/org/project/archive/v1.2.3.tar.gz\n')
        generated = prepare.apply_package_mirrors(original)
        self.assertIn(original.splitlines()[-2], generated)
        self.assertIn(original.splitlines()[-1], generated)
        self.assertTrue(generated.startswith('RUN --mount=type=cache,target=/tmp/cache \\\n    for source'))
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'ubuntu.sources'
            source.write_text(
                'URIs: http://archive.ubuntu.com/ubuntu\nSuites: noble noble-updates\n'
                'URIs: https://security.ubuntu.com/ubuntu/\n'
                'deb http://deb.debian.org/debian bookworm main\n'
                'deb https://deb.debian.org/debian-security bookworm-security main\n'
                'deb https://security.debian.org/debian-security bookworm-security main\n'
                'URIs: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/\n')
            setup = generated.split('    ', 1)[1].split(' && apt-get update', 1)[0]
            setup = setup.replace('/etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources',
                                  str(source))
            # BSD/macOS sed needs an explicit suffix; Linux CI uses GNU sed.
            setup = setup.replace('sed -i ', 'sed -i.bak ')
            result = subprocess.run(['sh', '-c', setup], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            content = source.read_text()
            self.assertNotIn('ubuntu.com', content)
            self.assertNotIn('debian.org', content)
            self.assertIn('https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security', content)
            self.assertIn('Suites: noble noble-updates', content)
            self.assertIn('https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/', content)

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
                            'cp "$DOCKER_CONFIG/cargo-config.toml" "$TEST_LOG.cargo"\n'
                            'printf "%s\\n" "$PIP_INDEX_URL" "$UV_DEFAULT_INDEX" > "$TEST_LOG.indexes"\n'
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
            self.assertIn("--secret id=cargo-config,src=", calls)
            self.assertIn('replace-with = "build-mirror"', (work / 'calls.cargo').read_text())
            self.assertIn('sparse+https://rsproxy.cn/index/', (work / 'calls.cargo').read_text())
            self.assertEqual((work / 'calls.indexes').read_text(), 'https://index.invalid/simple\n' * 2)
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
