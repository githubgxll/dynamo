# SPDX-License-Identifier: Apache-2.0
"""Run the image's repair branch against a real ELF wheel, without compiling Dynamo."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

pytestmark = [
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.skipif(
        sys.platform != "linux" or os.uname().machine != "x86_64",
        reason="the native wheel fixture targets Linux x86_64",
    ),
]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def native_wheel(tmp_path):
    if not shutil.which("cc") or not shutil.which("patchelf"):
        pytest.skip("requires a C compiler and patchelf")
    if importlib.util.find_spec("auditwheel") is None:
        pytest.skip("requires auditwheel")
    libdir = tmp_path / "lib"
    libdir.mkdir()
    media_c = tmp_path / "media.c"
    media_c.write_text("unsigned avcodec_version(void) { return 63; }\n")
    media = libdir / "libavcodec_dynamo.so.63"
    subprocess.run([
        "cc", "-shared", "-fPIC", str(media_c),
        "-Wl,-soname,libavcodec_dynamo.so.63", "-o", str(media),
    ], check=True, timeout=30)
    core_c = tmp_path / "core.c"
    core_c.write_text(
        "extern unsigned avcodec_version(void);\n"
        "unsigned version(void) { return avcodec_version(); }\n"
    )
    core = tmp_path / "_core.abi3.so"
    subprocess.run([
        "cc", "-shared", "-fPIC", str(core_c), str(media), "-o", str(core),
    ], check=True, timeout=30)
    wheel = tmp_path / "ai_dingo_runtime-1.7.0-cp311-abi3-linux_x86_64.whl"
    info = "ai_dingo_runtime-1.7.0.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(core, "ai_dingo_runtime/_core.abi3.so")
        archive.writestr(f"{info}/METADATA", "Metadata-Version: 2.1\nName: ai-dingo-runtime\nVersion: 1.7.0\n")
        archive.writestr(
            f"{info}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp311-abi3-linux_x86_64\n",
        )
        archive.writestr(
            f"{info}/RECORD",
            "\n".join(f"{name},," for name in [*archive.namelist(), f"{info}/RECORD"]) + "\n",
        )
    return wheel, libdir


@pytest.mark.parametrize("device", ["cuda", "cpu", "xpu"])
@pytest.mark.parametrize("media_enabled", [True, False])
def test_runtime_wheel_branch_produces_usable_artifact(tmp_path, native_wheel, device, media_enabled):
    jinja2 = pytest.importorskip("jinja2")
    template = (ROOT / "container/templates/wheel_builder.Dockerfile").read_text()
    branch = template.split('    if [ "$ENABLE_MEDIA_FFMPEG" = "true" ]; then', 1)[1]
    branch = 'if [ "$ENABLE_MEDIA_FFMPEG" = "true" ]; then' + branch.split(
        '    /tmp/use-sccache.sh show-stats "Dynamo Runtime"', 1
    )[0] + '    /tmp/use-sccache.sh show-stats "Dynamo Runtime"'
    command = jinja2.Environment().from_string(branch).render(
        device=device, framework="sglang", target="runtime",
    )
    command = "".join(
        line for line in command.splitlines(keepends=True)
        if not line.lstrip().startswith("#")
    ).replace("\\\n", " ")
    dynamo = tmp_path / "dynamo"
    work = dynamo / "lib/bindings/python"
    work.mkdir(parents=True)
    (dynamo / "dist").mkdir()
    tools = tmp_path / "bin"
    tools.mkdir()
    wheel, libdir = native_wheel
    # Supply a compiled fixture at maturin's output boundary; auditwheel itself is real.
    maturin = tools / "maturin"
    maturin.write_text(
        f"#!{sys.executable}\nimport pathlib,shutil,sys\n"
        "output=pathlib.Path(sys.argv[sys.argv.index('--out')+1]); output.mkdir(parents=True,exist_ok=True)\n"
        f"shutil.copy2({str(wheel)!r}, output / {wheel.name!r})\n"
    )
    maturin.chmod(0o755)
    auditwheel = tools / "auditwheel"
    auditwheel.write_text(f'#!/bin/sh\nexec "{sys.executable}" -m auditwheel "$@"\n')
    auditwheel.chmod(0o755)
    stats = tools / "stats"
    stats.write_text("#!/bin/sh\nexit 0\n")
    stats.chmod(0o755)
    command = command.replace("/opt/dynamo", str(dynamo)).replace("/tmp/use-sccache.sh", str(stats))
    env = dict(
        os.environ, PATH=f"{tools}:{os.environ['PATH']}",
        LD_LIBRARY_PATH=str(libdir), TARGETARCH="amd64",
        ENABLE_MEDIA_FFMPEG="true" if media_enabled else "false",
    )
    result = subprocess.run(
        ["/bin/sh", "-c", command], cwd=work, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    artifacts = list((dynamo / "dist").glob("*.whl"))
    assert len(artifacts) == 1
    with zipfile.ZipFile(artifacts[0]) as archive:
        metadata = archive.read("ai_dingo_runtime-1.7.0.dist-info/WHEEL").decode()
        if media_enabled or device == "cuda":
            policy = {"cuda": "2_28", "cpu": "2_35", "xpu": "2_39"}[device]
            assert f"manylinux_{policy}_x86_64" in metadata
        else:
            assert "Tag: cp311-abi3-linux_x86_64" in metadata
        assert "ai_dingo_runtime/_core.abi3.so" in archive.namelist()
        assert not any(".libs/" in name for name in archive.namelist())
