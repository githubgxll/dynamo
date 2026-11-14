#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install Dingo's Python startup bootstrap into a site-packages directory.

Writes two files into the target (default: this interpreter's purelib):

  1. ``dingo_python_bootstrap.py`` -- stdlib-only module run on every Python
     startup, including multiprocessing spawn children. It imports
     ``logging.config`` (SGLang calls ``logging.config.dictConfig`` directly
     from spawned workers) and, when DINGO_SITE_PACKAGES names an absolute
     directory, appends that site-packages to sys.path (base stack stays
     first) and pins ``dynamo.__path__`` to its dynamo copy so the base
     image's older dynamo never loads.
  2. ``dingo_python_bootstrap.pth`` -- executes the bootstrap module. A .pth
     file is used instead of sitecustomize.py because base images ship an
     apport sitecustomize.py that shadows any site-packages sitecustomize.py
     (see container/deps/vllm/install_vllm_omni_patches.sh for the same trap).

A dangling or dynamo-less DINGO_SITE_PACKAGES aborts startup (SystemExit):
silently falling back to the stale base dynamo produces ABI mismatches.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BOOTSTRAP_MODULE_NAME = "dingo_python_bootstrap"

BOOTSTRAP_SOURCE = '''\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dingo Python startup bootstrap, installed by
container/deps/sglang/install_dingo_python_bootstrap.py.

Runs on every interpreter start (fresh processes and multiprocessing spawn
children) via the sibling dingo_python_bootstrap.pth file."""

import os
import site
import sys


def _bootstrap() -> None:
    # SGLang configures logging with logging.config.dictConfig directly from
    # spawned workers; keep the submodule importable in every interpreter.
    import logging.config  # noqa: F401

    selected = os.environ.get("DINGO_SITE_PACKAGES", "")
    if not selected:
        return

    # The selection must be an absolute, normalized path: it is compared
    # against sys.path and joined onto submodule imports.
    if not os.path.isabs(selected):
        raise SystemExit(
            f"DINGO_SITE_PACKAGES must be an absolute path, got {selected!r}"
        )
    selected = os.path.normpath(selected)

    dynamo_dir = os.path.join(selected, "dynamo")
    if not os.path.isdir(dynamo_dir):
        # Fail closed: a missing directory (or one without a dynamo copy)
        # must abort startup rather than let a worker silently run the stale
        # base dynamo (ABI mismatch). Plain exceptions raised from a .pth
        # hook are caught and logged by the site module while startup
        # continues, so escalate to SystemExit, which aborts the
        # interpreter on every CPython we ship.
        raise SystemExit(
            "DINGO_SITE_PACKAGES must contain a dynamo/ directory: "
            f"missing {dynamo_dir!r} (unset the variable or fix the "
            "deployment path)"
        )

    # Append the selected site-packages so the base SGLang/torch stack keeps
    # import priority; only names the base environment lacks resolve into
    # the Dingo copy. addsitedir (not a raw append) also honors the selected
    # directory's own .pth files.
    if selected not in sys.path:
        site.addsitedir(selected)

    # Pin dynamo to the selected copy. dynamo installs as a namespace
    # package (no __init__.py), so importing it first is safe and leaves
    # standard namespace metadata (__spec__) intact; redirecting __path__
    # then sends every submodule import to the selected directory alone.
    import dynamo

    if dynamo.__path__ != [dynamo_dir]:
        dynamo.__spec__.submodule_search_locations = [dynamo_dir]
        dynamo.__path__ = [dynamo_dir]


_bootstrap()
'''

PTH_FILENAME = f"{BOOTSTRAP_MODULE_NAME}.pth"
PTH_SOURCE = f"import {BOOTSTRAP_MODULE_NAME}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install the Dingo Python startup bootstrap (.pth hook)."
    )
    parser.add_argument(
        "--site-packages",
        default=None,
        help=(
            "Target site-packages directory. "
            "Defaults to this interpreter's purelib directory."
        ),
    )
    args = parser.parse_args(argv)

    site_packages = (
        Path(args.site_packages)
        if args.site_packages
        else Path(sysconfig_purelib())
    )

    if not site_packages.is_dir():
        print(
            f"install_dingo_python_bootstrap: site-packages not found: {site_packages}",
            file=sys.stderr,
        )
        return 1

    # Write the bootstrap module first, the .pth trigger last: a half-finished
    # install never activates an incomplete module.
    module_path = site_packages / f"{BOOTSTRAP_MODULE_NAME}.py"
    module_path.write_text(BOOTSTRAP_SOURCE, encoding="utf-8")
    os.chmod(module_path, 0o644)
    pth_path = site_packages / PTH_FILENAME
    pth_path.write_text(PTH_SOURCE, encoding="utf-8")
    os.chmod(pth_path, 0o644)

    print(f"OK: installed {module_path}")
    print(f"OK: installed {pth_path}")
    return 0


def sysconfig_purelib() -> str:
    """This interpreter's purelib directory (where .pth files are processed)."""
    import sysconfig

    return sysconfig.get_path("purelib")


if __name__ == "__main__":
    raise SystemExit(main())
