#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import os
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def get_components():
    """
    Scan component package directories to get the list of available components.
    Returns full paths to component directories and RuntimeError if no components are found.
    """
    root = os.path.dirname(__file__)
    components_dir = os.path.join(root, "components", "src", "dynamo")

    components = []
    if os.path.exists(components_dir):
        for item in os.listdir(components_dir):
            item_path = os.path.join(components_dir, item)
            if os.path.isdir(item_path) and not item.startswith("."):
                components.append(item_path)

    for component_name in (
        "common",
        "frontend",
        "global_planner",
        "global_router",
        "mocker",
        "planner",
        "profiler",
        "replay",
        "router",
        "sglang",
        "thunderagent_router",
        "vllm",
    ):
        dingo_component = os.path.join(root, "dingo", component_name)
        if os.path.isdir(dingo_component):
            components.append(dingo_component)

    if not components:
        raise RuntimeError(
            f"No components found in {components_dir} or {os.path.join(root, 'dingo')}"
        )

    return components


class VersionWriterHook(BuildHookInterface):
    """
    A Hatch build hook to write the project version to a file.
    """

    def initialize(self, version, build_data):
        """
        This method is called before the build process begins.
        """

        full_version = self.metadata.version
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=True,
            )
            git_version = result.stdout.strip()
            if git_version:
                full_version += f"+{git_version}"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        version_content = f'#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n#  SPDX-License-Identifier: Apache-2.0\n\n# This file is auto-generated at build time\n__version__ = "{full_version}"\n'

        for component in get_components():
            version_file_path = os.path.join(component, "_version.py")
            with open(version_file_path, "w") as f:
                f.write(version_content)
