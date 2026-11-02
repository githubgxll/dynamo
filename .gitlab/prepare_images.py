#!/usr/bin/env python3
"""Render the existing Dingo image selection for the XB01 GitLab runner."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".gitlab-render"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    matrix = load_module("image_matrix", ROOT / ".github/scripts/prepare_dingo_image_matrix.py")
    renderer = load_module("renderer", ROOT / "container/render.py")
    config = matrix.load_config(ROOT / ".github/dingo-images.json")
    config["registry"] = os.environ["CI_IMAGE_REGISTRY_HOST"]
    config["namespace"] = "openclaw"
    images = matrix.build_matrix(config, os.environ["CI_COMMIT_SHA"], "configured")
    context = yaml.safe_load((ROOT / "container/context.yaml").read_text())
    OUT.mkdir(exist_ok=True)
    mappings = {}
    rows = []
    seen_images = set()

    def mirror(source, name, *, backend=False):
        # Backend bases live directly in openclaw; shared CI bases stay in ci.
        namespace = "openclaw" if backend else "openclaw/ci"
        target = f'{config["registry"]}/{namespace}/{name}'
        mappings[source] = target
        return target

    for image in images:
        # Runtime and pre_runtime can be built from the same commit. Keep their
        # tags distinct even when users configure them in the same repository.
        repository, tag = image["image"].rsplit(":", 1)
        tag = f'{image["framework"]}-{image["docker_target"]}-{tag}'
        if not matrix.TAG_PATTERN.fullmatch(tag):
            raise ValueError("Backend/target image tag exceeds the registry tag limit")
        image["image"] = f"{repository}:{tag}"
        if image["image"] in seen_images:
            raise ValueError("Two configured targets would publish the same image tag")
        seen_images.add(image["image"])
        framework = image["framework"]
        args = argparse.Namespace(framework=framework, device="cuda", target=image["render_target"],
                                  platform="amd64", cuda_version=image["cuda_version"],
                                  make_efa=False, builder_image="", output_short_filename=False,
                                  show_result=False)
        renderer.validate_args(args)
        # Use the renderer's template/context and index injection without writing
        # into container/: generated build inputs live only in CI artifacts.
        template = renderer._make_jinja_env(ROOT / "container").get_template("Dockerfile.template")
        text = template.render(context=context, **renderer._render_context(args, context))
        text = renderer._inject_python_index_mounts(re.sub(r"\n{3,}", "\n\n", text))
        filename = f'{framework}-{image["docker_target"]}.Dockerfile'
        (OUT / filename).write_text(text)
        syntax = re.search(r"^# syntax=(\S+)", text).group(1)
        frontend = mirror(f"docker.io/{syntax}", syntax.removeprefix("docker/"))
        if image["render_target"] == "router":
            mirror("docker.io/library/rust:1.96.1-slim-bookworm", "rust:1.96.1-slim-bookworm")
            mirror("nvcr.io/nvidia/base/ubuntu:noble-20250619", "ubuntu:noble-20250619")
            mirror("ghcr.io/astral-sh/uv:0.10.7", "uv:0.10.7")
        else:
            device = context[framework][f'cuda{image["cuda_version"]}']
            base = f'{device["base_image"]}:{device["base_image_tag"]}'
            if device["base_image"] != "nvcr.io/nvidia/cuda-dl-base":
                raise ValueError(f"No approved XB01 mirror mapping for {base}")
            mirror(base, f'cuda-dl-base:{device["base_image_tag"]}')
            manylinux = (f'manylinux_2_28_x86_64:{context["dynamo"]["manylinux_image_tag"]}'
                         f'@{context["dynamo"]["manylinux_image_digest"]["x86_64"]}')
            mirror(f"quay.io/pypa/{manylinux}", manylinux)
            uv = f'uv:{context["dynamo"]["uv_version"]}'
            mirror(f"ghcr.io/astral-sh/{uv}", uv)
            if framework != "dynamo":
                repositories = {"sglang": "lmsysorg/sglang", "vllm": "vllm/vllm-openai"}
                if device["runtime_image"] != repositories[framework]:
                    raise ValueError("No approved XB01 runtime mirror mapping")
                mirror(f'docker.io/{device["runtime_image"]}:{device["runtime_image_tag"]}',
                       f'{device["runtime_image"].split("/")[-1]}:{device["runtime_image_tag"]}',
                       backend=True)
        rows.append("\t".join((image["image"], filename, image["docker_target"], image["platform"], frontend)))

    (OUT / "images.tsv").write_text("".join(row + "\n" for row in rows))
    policy = {"rules": [{"action": "CONVERT", "selector": {"identifier": f"docker-image://{source}"},
                         "updates": {"identifier": f"docker-image://{target}"}}
                        for source, target in sorted(mappings.items())]}
    (OUT / "source-policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    (OUT / "bases.tsv").write_text("".join(f"{source}\t{target}\n" for source, target in sorted(mappings.items())))
    print(f"Prepared {len(images)} configured image(s); base mirrors must already exist in XB01.")


if __name__ == "__main__":
    main()
