# SPDX-License-Identifier: Apache-2.0
"""Observe the exact IPC transfer operations without changing their calls."""
from __future__ import annotations
import ast


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("IPC observation anchor absent/ambiguous: " + before[:100])
    return source.replace(before, after, 1)


def transform(data):
    source = data.decode("utf-8")
    source = once(source, "from vllm_omni.diffusion.data import DiffusionOutput\n",
        "from vllm_omni.diffusion.data import DiffusionOutput\nfrom vllm_omni.diffusion.models.seedvr2 import sr_telemetry as _sr_metrics\n")
    source = once(source,
        "    elif output.output is not None:\n        packed_output = _pack_value_if_large(output.output, d2h_stream=d2h_stream, created=created)\n",
        "    elif output.output is not None:\n        with _sr_metrics.transfer_scope(output.output):\n            packed_output = _pack_value_if_large(output.output, d2h_stream=d2h_stream, created=created)\n")
    source = once(source, "            cpu.copy_(t, non_blocking=True)\n",
        "            with _sr_metrics.copy_stage(\"pinned_cpu\"):\n                cpu.copy_(t, non_blocking=True)\n")
    source = once(source, "            host_tensor.copy_(source, non_blocking=True)\n",
        "            with _sr_metrics.copy_stage(\"registered_shm\"):\n                host_tensor.copy_(source, non_blocking=True)\n")
    source = once(source, "        tensor = tensor.detach().cpu().contiguous()\n",
        "        with _sr_metrics.copy_stage(\"synchronous_cpu\"):\n            tensor = tensor.detach().cpu().contiguous()\n")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_tensor_to_shm")
    lines = source.splitlines(keepends=True)
    first = function.body[1].lineno - 1  # Preserve the original function docstring.
    last = function.end_lineno
    block = lines[first:last]
    lines[first:last] = ["    with _sr_metrics.tensor_transfer(tensor, d2h_stream):\n"] + ["    " + line if line.strip() else line for line in block]
    source = "".join(lines)
    ast.parse(source)
    return source.encode("utf-8")
