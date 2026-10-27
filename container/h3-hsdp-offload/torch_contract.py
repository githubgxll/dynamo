# SPDX-License-Identifier: Apache-2.0
"""Verify native conversion methods; never patch Torch internals."""
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent


def method_hash(source, class_name=None, method_name=None):
    tree = ast.parse(textwrap.dedent(source))
    if class_name:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        node = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method_name)
    else:
        node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def verify():
    import torch
    policy = json.loads((HERE / "torch-contract.json").read_text(encoding="utf-8"))
    if torch.__version__ != policy["version"]:
        raise RuntimeError("Unsupported Torch version for native FSDP conversion: " + torch.__version__)
    result = {}
    for entry in policy["methods"]:
        mod = importlib.import_module(entry["module"])
        method = getattr(getattr(mod, entry["class"]), entry["method"])
        actual = method_hash(inspect.getsource(method))
        if actual != entry["ast_sha256"]:
            raise RuntimeError("Native Torch method differs; review archived parent source: " + entry["module"] + "." + entry["class"] + "." + entry["method"]
                               + "; expected_ast=" + entry["ast_sha256"] + "; actual_ast=" + actual
                               + "; torch_git=" + str(torch.version.git_version))
        result[entry["module"] + "." + entry["class"] + "." + entry["method"]] = actual
    return {"status": "PASS", "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
            "methods": result, "scope": "Native conversion implementation identity; not a GPU test"}
