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
SCHEMA = "h3-hsdp-torch-native-contract/2"
FINGERPRINT = "python-ast-fields-json/1"


def canonical_ast(value):
    """Keep every AST field, including empty lists, independently of ast.dump.

    Python 3.13+ omits empty lists in ast.dump by default; 3.12 includes them.
    Location attributes are excluded, but new/unknown semantic fields are NOT
    discarded. A future parser change must fail the source lock for review.
    """
    if isinstance(value, ast.AST):
        return [type(value).__name__, [[key, canonical_ast(item)] for key, item in ast.iter_fields(value)]]
    if isinstance(value, list):
        return [canonical_ast(item) for item in value]
    if value is None:
        return ["none"]
    if value is Ellipsis:
        return ["ellipsis"]
    if type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        return ["float", value.hex()]
    if type(value) is complex:
        return ["complex", value.real.hex(), value.imag.hex()]
    if type(value) is bytes:
        return ["bytes", value.hex()]
    raise TypeError("Unsupported AST field type: " + type(value).__name__)


def unique(nodes, description):
    if len(nodes) != 1:
        raise ValueError("Expected exactly one " + description + "; found " + str(len(nodes)))
    return nodes[0]


def method_hash(source, class_name=None, method_name=None):
    tree = ast.parse(textwrap.dedent(source.replace("\r\n", "\n").replace("\r", "\n")))
    if class_name:
        cls = unique([n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name], "class " + class_name)
        body = cls.body
    else:
        body = tree.body
    node = unique([n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and (method_name is None or n.name == method_name)], "method " + str(method_name))
    encoded = json.dumps([FINGERPRINT, canonical_ast(node)], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def load_policy():
    policy = json.loads((HERE / "torch-contract.json").read_text(encoding="utf-8"))
    if policy.get("schema") != SCHEMA or policy.get("fingerprint") != FINGERPRINT:
        raise ValueError("Unsupported native Torch contract schema/fingerprint; regenerate only after source review")
    return policy


def verify():
    policy = load_policy()
    import torch
    if torch.__version__ != policy["version"]:
        raise RuntimeError("Unsupported Torch version for native FSDP conversion: " + torch.__version__)
    result = {}
    for entry in policy["methods"]:
        mod = importlib.import_module(entry["module"])
        method = getattr(getattr(mod, entry["class"]), entry["method"])
        actual = method_hash(inspect.getsource(method), method_name=entry["method"])
        if actual != entry["ast_sha256"]:
            raise RuntimeError("Native Torch method differs; review archived parent source: " + entry["module"] + "." + entry["class"] + "." + entry["method"]
                               + "; expected_ast=" + entry["ast_sha256"] + "; actual_ast=" + actual
                               + "; torch_git=" + str(torch.version.git_version))
        result[entry["module"] + "." + entry["class"] + "." + entry["method"]] = actual
    return {"status": "PASS", "schema": SCHEMA, "fingerprint": FINGERPRINT,
            "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
            "methods": result, "scope": "Native conversion implementation identity; not a GPU test"}
