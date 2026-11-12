"""Pure, hash-locked three-anchor SeedVR2 pipeline source patch."""
from __future__ import annotations

import ast
import copy
import hashlib

PARENT_PIPELINE_SHA256 = "bc25563a73707078626826cd665cf2eb7612a7212950a57c2bba32a3f9210658"
PIPELINE_RELATIVE_PATH = "vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py"
HELPER_MODULE = "vllm_omni.diffusion.models.seedvr2.bf16_runtime"

_IMPORT = b"from vllm_omni.diffusion.models.seedvr2 import c14_capture as _c14\n"
_FORWARD = b"    def forward(self, batch: DiffusionRequestBatch) -> list[DiffusionOutput]:\n        outputs = []\n"
_VELOCITY = b"                        velocity = self.transformer(video, self.text, shape, text_shape, timestep, runtime).vid_sample\n"
_PATCHES = (
    ("import_helper", _IMPORT,
     _IMPORT + b"from vllm_omni.diffusion.models.seedvr2 import bf16_runtime as _sr_bf16\n"),
    ("configure_first_in_forward", _FORWARD,
     _FORWARD.replace(b"        outputs = []\n", b"        _sr_bf16.configure_once(self)\n        outputs = []\n")),
    ("scoped_dit_and_checked_velocity", _VELOCITY,
     b"                        velocity = _sr_bf16.run_transformer(self, video, self.text, shape, text_shape, timestep, runtime)\n"
     b"                        velocity = _sr_bf16.velocity_to_fp16(self, velocity)\n"),
)


def _ast_sha(tree):
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


class _ExpectedAST(ast.NodeTransformer):
    """Independently describe the exact permitted semantic changes."""
    def __init__(self):
        self.in_pipeline = False
        self.in_forward = False
        self.counts = {name: 0 for name, _old, _new in _PATCHES}

    def visit_Module(self, node):
        self.generic_visit(node)
        positions = [i for i, child in enumerate(node.body)
                     if isinstance(child, ast.ImportFrom)
                     and child.module == "vllm_omni.diffusion.models.seedvr2"
                     and [(alias.name, alias.asname) for alias in child.names] == [("c14_capture", "_c14")]]
        if len(positions) != 1:
            raise ValueError("SRBF16 import AST anchor differs")
        node.body.insert(positions[0] + 1, ast.parse(
            "from vllm_omni.diffusion.models.seedvr2 import bf16_runtime as _sr_bf16").body[0])
        self.counts["import_helper"] += 1
        return node

    def visit_ClassDef(self, node):
        previous = self.in_pipeline
        self.in_pipeline = node.name == "SeedVR2Pipeline"
        self.generic_visit(node)
        self.in_pipeline = previous
        return node

    def visit_FunctionDef(self, node):
        previous = self.in_forward
        self.in_forward = self.in_pipeline and node.name == "forward"
        self.generic_visit(node)
        if self.in_forward:
            node.body.insert(0, ast.parse("_sr_bf16.configure_once(self)").body[0])
            self.counts["configure_first_in_forward"] += 1
        self.in_forward = previous
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if not self.in_forward:
            return node
        original = ast.parse(_VELOCITY.decode().strip()).body[0]
        if ast.dump(node, include_attributes=False) != ast.dump(original, include_attributes=False):
            return node
        self.counts["scoped_dit_and_checked_velocity"] += 1
        return ast.parse(
            "velocity = _sr_bf16.run_transformer(self, video, self.text, shape, text_shape, timestep, runtime)\n"
            "velocity = _sr_bf16.velocity_to_fp16(self, velocity)\n").body


def apply_source(source):
    """Return (patched bytes, JSON-safe proof); never read/write a source file."""
    if not isinstance(source, bytes):
        raise TypeError("SRBF16 source patch requires bytes")
    original_sha = hashlib.sha256(source).hexdigest()
    if original_sha != PARENT_PIPELINE_SHA256:
        raise ValueError("SRBF16 parent pipeline hash mismatch: " + original_sha)
    original_tree = ast.parse(source, filename=PIPELINE_RELATIVE_PATH)
    compile(original_tree, PIPELINE_RELATIVE_PATH, "exec")
    result, changes = source, []
    for name, old, new in _PATCHES:
        if source.count(old) != 1 or result.count(old) != 1:
            raise ValueError("SRBF16 expected exactly one text anchor: " + name)
        changes.append({"name": name, "original_offset": source.index(old),
                        "removed_bytes": len(old), "inserted_bytes": len(new),
                        "removed_sha256": hashlib.sha256(old).hexdigest(),
                        "inserted_sha256": hashlib.sha256(new).hexdigest()})
        result = result.replace(old, new, 1)
    patched_tree = ast.parse(result, filename=PIPELINE_RELATIVE_PATH)
    compile(patched_tree, PIPELINE_RELATIVE_PATH, "exec")
    modifier = _ExpectedAST()
    expected_tree = modifier.visit(copy.deepcopy(original_tree))
    expected_counts = {name: 1 for name, _old, _new in _PATCHES}
    if modifier.counts != expected_counts or _ast_sha(expected_tree) != _ast_sha(patched_tree):
        raise ValueError("SRBF16 patched AST differs from the three permitted changes")
    proof = {
        "schema": "seedvr2-dit-bf16-source-patch/1",
        "pipeline_relative_path": PIPELINE_RELATIVE_PATH,
        "original_sha256": original_sha,
        "patched_sha256": hashlib.sha256(result).hexdigest(),
        "original_ast_sha256": _ast_sha(original_tree),
        "patched_ast_sha256": _ast_sha(patched_tree),
        "coverage": expected_counts,
        "changes": changes,
        "non_anchor_bytes_unchanged": True,
        "compile_and_ast_verified": True,
    }
    return result, proof
