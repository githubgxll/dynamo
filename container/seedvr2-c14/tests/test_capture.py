"""Capture/source-contract regressions; no claim of CUDA or C14 quality acceptance."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import c14_capture as cap
import source_patches as patches


def sample_profile(root):
    return {"schema": cap.SCHEMA, "run_id": "c14-regression-a", "output_root": str(root),
            "capture_window_starts": cap.SELECTED, "expected_world_size": 8,
            "parent_base_sha256": hashlib.sha256(b"frozen-video").hexdigest(),
            "model_revision": "37255ff8cccfb01071b87f635a5948ca8d53117c",
            "model_sha256": {n: "a" * 64 for n in ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors", "pos_emb.pt")},
            "max_bytes": 50 * 1024**3, "max_tensor_bytes": 1024**3, "min_free_bytes": 1,
            "tolerance_policy_sha256": "b" * 64}


@contextmanager
def fake_dist(peer_error=None):
    dist = types.ModuleType("torch.distributed")
    dist.is_initialized = lambda: True
    dist.get_world_size = lambda: 8
    calls = []
    def gather(values, item):
        calls.append(item)
        values[:] = [item] + [peer_error] + [None] * 6
    dist.all_gather_object = gather
    torch = types.ModuleType("torch")
    torch.distributed = dist
    with mock.patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}):
        yield calls


class CaptureContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.value = sample_profile(self.root)
        self.profile_path = self.root / "profile.json"
        self.profile_path.write_text(json.dumps(self.value), encoding="utf-8")
        self.env = mock.patch.dict(os.environ, {cap.PROFILE_ENV: str(self.profile_path)})
        self.env.start()
        self.job = self.root / "nativejob"
        self.job.mkdir()
        (self.job / "input.mp4").write_bytes(b"frozen-video")
    def tearDown(self):
        self.env.stop()
        cap._active.capture = None
        cap._active.encoding = None
        self.tmp.cleanup()
    def context(self, start=156):
        return cap.window_context(self.job, start, min(17,362-start), 1840,1104,362,7723,"lab")
    def test_profile_accepts_frozen_configuration(self):
        self.assertEqual(cap.profile()["model_revision"], self.value["model_revision"])
    def test_profile_rejects_missing_actual_model_hash(self):
        self.value["model_sha256"].pop("ema_vae_fp16.safetensors")
        with self.assertRaisesRegex(ValueError,"model hash"):
            cap.validate_profile(self.value)
    def test_profile_rejects_malformed_type(self):
        with self.assertRaises(ValueError): cap.validate_profile([])
    def test_profile_rejects_changed_batch(self):
        self.value["capture_window_starts"] = [156,169]
        with self.assertRaises(ValueError): cap.validate_profile(self.value)
    def test_profile_rejects_budget_boolean(self):
        self.value["max_bytes"] = True
        with self.assertRaises(ValueError): cap.validate_profile(self.value)
    def test_profile_rejects_tensor_larger_than_budget(self):
        self.value["max_tensor_bytes"] = self.value["max_bytes"] + 1
        with self.assertRaises(ValueError): cap.validate_profile(self.value)
    def test_profile_rejects_path_traversal_run_id(self):
        self.value["run_id"] = "../old"
        with self.assertRaises(ValueError): cap.validate_profile(self.value)
    def test_default_off_has_no_io_or_torch_dependency(self):
        with mock.patch.dict(os.environ,{cap.PROFILE_ENV:""}), mock.patch.object(cap,"root_for",side_effect=AssertionError("I/O")):
            self.assertIsNone(cap.begin(None,None,None))
            self.assertIsNone(cap.window_context(None,None,None,None,None,None,None,None))
            self.assertIsNone(cap.media(None,"ignored",b"x"))
            with cap.session(None,None,None) as capture:
                self.assertIsNone(capture)
            cap.fused_frame(None,None,None)
            original={"a":1}
            self.assertIs(cap.child_options(None,original),original)
    def test_window_id_is_start_not_arrival_order(self):
        contexts=[self.context(start) for start in (182,130,169,143,156)]
        self.assertEqual([c["start"] for c in contexts],[182,130,169,143,156])
        self.assertEqual(contexts[-1]["window_id"],"nativejob:156")
    def test_window_contains_tolerance_and_padding(self):
        context=self.context(351)
        _,record=cap.context_dir(cap.profile(),context)
        self.assertEqual(record["tolerance_policy_sha256"],"b"*64)
        self.assertEqual(record["padded_frames"],13)
        self.assertEqual(sum(x["padding"] for x in record["frame_map"]),2)
    def test_input_hash_mismatch_rejected(self):
        (self.job/"input.mp4").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError,"source bytes"):self.context()
    def test_wrong_stride_rejected(self):
        with self.assertRaisesRegex(ValueError,"stride"):
            self.context(157)
    def test_partial_window_rejected(self):
        with self.assertRaises(ValueError):
            cap.window_context(self.job,156,16,1840,1104,362,7723,"lab")
    def test_window_cannot_be_overwritten(self):
        self.context()
        with self.assertRaises(FileExistsError): self.context()
    def test_profile_immutable_after_binding(self):
        self.context(130)
        self.value["max_bytes"] += 1
        self.profile_path.write_text(json.dumps(self.value))
        with self.assertRaisesRegex(ValueError,"already bound"):self.context(143)
    def test_different_job_rejected(self):
        self.context()
        old=self.job;self.job=self.root/"other";self.job.mkdir();shutil.copyfile(old/"input.mp4",self.job/"input.mp4")
        with self.assertRaisesRegex(ValueError,"another native job"):self.context(169)
    def test_child_identity_cannot_be_forged(self):
        context=self.context();context["window_id"]="wrong"
        with self.assertRaisesRegex(ValueError,"identity"):cap.context_dir(cap.profile(),context)
    def test_budget_refuses_without_silent_omission(self):
        self.value["max_bytes"]=5
        with self.assertRaisesRegex(RuntimeError,"budget"):cap.reserve(self.value,6)
        self.assertFalse((cap.root_for(self.value)/"budget.json").exists())
    def test_budget_accumulates(self):
        cap.reserve(self.value,10);cap.reserve(self.value,11)
        self.assertEqual(json.loads((cap.root_for(self.value)/"budget.json").read_text())["reserved_bytes"],21)
    def test_selected_media_hash_and_overwrite_guard(self):
        context=self.context();cap.media(context,"window-input.mkv",b"encoded")
        path,_=cap.context_dir(cap.profile(),context)
        self.assertEqual(json.loads((path/"window-input.mkv.json").read_text())["sha256"],hashlib.sha256(b"encoded").hexdigest())
        with self.assertRaises(FileExistsError):cap.media(context,"window-input.mkv",b"new")
    def test_nonselected_media_omits_payload(self):
        context=self.context(0);cap.media(context,"window-input.mkv",b"encoded")
        path,_=cap.context_dir(cap.profile(),context)
        self.assertFalse((path/"window-input.mkv").exists())
    def test_media_path_is_restricted(self):
        context=self.context()
        with self.assertRaises(ValueError): cap.media(context,"../outside",b"x")
    def test_peer_writer_failure_reaches_nonwriters(self):
        with fake_dist("rank0 disk full") as calls:
            with self.assertRaisesRegex(RuntimeError,"disk full"):cap._sync_error(None)
            self.assertEqual(calls,[None])
    def test_nonwriter_records_shape_and_participates(self):
        capture=object.__new__(cap.Capture);capture.tensors={};capture.rank=7;capture.record={"selected":True}
        tensor=types.SimpleNamespace(shape=(1,3,17,4,4),dtype="float16",stride=lambda:(816,272,16,4,1))
        with fake_dist() as calls:
            capture.tensor("decoded_rgb",tensor,"BCTHW")
            self.assertEqual(calls,[None])
        self.assertEqual(capture.tensors["decoded_rgb"]["shape"],[1,3,17,4,4])
    def test_exception_clears_active_hook(self):
        capture=types.SimpleNamespace(path=self.root,rank=0,finish=mock.Mock())
        with mock.patch.object(cap,"begin",return_value=capture):
            with self.assertRaisesRegex(RuntimeError,"model failed"):
                with cap.session(None,None,None):
                    cap._active.capture=capture
                    raise RuntimeError("model failed")
        self.assertIsNone(cap._active.capture)
        capture.finish.assert_not_called()
        self.assertEqual(json.loads((self.root/"failed-rank-0.json").read_text())["status"],"failed")
    def test_failed_window_never_verifies(self):
        context=self.context();path,_=cap.context_dir(cap.profile(),context)
        cap.json_once(path/"failed-rank-0.json",{"status":"failed"})
        with self.assertRaisesRegex(ValueError,"failed rank"):cap.verify_window(path)
    def test_missing_ranks_never_verify(self):
        context=self.context();path,_=cap.context_dir(cap.profile(),context)
        with self.assertRaises(FileNotFoundError):cap.verify_window(path)
    def test_mapping_overlap_sums_to_one(self):
        earlier=cap.mapping(143,17);later=cap.mapping(156,17)
        for i in range(4):self.assertAlmostEqual(earlier[13+i]["blend_weight"]+later[i]["blend_weight"],1)
    def test_mux_records_exact_real_call_and_failure(self):
        invoke=mock.Mock(side_effect=RuntimeError("encoder failed"))
        with self.assertRaises(RuntimeError):cap.run_mux(self.job,invoke,["ffmpeg","-c:v","copy"],check=True)
        record=json.loads((cap.root_for(cap.profile())/"long-audio-mux.json").read_text())
        self.assertEqual(record["argv"],["ffmpeg","-c:v","copy"])
        self.assertEqual(record["status"],"failed")
    def test_generic_encoder_is_noop_without_context(self):
        with cap.encoding_context({},None,None,{"audio":None}):
            cap.muxer_settings(None,None,None,None)
            cap.encoding_route(None,None,None,None,None,None,None,None)
        self.assertFalse((cap.root_for(self.value)/"encoding.json").exists())
    def test_actual_encoder_records_merged_crf_qp_without_mutating(self):
        context=self.context();path,_=cap.context_dir(cap.profile(),context)
        values={"crf":"18","preset":"ultrafast","qp":"0"}
        codec=types.SimpleNamespace(colorspace=2,color_range=0,color_primaries=2,color_trc=2)
        stream=types.SimpleNamespace(codec_context=codec,pix_fmt="yuv420p",width=1840,height=1104,average_rate=24)
        with mock.patch.dict(sys.modules,{"av":types.SimpleNamespace(__version__="fixture")}):
            with cap.encoding_context({"seedvr2_c14":context},{"qp":"0"},None,{}):
                cap.muxer_settings(stream,"h264",24,values)
                cap.encoding_route("direct_planar",[1]*17,(1104,1840,3),"uint8",24,None,1,None)
        record=json.loads((path/"encoding.json").read_text())
        self.assertEqual(record["muxer"]["options"],values)
        self.assertEqual(record["status"],"completed")
        self.assertEqual(values,{"crf":"18","preset":"ultrafast","qp":"0"})
    def test_encoder_failure_preserves_incomplete_evidence(self):
        context=self.context();path,_=cap.context_dir(cap.profile(),context)
        with self.assertRaises(RuntimeError):
            with cap.encoding_context({"seedvr2_c14":context},None,None,{}):
                raise RuntimeError("encoder failed")
        self.assertEqual(json.loads((path/"encoding.json").read_text())["status"],"started")
        self.assertIsNone(cap._active.encoding)


class SourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock=json.loads((HERE/"source-patches.lock.json").read_text())
    def test_six_sources_derived_from_exact_originals(self):
        self.assertEqual(len(self.lock["files"]),6)
        for relative,hashes in self.lock["files"].items():
            original=(HERE/"vendor"/"original"/relative).read_bytes()
            self.assertEqual(hashlib.sha256(original).hexdigest(),hashes["before_sha256"])
            after=patches.transform(relative,original)
            self.assertEqual(hashlib.sha256(after).hexdigest(),hashes["after_sha256"])
            ast.parse(after)
    def test_helper_is_locked(self):
        self.assertEqual(hashlib.sha256((HERE/"c14_capture.py").read_bytes()).hexdigest(),self.lock["helper_sha256"])
    def test_unknown_file_rejected(self):
        with self.assertRaises(ValueError):patches.transform("unknown.py",b"x=1")
    def test_stale_anchor_rejected(self):
        with self.assertRaises(ValueError):patches.transform("pipeline_seedvr2.py",b"x=1")
    def test_all_sources_validated_before_any_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);shutil.copytree(HERE/"vendor"/"original",root,dirs_exist_ok=True)
            paths=list(self.lock["files"]);(root/paths[-1]).write_text("changed")
            expected=(root/paths[0]).read_bytes()
            with self.assertRaises(ValueError):patches.apply_sources(root)
            self.assertEqual((root/paths[0]).read_bytes(),expected)
    def test_rmsnorm_and_numerical_algorithms_not_replaced(self):
        self.assertFalse(any(p.endswith("nadit.py") for p in self.lock["files"]))
        relative="vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py"
        original=(HERE/"vendor"/"original"/relative).read_bytes();after=patches.transform(relative,original)
        def math_nodes(data):
            return {node.name:ast.dump(node) for node in ast.parse(data).body if isinstance(node,ast.FunctionDef) and node.name not in ("finish_video",)}
        self.assertEqual(math_nodes(original),math_nodes(after))
    def test_native_window_bytes_and_order_preserved(self):
        relative="vllm_omni/diffusion/models/seedvr2/long_video.py"
        original=(HERE/"vendor"/"original"/relative).read_bytes();after=patches.transform(relative,original)
        def windows(data):
            tree=ast.parse(data)
            function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="windows")
            function.returns=None
            scope={"source_frames":iter(range(362)),"itertools":itertools,"window":17,"target":362,"OVERLAP":4,"JobError":RuntimeError}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),"windows","exec"),scope)
            return list(scope["windows"]())
        before=windows(original);current=windows(after)
        self.assertEqual(before,[frames for _,frames in current])
        self.assertEqual([start for start,_ in current],list(range(0,352,13)))
    def test_no_new_tensor_conversion_when_capture_disabled(self):
        relative="vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py"
        original=(HERE/"vendor"/"original"/relative).read_bytes();after=patches.transform(relative,original)
        original_finish=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=="finish_video")
        modified_finish=next(n for n in ast.parse(after).body if isinstance(n,ast.FunctionDef) and n.name=="finish_video")
        before_correct=next(n for n in ast.walk(original_finish) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="corrected" for t in n.targets))
        disabled=next(n for n in ast.walk(modified_finish) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=="capture")
        self.assertEqual(ast.dump(before_correct),ast.dump(disabled.body[0]))


if __name__=="__main__":
    unittest.main()
