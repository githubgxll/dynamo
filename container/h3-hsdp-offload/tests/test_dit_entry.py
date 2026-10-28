"""CPU source/control-flow contracts; no torch, FSDP, CUDA, or media execution.

The real patched diffuse, original component context, original transformer
selection, and original sequential-offload lifecycle run with small stubs.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import difflib
import importlib.util
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_dit_entry_source_patches', HERE / 'source_patches.py')
patches = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patches)
patcher = SimpleNamespace(transform=patches.transform_pipeline, TARGET=patches.PIPELINE,
                          sha256=lambda data: hashlib.sha256(data).hexdigest())
LOCK = json.loads((HERE / 'source-patches.lock.json').read_text(encoding='utf-8'))['pipeline']
SOURCE = HERE / 'vendor/original' / patches.PIPELINE
ORIGINAL = SOURCE.read_bytes()
CANDIDATE = patcher.transform(ORIGINAL)
# The vendored pipeline and backend share the same original vllm_omni tree.
BACKEND = SOURCE.parents[2] / 'offloader' / 'sequential_backend.py'
BACKEND_SHA = '2d086ea9a2872ee9ebce3a4bc8a0ad840d355a4d728d541d80bae97457c4bc6a'
BACKEND_BYTES = BACKEND.read_bytes()
if patcher.sha256(BACKEND_BYTES) != BACKEND_SHA:
    raise ValueError('Sequential lifecycle test source is not the locked original')


def extracted(source, wanted):
    tree = ast.parse(source)
    found = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            found.append(copy.deepcopy(node))
        if isinstance(node, ast.ClassDef) and node.name == 'MiniMaxH3Pipeline':
            found.extend(copy.deepcopy(item) for item in node.body
                         if isinstance(item, ast.FunctionDef) and item.name in wanted)
    if {node.name for node in found} != set(wanted):
        raise AssertionError('Requested original source definitions not found')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    return compile(ast.fix_missing_locations(ast.Module(body=[future, *found], type_ignores=[])),
                   '<isolated-reviewed-source>', 'exec')


class Device:
    def __init__(self, name):
        self.name = name
        self.type, _, suffix = name.partition(':')
        self.index = int(suffix) if suffix else None

    def __eq__(self, other):
        return isinstance(other, Device) and self.name == other.name


class Fixture:
    def __init__(self, *, selected=None, dlo=False, source=CANDIDATE):
        self.events = []
        self.cuda = Device('cuda:0')
        self.cpu = Device('cpu')
        self.failure = None
        self.fail_step = None
        self.cleanup_failure = None
        self.cleanup_errors = []
        self.selected = selected
        self.dlo = dlo
        fixture = self

        class Registry:
            def __init__(self, module):
                self.module = module
                self.hooks = {}

            @staticmethod
            def get_or_create(module):
                return module._hook_registry

            def get_hook(self, name):
                return self.hooks.get(name)

            def register_hook(self, name, hook):
                self.hooks[name] = hook

            def remove_hook(self, name):
                self.hooks.pop(name, None)

            def reset_hook(self, name):
                fixture.events.append(('cache-reset', self.module.name))

        class Module:
            def __init__(self, name):
                self.name = name
                self.residency = fixture.cuda
                self.active = False
                self._hook_registry = Registry(self)

            def __call__(self):
                # Ordering model only: native FSDP pre-hook precedes forward.
                # This is NOT a real all-gather or a GPU correctness claim.
                fixture.events.append(('native-pre', self.name))
                self.active = True
                try:
                    hook = self._hook_registry.get_hook('sequential_offload')
                    if hook is not None:
                        hook.pre_forward(self)
                    fixture.events.append(('forward', self.name))
                    return 'video', 'audio'
                finally:
                    self.active = False

        self.main = Module('transformer')
        self.ref = Module('transformers_ref')
        self.encoder = Module('text_encoder')
        logger = SimpleNamespace(debug=lambda *a, **k: None,
                                 exception=lambda *a, **k: self.cleanup_errors.append(a))
        namespace = {
            'contextmanager': contextlib.contextmanager, 'nullcontext': contextlib.nullcontext,
            'torch': SimpleNamespace(device=Device), 'ModelHook': object,
            'HookRegistry': Registry, 'logger': logger,
            'current_omni_platform': SimpleNamespace(is_xpu=lambda: False, empty_cache=lambda: None,
                                                   synchronize=lambda: None, get_free_memory=lambda: 0),
            '_capture_tensor_devices': lambda modules: [], '_restore_tensor_devices': lambda snapshot: None,
            '_VideoVAEPartProxy': type('Proxy', (), {}),
        }
        exec(extracted(BACKEND_BYTES, {'SequentialOffloadHook', 'apply_sequential_offload',
                                      'remove_sequential_offload', '_get_sequential_offload_hook',
                                      'sequential_offload_component'}), namespace)

        def move(module, target, **kwargs):
            if module.residency == target:
                return False
            if module.active:
                raise RuntimeError('contract: migration requested after native pre-forward')
            if self.cleanup_failure is not None and target == self.cpu and module in (self.main, self.ref):
                raise self.cleanup_failure
            self.events.append(('move', module.name, target.name))
            module.residency = target
            return True

        namespace['SequentialOffloadHook']._move_params = staticmethod(move)
        original_context = namespace['sequential_offload_component']

        @contextlib.contextmanager
        def record_context(module):
            self.events.append(('component-enter', module.name))
            try:
                with original_context(module):
                    yield
            finally:
                self.events.append(('component-exit', module.name))

        namespace['sequential_offload_component'] = record_context
        namespace.update(TeaCacheHook=SimpleNamespace(_HOOK_NAME='teacache'),
                         MINIMAX_H3_IMGVID_COND_TIMESTEP=0.1,
                         MINIMAX_H3_AUDIO_REF_COND_TIMESTEP=0.1,
                         minimax_h3_denoise_loop=self.denoise)
        exec(extracted(source, {'diffuse', '_component_on_device', '_transformer_for_task',
                                '_resident_dit_layers_on_device'}), namespace)
        self.original_dlo_context = namespace['_resident_dit_layers_on_device']
        pipeline_type = type('IsolatedPipeline', (), {name: namespace[name] for name in
                             ('diffuse', '_component_on_device', '_transformer_for_task')})
        self.pipeline = pipeline_type()
        self.pipeline.transformer = self.main
        self.pipeline.transformers_ref = self.ref
        self.pipeline.device = self.cuda
        self.pipeline._build_denoise_inputs = self.build_inputs
        self.pipeline._resident_dit_layers_on_device = self.dlo_context
        self.pipeline._dlo_residency_controller = (
            SimpleNamespace(load_resident_layers=lambda: self.events.append(('dlo-layers-load',)),
                            offload_resident_layers=lambda: self.events.append(('dlo-layers-offload',)))
            if dlo else None)
        self.pipeline._cache_dit_runtime = SimpleNamespace(refresh=lambda n: self.events.append(('cache-refresh', n)))
        self.pipeline.progress_bar = self.progress
        self.pipeline._unpack_denoised_rows = self.unpack
        # Legacy DLO reports True here. The new module-only guard must avoid
        # calling this branch, not merely make its visible effects harmless.
        self.pipeline._uses_manual_component_offload = self.forbidden_manual
        if selected is not None:
            self.pipeline._model_cpu_offload_modules = [self.main, self.ref, self.encoder]
            namespace['apply_sequential_offload'](
                [self.main, self.ref], [self.encoder], self.cuda, use_hsdp=True,
                offload_dit_modules=[self.main, self.ref] if 'dit' in selected else [],
                offload_encoder_modules=[self.encoder] if 'text_encoder' in selected else [])
            if 'dit' in selected:
                self.main.residency = self.ref.residency = self.cpu

    def forbidden_manual(self, module):
        raise AssertionError('baseline/DLO must not enter added whole-component context')

    def build_inputs(self, **kwargs):
        self.events.append(('build', kwargs['task']))
        return dict(branch='branch', sampler='sampler', video_rows='initial-video', audio_rows='initial-audio',
                    cond_anchor=None, audio_anchor=None, sigmas_video=list(range(kwargs['num_steps'] + 1)),
                    sigmas_audio=list(range(kwargs['num_steps'] + 1)), video_edit=None, audio_edit=None)

    @contextlib.contextmanager
    def dlo_context(self, *, enabled):
        self.events.append(('dlo-enter', enabled))
        try:
            with self.original_dlo_context(self.pipeline, enabled=enabled):
                yield
        finally:
            self.events.append(('dlo-exit', enabled))

    @contextlib.contextmanager
    def progress(self, *, total):
        self.events.append(('progress-enter', total))
        try:
            yield SimpleNamespace(update=lambda: self.events.append(('progress-step',)))
        finally:
            self.events.append(('progress-exit',))

    def denoise(self, **kwargs):
        self.events.append(('denoise-enter', kwargs['model'].name))
        for step in range(len(kwargs['sigmas_video']) - 1):
            if step == self.fail_step:
                raise self.failure
            video, audio = kwargs['model']()
            kwargs['on_step'](step, video, audio)
        self.events.append(('denoise-exit',))
        return 'finished-video', 'finished-audio'

    def unpack(self, branch, video, audio, **kwargs):
        self.events.append(('unpack',))
        return video, audio

    def run(self, task='ref2va'):
        return self.pipeline.diffuse(task=task, text_embeddings=None, text_tags=None, seed=0,
                                     latent_t=1, latent_h=2, latent_w=2, audio_t=1, num_frames=5,
                                     num_steps=50, video_shift=12, audio_shift=3, base_schedule=None,
                                     visual_condition=None, visual_condition_shape=None,
                                     audio_condition=None, ref_audio_t=None)


class PatchFileTests(unittest.TestCase):
    def test_source_hash_and_exact_candidate_patch_match_lock(self):
        self.assertEqual(patcher.sha256(ORIGINAL), LOCK['before_sha256'])
        self.assertEqual(patcher.sha256(CANDIDATE), LOCK['after_sha256'])
        diff = ''.join(difflib.unified_diff(ORIGINAL.decode().splitlines(keepends=True),
                                          CANDIDATE.decode().splitlines(keepends=True),
                                          fromfile='a/' + patcher.TARGET, tofile='b/' + patcher.TARGET))
        self.assertEqual((HERE / LOCK['patch_file']).read_bytes(), diff.encode())
        self.assertEqual(patcher.sha256(diff.encode()), LOCK['patch_sha256'])

    def test_only_diffuse_changes_in_full_pipeline_ast(self):
        before, after = ast.parse(ORIGINAL), ast.parse(CANDIDATE)
        def remove_diffuse(tree):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MiniMaxH3Pipeline')
            diffuse = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'diffuse')
            cls.body.remove(diffuse)
            return ast.dump(tree), diffuse
        before_other, before_diffuse = remove_diffuse(before)
        after_other, after_diffuse = remove_diffuse(after)
        self.assertEqual(before_other, after_other)
        self.assertNotEqual(ast.dump(before_diffuse), ast.dump(after_diffuse))
        for call in ('minimax_h3_denoise_loop', '_build_denoise_inputs', '_unpack_denoised_rows'):
            def calls(node):
                return [ast.dump(n) for n in ast.walk(node) if isinstance(n, ast.Call)
                        and ((isinstance(n.func, ast.Name) and n.func.id == call)
                             or (isinstance(n.func, ast.Attribute) and n.func.attr == call))]
            self.assertEqual(calls(before_diffuse), calls(after_diffuse))

    def test_transform_does_not_modify_original_source(self):
        before = SOURCE.read_bytes()
        patcher.transform(before)
        self.assertEqual(SOURCE.read_bytes(), before)

    def test_wrong_source_and_second_application_are_rejected(self):
        for body in (ORIGINAL + b'\n', CANDIDATE):
            with self.subTest(sha=patcher.sha256(body)), self.assertRaisesRegex(ValueError, 'Source SHA256'):
                patcher.transform(body)


class DiffuseContracts(unittest.TestCase):
    def test_module_context_precedes_native_prehook_and_spans_all_50_steps_once(self):
        f = Fixture(selected={'dit', 'text_encoder'})
        self.assertEqual(f.run(), ('finished-video', 'finished-audio'))
        events = f.events
        self.assertEqual(events.count(('component-enter', 'transformers_ref')), 1)
        self.assertEqual(events.count(('component-exit', 'transformers_ref')), 1)
        self.assertEqual(events.count(('native-pre', 'transformers_ref')), 50)
        self.assertEqual(events.count(('forward', 'transformers_ref')), 50)
        self.assertEqual(events.count(('progress-step',)), 50)
        self.assertEqual(events.count(('move', 'transformers_ref', 'cuda:0')), 1)
        self.assertEqual(events.count(('move', 'transformers_ref', 'cpu')), 1)
        self.assertLess(events.index(('cache-refresh', 50)), events.index(('component-enter', 'transformers_ref')))
        self.assertLess(events.index(('move', 'transformers_ref', 'cuda:0')), events.index(('native-pre', 'transformers_ref')))
        self.assertLess(events.index(('denoise-exit',)), events.index(('dlo-exit', False)))
        self.assertLess(events.index(('dlo-exit', False)), events.index(('component-exit', 'transformers_ref')))
        self.assertLess(events.index(('component-exit', 'transformers_ref')), events.index(('unpack',)))

    def test_baseline_and_legacy_dlo_keep_original_context_and_call_order(self):
        for dlo in (False, True):
            with self.subTest(dlo=dlo):
                old = Fixture(dlo=dlo, source=ORIGINAL)
                new = Fixture(dlo=dlo)
                self.assertEqual(new.run('fl2va'), old.run('fl2va'))
                self.assertEqual(new.events, old.events)
                self.assertFalse(any(e[0].startswith('component-') for e in new.events))
                self.assertIn(('dlo-enter', True), new.events)
                self.assertEqual(new.events.count(('dlo-layers-load',)), int(dlo))
                self.assertEqual(new.events.count(('dlo-layers-offload',)), int(dlo))

    def test_empty_module_registration_is_same_as_baseline(self):
        f = Fixture()
        f.pipeline._model_cpu_offload_modules = []
        f.run()
        self.assertFalse(any(e[0].startswith('component-') for e in f.events))

    def test_text_encoder_only_keeps_dit_resident_and_offloads_only_selected_encoder(self):
        f = Fixture(selected={'text_encoder'})
        f.run()
        self.assertEqual(f.ref.residency, f.cuda)
        self.assertEqual(f.main.residency, f.cuda)
        self.assertEqual(f.encoder.residency, f.cpu)
        self.assertEqual([e for e in f.events if e[0] == 'move'], [('move', 'text_encoder', 'cpu')])
        self.assertEqual(f.events.count(('component-enter', 'transformers_ref')), 1)

    def test_dit_only_keeps_encoder_resident(self):
        f = Fixture(selected={'dit'})
        f.run()
        self.assertEqual(f.encoder.residency, f.cuda)
        self.assertEqual(f.ref.residency, f.cpu)
        self.assertEqual([e for e in f.events if e[0] == 'move'],
                         [('move', 'transformers_ref', 'cuda:0'), ('move', 'transformers_ref', 'cpu')])

    def test_cleanup_failure_after_success_is_not_hidden(self):
        f = Fixture(selected={'dit', 'text_encoder'})
        f.cleanup_failure = RuntimeError('successful denoise cleanup failed')
        with self.assertRaisesRegex(RuntimeError, 'successful denoise cleanup failed'):
            f.run()
        self.assertIn(('denoise-exit',), f.events)
        self.assertNotIn(('unpack',), f.events)

    def test_ref_and_fl_use_selected_transformer_and_preserve_dlo_enabled_flag(self):
        for task, name, enabled in (('ref2va', 'transformers_ref', False), ('fl2va', 'transformer', True)):
            with self.subTest(task=task):
                f = Fixture(selected={'dit', 'text_encoder'})
                f.run(task)
                self.assertIn(('component-enter', name), f.events)
                self.assertIn(('denoise-enter', name), f.events)
                self.assertIn(('dlo-enter', enabled), f.events)
                self.assertEqual([e[1] for e in f.events if e[0] == 'forward'], [name] * 50)

    def test_ref_only_loading_without_secondary_transformer_uses_main(self):
        f = Fixture(selected={'dit', 'text_encoder'})
        del f.pipeline.transformers_ref
        f.run('ref2va')
        self.assertIn(('component-enter', 'transformer'), f.events)
        self.assertIn(('dlo-enter', True), f.events)

    def test_denoise_failure_runs_cleanup_and_preserves_original_error(self):
        f = Fixture(selected={'dit', 'text_encoder'})
        error = RuntimeError('denoise fixture failed')
        f.fail_step, f.failure = 7, error
        with self.assertRaises(RuntimeError) as caught:
            f.run()
        self.assertIs(caught.exception, error)
        self.assertEqual(f.ref.residency, f.cpu)
        self.assertIn(('progress-exit',), f.events)
        self.assertIn(('dlo-exit', False), f.events)
        self.assertIn(('component-exit', 'transformers_ref'), f.events)
        self.assertNotIn(('unpack',), f.events)

    def test_cleanup_failure_does_not_hide_original_denoise_error(self):
        f = Fixture(selected={'dit', 'text_encoder'})
        error = RuntimeError('original denoise error')
        f.fail_step, f.failure = 1, error
        f.cleanup_failure = RuntimeError('cleanup failed')
        with self.assertRaises(RuntimeError) as caught:
            f.run()
        self.assertIs(caught.exception, error)
        self.assertTrue(f.cleanup_errors)
        self.assertIn(('component-exit', 'transformers_ref'), f.events)

    def test_original_bare_forward_path_demonstrates_late_migration_contract_gap(self):
        f = Fixture(selected={'dit', 'text_encoder'}, source=ORIGINAL)
        with self.assertRaisesRegex(RuntimeError, 'after native pre-forward'):
            f.run()
        self.assertFalse(any(e[0] == 'component-enter' for e in f.events))


if __name__ == '__main__':
    unittest.main()
