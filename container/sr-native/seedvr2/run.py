"""Native reference launcher: call pinned official functions without math patches."""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = Path('/opt/sources/SeedVR')


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_argument(value):
    if not re.fullmatch('[a-f0-9]{64}', value):
        raise argparse.ArgumentTypeError('Expected a full lowercase SHA256')
    return value


def validate_geometry(width, height, frames, sp, variant):
    if variant not in ('3b', '7b') or sp not in (1, 2, 4):
        raise ValueError('Reference cohort supports 3b/7b with SP1/2/4; SP8 is not a shared topology')
    if min(width, height) <= 0 or width % 16 or height % 16:
        raise ValueError('SeedVR output width/height must be positive multiples of 16')
    if frames <= 1 or (frames - 1) % (4 * sp):
        raise ValueError('Use a complete video window with (frames-1) divisible by 4*SP; no hidden temporal padding')


def selected_assets(variant):
    rows = json.loads((HERE / 'assets.lock.json').read_text())['assets']
    names = {f'seedvr2_ema_{variant}.pth', 'ema_vae.pth', 'pos_emb.pt', 'neg_emb.pt'}
    return [row for row in rows if row['filename'] in names]


def verify_assets(root, variant):
    evidence = []
    for row in selected_assets(variant):
        path = root / row['filename']
        if not path.is_file() or path.stat().st_size != row['size']:
            raise RuntimeError(f'Missing/wrong-size native checkpoint: {path}')
        actual = digest_file(path)
        if actual != row['sha256']:
            raise RuntimeError(f'Checkpoint SHA256 differs: {path}')
        evidence.append({'file': row['filename'], 'sha256': actual, 'revision': row['revision']})
    return evidence


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def prepare_workspace(args):
    """All writes stay in the new output directory; mounted checkpoints stay untouched."""
    work = args.output / 'work'
    work.mkdir()
    for name in ('common', 'configs_3b', 'configs_7b', 'data', 'models', 'projects'):
        (work / name).symlink_to(SOURCE / name, target_is_directory=True)
    (work / 'ckpts').mkdir()
    for row in selected_assets(args.variant):
        name = row['filename']
        link = work / name if name.endswith('.pt') else work / 'ckpts' / name
        link.symlink_to(args.weights / name)
    (work / 'inputs').mkdir()
    (work / 'inputs' / 'input.mp4').symlink_to(args.input)
    return work


def capture_tensor(path, value):
    import torch
    cpu = value.detach().to('cpu').contiguous()
    if not torch.isfinite(cpu).all():
        raise RuntimeError(f'Nonfinite tensor at {path.name}')
    torch.save(cpu, path)
    return {'file': str(path.name), 'shape': list(cpu.shape), 'dtype': str(cpu.dtype),
            'sha256': digest_file(path), 'min': float(cpu.float().min()), 'max': float(cpu.float().max())}


def worker(args):
    import torch
    import torch.distributed as dist
    from torchvision.io import read_video
    from audit import verify_source
    verify_source()
    rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != args.sp:
        raise RuntimeError('WORLD_SIZE must equal requested SP; no hidden DP replicas')
    os.chdir(args.output / 'work')
    sys.path.insert(0, str(SOURCE))
    official = importlib.import_module(f'projects.inference_seedvr2_{args.variant}')
    # This selects the explicitly documented pre-color-correction diagnostic output.
    official.use_colorfix = False
    frames, _, info = read_video(str(args.input), output_format='TCHW')
    if tuple(frames.shape) != (args.frames, 3, args.input_height, args.input_width):
        raise RuntimeError(f'Frozen input shape changed: {tuple(frames.shape)}')
    if abs(info['video_fps'] - args.fps) > 1e-6:
        raise RuntimeError('Frozen input FPS changed')
    if args.width * args.input_height != args.height * args.input_width:
        raise RuntimeError('Reference cohort forbids aspect-ratio crop')
    input_pixel_sha = hashlib.sha256(frames.contiguous().numpy().tobytes()).hexdigest()
    # Same HWC RGB uint8 per-frame convention used by the FlashVSR+ runner.
    decoded_rgb_frame_sha = [hashlib.sha256(frame.permute(1, 2, 0).contiguous().numpy().tobytes()).hexdigest()
                             for frame in frames]
    del frames
    report = {'status': 'RUNNING', 'rank': rank, 'variant': args.variant, 'sp': args.sp,
              'input_pixel_sha256': input_pixel_sha,
              'decoded_rgb_frame_sha256': decoded_rgb_frame_sha,
              'seed': args.seed, 'boundaries': []}
    boundary = args.output / 'boundaries'
    if rank == 0:
        boundary.mkdir(exist_ok=True)
    runner = official.configure_runner(args.sp)
    report['weight_dtypes'] = {
        name: dict(Counter(str(p.dtype) for p in model.parameters()))
        for name, model in [('dit', runner.dit), ('vae', runner.vae)]}
    write_json(args.output / f'rank-{rank}-loaded.json', report)
    captures = Counter()

    def save(name, value):
        if rank == 0:
            index = captures[name]
            captures[name] += 1
            report['boundaries'].append(capture_tensor(boundary / f'{name}-{index:03}.pt', value))

    native_encode = runner.vae_encode
    def encode(samples):
        if len(samples) != 1 or tuple(samples[0].shape) != (3, args.frames, args.height, args.width):
            raise RuntimeError('Official preprocessing changed canvas/frame count; reject comparison')
        save('preprocessed_input', samples[0])
        result = native_encode(samples)
        save('condition_scaled_latent', result[0])
        return result
    runner.vae_encode = encode

    native_decode = runner.vae.decode
    def decode(latent, *pos, **kw):
        save('decoder_input_native_units', latent)
        return native_decode(latent, *pos, **kw)
    runner.vae.decode = decode

    native_vae_decode = runner.vae_decode
    def vae_decode(latents):
        save('restored_scaled_latent', latents[0])
        result = native_vae_decode(latents)
        if tuple(result[0].shape) != (3, args.frames, args.height, args.width):
            raise RuntimeError('Native decoded shape differs from frozen comparison canvas')
        save('raw_rgb_before_color_clamp', result[0])
        if rank == 0:
            from PIL import Image
            raw = result[0].detach().float().cpu()
            images = raw.permute(1, 2, 3, 0).clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
            (args.output / 'native-png').mkdir()
            for index, image in enumerate(images):
                Image.fromarray(image.numpy()).save(args.output / 'native-png' / f'{index:05}.png')
        return result
    runner.vae_decode = vae_decode

    def dit_hook(module, inputs, output):
        save('dit_velocity', output.vid_sample)
    handle = runner.dit.register_forward_hook(dit_hook)
    started = time.monotonic()
    try:
        official.generation_loop(runner, video_path=str(args.output / 'work' / 'inputs'),
            output_dir=str(args.output / 'native-video'), seed=args.seed,
            res_h=args.height, res_w=args.width, sp_size=args.sp, out_fps=args.fps,
            batch_size=1, cfg_scale=1.0, cfg_rescale=0.0, sample_steps=1)
        torch.cuda.synchronize()
        dist.barrier()
        report.update(status='PASS_NATIVE_FULL_CHAIN', elapsed_seconds_rank_including_capture=time.monotonic() - started,
                      max_memory_allocated=torch.cuda.max_memory_allocated(),
                      scope='Native reference generation only; human quality and Gateway acceptance are separate')
        if rank == 0:
            decoded, _, media = read_video(str(args.output / 'native-video' / 'input.mp4'), output_format='TCHW')
            if tuple(decoded.shape) != (args.frames, 3, args.height, args.width) or abs(media['video_fps']-args.fps) > 1e-6:
                raise RuntimeError('Encoded native-video frame count/canvas/FPS failed')
            report['media_sha256'] = digest_file(args.output / 'native-video' / 'input.mp4')
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        handle.remove()
        write_json(args.output / f'rank-{rank}-result.json', report)
        if dist.is_initialized():
            dist.destroy_process_group()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=('3b', '7b'), required=True)
    p.add_argument('--weights', type=Path, required=True, help='Read-only original .pth and prompt assets directory')
    p.add_argument('--input', type=Path, required=True, help='Already frozen 17/65-frame video, not Omni latents')
    p.add_argument('--input-sha256', type=digest_argument, help='Expected frozen input SHA256; omit only for identity discovery')
    p.add_argument('--output', type=Path, required=True, help='New output directory; existing directory is refused')
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--height', type=int, required=True)
    p.add_argument('--input-width', type=int, default=1280)
    p.add_argument('--input-height', type=int, default=768)
    p.add_argument('--frames', type=int, default=17)
    p.add_argument('--fps', type=float, default=24)
    p.add_argument('--sp', type=int, choices=(1, 2, 4), default=4)
    p.add_argument('--seed', type=int, default=666)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return p


def main():
    args = parser().parse_args()
    args.weights, args.input, args.output = (p.resolve() for p in (args.weights, args.input, args.output))
    validate_geometry(args.width, args.height, args.frames, args.sp, args.variant)
    if args.worker:
        worker(args)
        return
    if args.output.exists():
        raise RuntimeError('Use a new output directory; preserved evidence is never overwritten')
    identity = verify_assets(args.weights, args.variant)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    input_sha256 = digest_file(args.input)
    if args.input_sha256 is not None and input_sha256 != args.input_sha256:
        raise RuntimeError('Frozen input file SHA256 differs; no GPU work was started')
    args.output.mkdir(parents=True)
    write_json(args.output / 'request.json', dict(variant=args.variant, weights=identity,
        input=str(args.input), input_sha256=input_sha256,
        input_sha256_expected=args.input_sha256,
        input_identity_status='MATCHED' if args.input_sha256 else 'RECORDED_ONLY_REQUIRES_COHORT_CHECK', width=args.width,
        height=args.height, frames=args.frames, fps=args.fps, sp=args.sp, seed=args.seed,
        color_fix=False, native_checkpoint_loading=True,
        source_commit=os.environ.get('DYNAMO_COMMIT_SHA')))
    prepare_workspace(args)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={args.sp}', str(HERE / 'run.py'), *sys.argv[1:], '--worker']
    result = subprocess.run(command)
    reports = [json.loads((args.output / f'rank-{r}-result.json').read_text())
               for r in range(args.sp) if (args.output / f'rank-{r}-result.json').is_file()]
    passed = (result.returncode == 0 and len(reports) == args.sp
              and all(r['status'] == 'PASS_NATIVE_FULL_CHAIN' for r in reports)
              and len({r['input_pixel_sha256'] for r in reports}) == 1)
    write_json(args.output / 'result.json', {'status': 'PASS_NATIVE_FULL_CHAIN' if passed else 'FAIL',
        'rank_reports': reports, 'exit_code': result.returncode, 'quality_acceptance': 'NOT_RUN'})
    if not passed:
        raise RuntimeError('Native full-chain run failed; preserve this directory for diagnosis')


if __name__ == '__main__':
    main()
