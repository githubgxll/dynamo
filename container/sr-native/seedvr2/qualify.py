"""Kernel + NCCL qualification. Full 17-frame model inference is a separate gate."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    import torch
    import torch.distributed as dist
    from apex.normalization import FusedRMSNorm
    from flash_attn import flash_attn_varlen_func
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world not in (1, 2, 4):
        raise RuntimeError('Use the shared SP1/2/4 qualification topology')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'rank': rank, 'world_size': world, 'status': 'FAIL', 'model_inference': 'NOT_RUN'}
    try:
        if torch.cuda.get_device_capability(local) != (9, 0):
            raise RuntimeError('This compiled candidate is qualified for H100 sm90 only')
        torch.manual_seed(12)
        x = torch.randn((4, 128), device='cuda', dtype=torch.bfloat16)
        norm = FusedRMSNorm(128, eps=1e-5).to(device='cuda', dtype=torch.bfloat16)
        actual = norm(x)
        expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)).to(x.dtype)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        q, k, v = [torch.randn((17, 4, 128), device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        cu = torch.tensor([0, 17], dtype=torch.int32, device='cuda')
        actual = flash_attn_varlen_func(q, k, v, cu, cu, 17, 17, dropout_p=0.0, causal=False)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            expected = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0).float(), k.transpose(0, 1).unsqueeze(0).float(),
                v.transpose(0, 1).unsqueeze(0).float()).squeeze(0).transpose(0, 1).to(q.dtype)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        value = torch.tensor(float(rank + 1), device='cuda')
        dist.all_reduce(value)
        if value.item() != world * (world + 1) / 2:
            raise RuntimeError('NCCL all-reduce result differs')
        torch.cuda.synchronize()
        report.update(status='PASS_GPU_KERNELS_ONLY', device=torch.cuda.get_device_name(local),
                      torch=torch.__version__, cuda=torch.version.cuda)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output_dir / f'kernel-rank-{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
