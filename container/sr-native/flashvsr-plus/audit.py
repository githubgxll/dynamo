"""Offline CPU audit. Does not claim a CUDA kernel or full model inference passed."""
from __future__ import annotations
import argparse
import importlib.metadata as md
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from runtime_common import SOURCE, HERE, verify_source, write_json, sha256


def native_snapshot(path):
    import torch
    import block_sparse_attn
    import block_sparse_attn_cuda
    root=Path(block_sparse_attn.__file__).resolve().parent
    files=[*root.rglob('*.py'),Path(block_sparse_attn_cuda.__file__).resolve()]
    write_json(path,{'files':{str(p):sha256(p) for p in files},
                     'torch':torch.__version__,'cuda':torch.version.cuda,
                     'cxx11_abi':torch._C._GLIBCXX_USE_CXX11_ABI,
                     'build':'pinned source BSA, SM90, compiled against installed Torch'})


def audit(output):
    output.mkdir(parents=True, exist_ok=True)
    source = verify_source()
    sys.path.insert(0, str(SOURCE))
    import torch
    import cv2
    import numpy as np
    import imageio.v2 as imageio
    import block_sparse_attn
    import block_sparse_attn_cuda
    from inference.presets import stream_config_for_mode
    from inference.preprocessing.input import load_lq_video
    from inference.request import InferenceRequest
    from inference.preprocessing.temporal import expand_for_temporal_compression
    from models.attention_backends.kernels import run_attention
    from inference import run_stream
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('Recipe requires Python 3.10')
    if torch.__version__ != '2.7.1+cu126' or torch.version.cuda != '12.6':
        raise RuntimeError('Torch/CUDA differs from reviewed combination')
    pins=re.findall(r'^([a-zA-Z0-9_.-]+)==([^\s\\]+)',(HERE/'requirements.lock').read_text(),re.M)
    for name,version in pins:
        if md.version(name)!=version:
            raise RuntimeError('Installed locked dependency differs: '+name)
    native=json.loads(Path('/opt/sr-native/build-info/native-extension.json').read_text())
    for path,expected in native['files'].items():
        if sha256(path)!=expected:
            raise RuntimeError('Native BSA installed file differs: '+path)
    if md.version('block_sparse_attn')!='0.0.2':
        raise RuntimeError('BSA distribution version differs')
    if not Path(run_stream.__file__).resolve().is_relative_to(SOURCE):
        raise RuntimeError('Official module import is shadowed')
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    (output/'pip-check.txt').write_text(result.stdout+result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('pip check failed; see pip-check.txt')
    # Small round trip uses actual installed OpenCV, FFmpeg, PIL and native input code.
    with tempfile.TemporaryDirectory(prefix='flashvsr-cpu-') as name:
        video = Path(name)/'input.mp4'
        writer = imageio.get_writer(str(video), fps=24, codec='libx264', macro_block_size=None,
                                   output_params=['-crf','0'])
        try:
            for index in range(17):
                frame=np.zeros((64,128,3),dtype=np.uint8)
                frame[:,:,index%3]=index*12
                writer.append_data(frame)
        finally:
            writer.close()
        tensor,fps=load_lq_video(InferenceRequest(input_path=video,height=128,width=256),stream=True)
        if tuple(tensor.shape)!=(1,3,17,128,256) or abs(fps-24)>1e-5 or not torch.isfinite(tensor).all():
            raise RuntimeError('Native media round trip failed')
        expanded=expand_for_temporal_compression(tensor,4,model_compression=4)
        if expanded.source_frames!=17 or expanded.expanded_frames!=17:
            raise RuntimeError('TC4 unexpectedly pads 17 frames')
    cfg=stream_config_for_mode('base',hardware_profile='generic',attn_kernel='sdpa')
    cfg.validate()
    if cfg.num_inference_steps!=1 or cfg.temporal_compression!=4 or cfg.dtype!='bf16' or cfg.is_hybrid_sliding:
        raise RuntimeError('Official Base/TC4 preset changed')
    q=torch.ones(1,2,256,32)
    masked=run_attention(q,q,q,kernel='sdpa',block_mask=torch.eye(2,dtype=torch.bool)[None,None].expand(1,2,2,2),block_token_size=128)
    if not torch.isfinite(masked).all():
        raise RuntimeError('CPU SDPA reference failed')
    report={'status':'PASS_CPU_ONLY','backend':'flashvsr-plus','euid':os.geteuid(),
            'source_commit':os.environ.get('DYNAMO_COMMIT_SHA',''),
            'upstream':source,'locked_packages':len(pins),'native_files':len(native['files']),
            'python':sys.version,'torch':torch.__version__,
            'cuda_toolkit':torch.version.cuda,'bsa':md.version('block_sparse_attn'),
            'bsa_extension':block_sparse_attn_cuda.__file__,
            'runtime_attention':'block_sparse', 'cpu_attention_test':'SDPA reference only',
            'media_roundtrip':'17 frames, 24 FPS, official PIL bicubic preprocessing',
            'gpu_validation_status':'NOT_RUN','model_weights':'EXTERNAL_NOT_LOADED'}
    write_json(output/'runtime-validation.json', report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output-dir',type=Path)
    group.add_argument('--capture-native-build',type=Path)
    args=parser.parse_args()
    if args.capture_native_build:
        if os.geteuid()!=0:
            raise RuntimeError('Native baseline creation is a root image-build step')
        native_snapshot(args.capture_native_build)
    else:
        try:
            print(json.dumps(audit(args.output_dir),indent=2))
        except Exception as exc:
            args.output_dir.mkdir(parents=True,exist_ok=True)
            write_json(args.output_dir/'runtime-validation.json',
                       {'status':'FAIL','backend':'flashvsr-plus','euid':os.geteuid(),
                        'source_commit':os.environ.get('DYNAMO_COMMIT_SHA',''),
                        'gpu_validation_status':'NOT_RUN',
                        'error':type(exc).__name__+': '+str(exc)})
            raise
