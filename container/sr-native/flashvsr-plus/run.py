"""Run the fixed official FlashVSR+ Base/TC4 entry with explicit BSA and evidence."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from runtime_common import (SOURCE, HERE, UPSTREAM_REVISION, sha256, verify_assets,
                            verify_source, write_json, require_digest, media_probe,
                            validate_geometry, validate_timeline)


def kernel_check(output):
    import torch
    from models.attention_backends.kernels import run_attention
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one GPU to this native reference process')
    if torch.cuda.get_device_capability(0)!=(9,0):
        raise RuntimeError('This compiled qualification recipe targets H100/Hopper SM90')
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(123)
        q,k,v=[torch.randn(1,2,256,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        mask=torch.tensor([[True,False],[True,True]],device='cuda')[None,None].expand(1,2,2,2)
        actual=run_attention(q,k,v,kernel='block_sparse',block_mask=mask,block_token_size=128)
        reference=run_attention(q,k,v,kernel='sdpa',block_mask=mask,block_token_size=128)
    if not torch.isfinite(actual).all():
        raise RuntimeError('BSA produced nonfinite output')
    torch.testing.assert_close(actual,reference,atol=0.03,rtol=0.03)
    report={'status':'PASS_GPU_KERNEL_ONLY','kernel':'block_sparse','dtype':'bfloat16',
            'max_abs_error':float((actual.float()-reference.float()).abs().max()),
            'device':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0))}
    write_json(output/'kernel-validation.json',report)
    return report


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path)
    p.add_argument('--input-sha256',type=require_digest)
    p.add_argument('--weights',type=Path,default=Path('/weights/flashvsr-plus'))
    p.add_argument('--output-dir',type=Path,required=True,help='New empty writable run directory')
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1152)
    p.add_argument('--frames',type=int,default=17)
    p.add_argument('--fps',type=float,default=24)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--kernel-only',action='store_true')
    args=p.parse_args(argv)
    if not args.kernel_only and (args.input is None or args.input_sha256 is None):
        p.error('--input and --input-sha256 are required for full inference')
    return args


def run(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Refusing to overwrite an existing run')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    verify_source()
    sys.path.insert(0,str(SOURCE))
    import torch
    import cv2
    from PIL import Image
    from inference.output import tensor_to_uint8_frame
    from inference import run_stream as entry
    kernel=kernel_check(args.output_dir)
    if args.kernel_only:
        return kernel
    if sha256(args.input)!=args.input_sha256:
        raise RuntimeError('Frozen input file SHA256 mismatch')
    input_media=media_probe(args.input)
    validate_timeline(input_media,args.frames,args.fps)
    geometry=validate_geometry(input_media['width'],input_media['height'],args.width,args.height)
    assets=verify_assets(args.weights)
    # Read every original frame using the same decoder used by upstream; record
    # hashes before any model resize. This never feeds Omni intermediate tensors.
    cap=cv2.VideoCapture(str(args.input))
    decoded_hashes=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            decoded_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
    finally:
        cap.release()
    if len(decoded_hashes)!=args.frames:
        raise RuntimeError('OpenCV and ffprobe decoded frame counts differ')
    original_paths=entry.model_paths_for_mode
    original_build=entry.build_inference_pipeline
    observed={}
    def external_paths(root,mode,ckpt_dir=None):
        paths=original_paths(args.weights,mode,ckpt_dir)
        paths.context=str(SOURCE/'prompt_tensor/posi_prompt.pth')
        return paths
    def observed_build(config,paths):
        if config.attn_kernel!='block_sparse' or config.temporal_compression!=4 or config.dtype!='bf16':
            raise RuntimeError('Unexpected native model configuration')
        observed['configuration']=dataclasses.asdict(config)
        pipeline=original_build(config,paths)
        real_infer=pipeline.infer
        def capture(*a,**kw):
            result=real_infer(*a,**kw)
            video=result.video
            if video is None or tuple(video.shape)!=(1,3,args.frames,args.height,args.width):
                raise RuntimeError('Native output shape differs from the frozen contract')
            if not torch.isfinite(video).all():
                raise RuntimeError('Native RGB contains NaN/Inf')
            if result.metadata['attn_kernel']!='block_sparse' or result.metadata['color_correction']['enabled']:
                raise RuntimeError('Unexpected attention or color correction')
            if result.metadata['expanded_model_frames']!=args.frames:
                raise RuntimeError('Reference windows must require no temporal padding')
            # Native float tensor before clamp, display mapping or video encoding.
            raw=video.detach().cpu()
            torch.save(raw,args.output_dir/'native-rgb.pt')
            png_dir=args.output_dir/'native-png'; png_dir.mkdir()
            for index,frame in enumerate(raw[0].permute(1,0,2,3)):
                Image.fromarray(tensor_to_uint8_frame(frame)).save(png_dir/f'{index:06d}.png')
            observed['result_metadata']=result.metadata
            observed['raw']={'shape':list(raw.shape),'dtype':str(raw.dtype),
                             'sha256':sha256(args.output_dir/'native-rgb.pt'),
                             'min':float(raw.min()),'max':float(raw.max()),
                             'stage':'native result before display clamp, color correction disabled'}
            return result
        pipeline.infer=capture
        return pipeline
    entry.model_paths_for_mode=external_paths
    entry.build_inference_pipeline=observed_build
    command=['inference.run_stream','--input',str(args.input.resolve()),
             '--output',str((args.output_dir/'native.mp4').resolve()),
             '--mode','base','--height',str(args.height),'--width',str(args.width),
             '--device','cuda:0','--attn-kernel','block_sparse','--hardware-profile','auto',
             '--temporal-compression','4','--decode-every','auto','--seed',str(args.seed),
             '--tiny-decoder-mode','serial','--cache-offload-mode','never','--no-color-correction']
    evidence={'status':'RUNNING','backend':'flashvsr-plus','upstream_revision':UPSTREAM_REVISION,
              'source_commit':os.environ.get('DYNAMO_COMMIT_SHA'),'kernel':kernel,
              'input':{'path':str(args.input),'sha256':args.input_sha256,'media':input_media,
                       'decoded_rgb_frame_sha256':decoded_hashes},
              'geometry':geometry,'assets':assets,'argv':command,
              'precision':'official BF16', 'topology':'single H100',
              'scope':'native reference; no Gateway/H3 load; no quality claim'}
    write_json(args.output_dir/'run.json',evidence)
    previous_argv=sys.argv
    start=time.monotonic()
    try:
        sys.argv=command
        entry.main()
        torch.cuda.synchronize()
        output_media=media_probe(args.output_dir/'native.mp4')
        validate_timeline(output_media,args.frames,args.fps)
        if (output_media['width'],output_media['height'])!=(args.width,args.height):
            raise RuntimeError('Encoded output dimensions differ')
        evidence.update(status='PASS_NATIVE_INFERENCE_MEDIA_CHECKS_HUMAN_REVIEW_REQUIRED',
                        elapsed_seconds=time.monotonic()-start,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        output_media=output_media,output_sha256=sha256(args.output_dir/'native.mp4'),
                        **observed)
    except BaseException as exc:
        evidence.update(status='FAIL',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        sys.argv=previous_argv
        entry.model_paths_for_mode=original_paths
        entry.build_inference_pipeline=original_build
        write_json(args.output_dir/'run.json',evidence)
    return evidence


if __name__=='__main__':
    result=run(parse_args())
    print(json.dumps({'status':result['status'],'output_dir':str(result.get('output_media','see run.json'))},indent=2))
