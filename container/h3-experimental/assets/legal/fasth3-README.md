---
license: other
license_name: minimax-h3-community
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3
library_name: minimax-h3
tags:
- video-generation
- text-to-audio-video
- minimax-h3
- lora
- distillation
- dmd2
- few-step
- fastvideo
- fasth3
- preview
---

<p align="center">
  <a href="https://github.com/hao-ai-lab/FastVideo"><img src="https://raw.githubusercontent.com/hao-ai-lab/FastVideo/main/assets/logos/logo.svg" width="320" alt="FastVideo"></a>
</p>

# FastVideo-FastH3-4-step-Preview-v1-LoRA

The compact LoRA releases for FastH3 Preview v1 from
[FastVideo](https://github.com/hao-ai-lab/FastVideo). Each adapter reconstructs
one four-forward FastH3 transformer from the MiniMax H3 base model.

[Blog](https://haoailab.com/blogs/fasth3-preview/) ·
[Recommended full checkpoint](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) ·
[FastH3 collection](https://huggingface.co/collections/FastVideo/fastvideo-fasth3)

> The three VSA adapters require FastVideo's VSA-H3 backend and kernel. Use
> FastVideo's launchers rather than a generic PEFT loader; these adapters also
> contain exact delta and VSA gate tensors.

## Run the recommended adapter

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then use
the CUDA 13 / Blackwell path below. It selects FastVideo's published CUDA
kernel wheel instead of compiling the kernel locally. See the
[installation guide](https://hao-ai-lab.github.io/FastVideo/getting_started/installation/)
for other platforms.

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
uv venv --python 3.12 --seed
source .venv/bin/activate
UV_TORCH_BACKEND=cu130 uv pip install \
  --no-sources-package fastvideo-kernel \
  -e ".[fasth3]"
```

```bash
bash examples/inference/basic/run_fasth3_lora_preview_vsa_datafree.sh \
  --prompt "your prompt" \
  --no-warmup \
  --repeats 1
```

Set `FASTH3_LORA_STRENGTH` to change the adapter strength from its default of
`1.0`. The tested defaults use four B200 GPUs. On other multi-GPU CUDA
systems, follow the installation guide and add
`--no-replicated-dit --vsa-kernel triton --no-fa4` to a VSA launcher. The GPU
count must divide H3's 56 attention heads.

## Variants

| Adapter | Full checkpoint | FastVideo launcher |
|---|---|---|
| [VSA / Data-Free](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/vsa-datafree) | [VSA / Data-Free](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) | `run_fasth3_lora_preview_vsa_datafree.sh` |
| [VSA / Synthetic, step 1300](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/vsa-synthetic-step1300) | [VSA / Synthetic, step 1300](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-Synthetic-Step1300) | `run_fasth3_lora_preview_vsa_synthetic_step1300.sh` |
| [VSA / Synthetic, step 1900](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/vsa-synthetic-step1900) | [VSA / Synthetic, step 1900](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-Synthetic-Step1900) | `run_fasth3_lora_preview_vsa_synthetic_step1900.sh` |
| [Dense / Data-Free](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/dense-datafree) | [Dense / Data-Free](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree) | `run_fasth3_lora_preview_dense_datafree.sh` |

The VSA launchers select VSA-H3 automatically; the dense launcher disables it.
This preview is for text-to-audio-video generation and inherits the
[MiniMax H3 Community License](LICENSE).

## Acknowledgements

We thank [Nuva Lab](https://nuvalab.ai/) for bringing production grounding to FastH3 through its experience with real-world creative video-agent workloads. Its production-aligned post-training insights help bridge open-source research to practical data-assisted distillation for commercial video workflows, with Omni Ref as the next focus.

We thank the [NVIDIA FastGen](https://github.com/NVlabs/FastGen) team for the [DMD2](https://arxiv.org/abs/2405.14867) framework and H3 reference experiment that helped us align the score clock, modality shifts, and backward simulation.

We also thank [MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-H3) for releasing H3-Base, and the [vLLM project](https://vllm.ai/), [NVIDIA](https://www.nvidia.com/en-us/), and [MBZUAI](https://mbzuai.ac.ae/) for their continued sponsorship and support of [FastVideo](https://github.com/hao-ai-lab/FastVideo).
