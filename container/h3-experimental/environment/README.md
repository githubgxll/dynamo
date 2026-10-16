# H3 实验环境层

此目录供 Linux 镜像构建使用。本地 Windows 仅运行单元回归；不要在 K8s 启动时重新安装。

构建顺序：固定main runtime → 根用户安装系统ffmpeg/ffprobe并设置CLI路径 → 此目录完整依赖层 → 四权重 / Ref8补丁 → 最终非root复验。参考main自己的 `examples/backends/vllm/omni/video_audio.Dockerfile`；本轮不新增FastH3 VSA可选kernel。

调用接口：

```bash
# Docker build阶段，root。固定环境值应由受审查的Dockerfile设置。
export DYNAMO_MAIN_SOURCE_SHA=049eece3f32f8d38dc1d8c52d4b793cb30e394bb
export DYNAMO_VLLM_BASE_IMAGE=vllm/vllm-openai:v0.31.0-ubuntu2404@sha256:3fb023cc987cc7c2bca7ca14c1e0aff5ae06bd9431bf07773731ce7e8a682460
bash /opt/h3-experimental/environment/install_environment.sh

# 最终USER dynamo。输入快照仍读取 /opt/dynamo/build-info。
python3 /opt/h3-experimental/environment/audit_environment.py --output-dir /opt/h3-experimental/build-info
python3 /opt/h3-experimental/environment/media_smoke.py --output-dir /opt/h3-experimental/build-info
```

构建端可用BuildKit secrets `pip-index-url`、`uv-default-index`传包索引。脚本读取后不打印索引或凭据；这不修复BuildKit拉取FROM镜像的网络问题。构建全程不关闭TLS。

严格审计包括：完整默认依赖及被递归请求的extras、Python版本要求、真实元数据路径、已知Ubuntu元数据遮蔽的实际import/RECORD验证、所有受保护GPU版本、Omni wheel direct_url/hash凭证。只有政策JSON记录的NCCL与PyNvVideoCodec两条精确例外；未知依赖问题不能跳过。

范围：ai-dynamo/runtime、aisimulate、GMS、vLLM、Omni的**默认依赖**，main `requirements.vllm.txt`下界及uvloop、blake3、soundfile、librosa、ray辅助要求。未整体激活 `ai-dynamo[vllm]` extra；main CUDA模板继承的实际NIXL被固定保护，不能用extra中NIXL1.3.2声明未经核验替换native库，也不声称整个optional extra验收通过。

OpenCV两发行包共享cv2：保留同版本两套metadata满足两条上游要求，headless最后安装成为确定提供者；检查其所有cv2 RECORD哈希、实际import、native版本和真实MP4decode。禁止Pod运行时单独卸载任一个。

最终输出：

- `environment-check.json`：`PASS_EXPERIMENTAL_ENVIRONMENT_CPU_ONLY`。
- `dependency-audit.json`、`uv-pip-check.txt`：所有默认依赖和精确例外明细。
- `media-smoke.json`：`PASS_REAL_CPU_MEDIA_SMOKE`，真实CPU H.264/AAC encode/decode。
- `opencv-provider.json`、`omni-wheel-origin.json`、`installed-environment.json`：路径、哈希与安装清单。
- `dependency-license-inventory.json`：`METADATA_INVENTORY_ONLY_NOT_LEGAL_CLEARANCE`；Python声明许可证及主镜像到扩展层的版本差异。另保存dpkg列表、ffmpeg copyright。

扩展层恢复了H.264/AAC及相关动态库，**不能沿用main精简镜像的royalty-free声明或直接冒充最终许可通过**。这些增量清单供复核，不是完整native/OS许可证扫描。CPU检查也不证明H100内核、模型生成或感知质量。
