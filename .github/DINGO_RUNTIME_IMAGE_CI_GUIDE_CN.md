# Dingo Runtime 镜像自动化构建说明

本文档说明 `DingoRouter-base` 分支当前的 Runtime 镜像构建流程。默认流程只构建
standalone Dynamo，不继承或安装任何推理框架 Runtime；原有的 vLLM、SGLang
Runtime 构建能力继续保留，可在手动触发时选择。

## 1. 构建流程

代码推送到 `DingoRouter-base` 或手动触发工作流后，self-hosted Runner 会：

1. 读取并校验 `.github/dingo-images.json`。
2. 按触发参数生成镜像矩阵；默认配置只包含 Dynamo。
3. 默认执行以下 Dockerfile 渲染命令：

   ```bash
   python3 container/render.py \
     --framework dynamo \
     --target runtime \
     --cuda-version 13.0 \
     --platform linux/amd64
   ```

4. 从 `container/context.yaml` 的 `dynamo.cuda13.0` 读取 Dynamo 基础镜像。
5. 使用 Docker Buildx 构建并推送镜像。
6. 在 GitHub Actions Job Summary 中输出最终镜像地址。

`dynamo` 配置只声明 `base_image`，不声明 `runtime_image`。因此默认构建的
Dockerfile 直接从 NVIDIA CUDA 基础镜像构建，不会继承推理框架镜像。手动选择
其他 Runtime 时仍沿用原来的对应模板和 `context.yaml` 配置。

## 2. 相关文件

| 文件 | 作用 |
|---|---|
| `.github/workflows/dingo-router-ci.yml` | GitHub Actions 主工作流 |
| `.github/dingo-images.json` | Registry、镜像名、标签和平台配置 |
| `.github/scripts/prepare_dingo_image_matrix.py` | 校验配置并生成构建矩阵 |
| `container/render.py` | 渲染所选 Runtime Dockerfile |
| `container/context.yaml` | 提供 `dynamo.cuda13.0` 基础镜像和构建参数 |
| `container/templates/dynamo_runtime.Dockerfile` | standalone Dynamo Runtime 模板 |

默认工作流只使用 `container/context.yaml` 的 `dynamo` 段；手动覆盖选择会使用
相应的其他配置段。

## 3. 触发方式

### 自动触发

任何提交推送到 `DingoRouter-base` 都会触发构建：

```yaml
on:
  push:
    branches:
      - DingoRouter-base
```

当前没有配置 `paths` 过滤，因此仅修改文档也会触发。

### 手动触发

可以在 GitHub Actions 页面手动运行 `Dingo runtime images`：

| 选项 | 行为 |
|---|---|
| `configured` | 按 `enabled` 构建，当前默认只有 Dynamo |
| `dynamo` | 只构建 standalone Dynamo |
| `vllm` | 只构建 vLLM Runtime |
| `sglang` | 只构建 SGLang Runtime |
| `all` | 构建全部已配置镜像 |

## 4. 当前默认配置

```json
{
  "registry": "registry.hd-04.alayanew.com:8443",
  "namespace": "openclaw",
  "platform": "linux/amd64",
  "cuda_version": "13.0",
  "commit_sha_length": 12,
  "keep_buildkit_state": false,
  "images": [
    {
      "framework": "dynamo",
      "enabled": true,
      "repository": "ai-dingo",
      "tag_prefix": "cu130-runtime"
    },
    {
      "framework": "vllm",
      "enabled": false,
      "repository": "ai-dingo-vllm",
      "tag_prefix": "v0.24.0-cu130-runtime"
    },
    {
      "framework": "sglang",
      "enabled": false,
      "repository": "ai-dingo-sglang",
      "tag_prefix": "v0.5.14-cu130-runtime"
    }
  ]
}
```

当前校验脚本只接受：

```text
Framework: dynamo、vllm、sglang
Platform:  linux/amd64
CUDA:      13.0
```

其他 Framework 会在构建开始前直接失败。默认 `configured` 模式只选择
`enabled: true` 的 Dynamo 项。

## 5. 镜像命名

镜像地址格式：

```text
<registry>/<namespace>/<repository>:<tag_prefix>-<commit-short-sha>
```

例如提交 SHA 为 `0123456789abcdef...` 时：

```text
registry.hd-04.alayanew.com:8443/openclaw/ai-dingo:cu130-runtime-0123456789ab
```

当前不推送 `latest` 等浮动标签，每个镜像都可以追溯到具体 Git Commit。

## 6. 工作流行为

- 配置准备任务超时为 10 分钟。
- 镜像构建任务超时为 240 分钟。
- 同一 Git Ref 的运行会排队，`cancel-in-progress: false`。
- 同一次运行中的多个镜像按 `max-parallel: 1` 串行构建。
- 构建成功后直接推送内部 Registry。
- 当前设置 `provenance: false` 和 `sbom: false`。
- 当前未接入完整的 compliance extract 和 policy gate。
- `keep_buildkit_state` 默认为 `false`，避免 self-hosted Runner 长期积累缓存。

## 7. 常见调整

### 修改镜像仓库

修改 `.github/dingo-images.json` 中的：

```json
{
  "registry": "新的-registry-host:port",
  "namespace": "新的命名空间",
  "images": [
    {
      "framework": "dynamo",
      "repository": "新的镜像仓库名称"
    }
  ]
}
```

`registry` 不能包含 `http://`、`https://` 或路径。修改 Registry 后还需要同步更新
Repository Secrets；若使用私有 CA，需要在 Runner 的 Docker daemon 中安装 CA。

### 修改 Registry 凭据

在 GitHub 仓库设置中维护：

```text
REGISTRY_USERNAME
REGISTRY_PASSWORD
```

不要把凭据直接写入工作流或 JSON 文件。

### 修改 CUDA 版本

不能只修改 JSON。增加 CUDA 版本时需要同步：

1. 在 `container/context.yaml` 的相应配置段下增加对应 CUDA 配置。
2. 让 `container/render.py` 接受该 CUDA 版本。
3. 扩展 `prepare_dingo_image_matrix.py` 的 `SUPPORTED_CUDA_VERSIONS`。
4. 更新 `dingo-images.json` 的 `cuda_version` 和 `tag_prefix`。
5. 实际渲染 Dockerfile并验证基础镜像。

### 增加 arm64 或多架构

当前只支持 `linux/amd64`。扩展架构时需要同步校验脚本、Dockerfile 文件名生成逻辑、
Buildx Builder 能力以及基础镜像的目标架构支持。

### 调整 BuildKit 缓存

磁盘空间充足时可以把：

```json
"keep_buildkit_state": true
```

开启后定期检查：

```bash
docker buildx du
docker system df
df -h /mnt/nvme0/docker
```

### 减少不必要的构建

如果不要求每个提交都有镜像，可以给 push 事件增加 `paths`：

```yaml
paths:
  - "dingo/**"
  - "lib/**"
  - "container/**"
  - "pyproject.toml"
  - ".github/dingo-images.json"
  - ".github/scripts/prepare_dingo_image_matrix.py"
  - ".github/workflows/dingo-router-ci.yml"
```

### 恢复交付合规能力

正式交付前应根据要求评估恢复 SBOM、Build Provenance、License 合规提取、
Compliance Policy Gate、源码归档和 Attribution 产物。

## 8. Runner 环境要求

Runner 机器至少需要：

- GitHub Actions Runner。
- Docker Engine 和 Docker Buildx。
- Python 3，以及 `jinja2`、`pyyaml`。
- Runner 账户有权访问 Docker daemon。
- 能访问 GitHub、Dockerfile 使用的软件源、基础镜像仓库和内部 Registry。
- 有足够空间完成 Dynamo 多阶段构建。

检查命令：

```bash
systemctl status actions.runner.zhaoxianhua-dynamo.jn123.service
journalctl -u actions.runner.zhaoxianhua-dynamo.jn123.service
docker version
docker buildx version
docker buildx ls
docker buildx du
python3 -c "import jinja2, yaml"
```

## 9. 安全注意事项

- Registry 凭据只保存在 GitHub Repository Secrets。
- 限制能够修改 `.github/workflows/` 的人员。
- 建议保护 `DingoRouter-base` 分支。
- self-hosted Runner 会直接执行分支中的工作流代码，相关提交需要严格审核。
- 私有 CA 应配置在 Docker daemon 层，不要关闭 TLS 校验。

## 10. 当前限制

- 默认只构建 standalone Dynamo，并保留手动构建其他已配置 Runtime 的能力。
- 只支持 CUDA 13.0 和 `linux/amd64`。
- 每次 Push 都会构建，没有路径过滤。
- 不推送浮动标签。
- 不生成 SBOM 和 Provenance。
- 未接入完整 compliance gate。
- 不自动清理宿主机已有镜像和其他 Builder 缓存。
