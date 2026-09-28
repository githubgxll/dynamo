<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 模型阶段计时 Schema

任务 API 的 `metrics` 包含 Gateway/Worker 通用耗时，模型专用计时放在
`metrics.model_execution`。诊断接口包含相同对象。没有可用模型计时时整个对象省略。
`model_execution.schema` 定义模型计时语义；`diagnostics.schema_version` 定义诊断对象的
结构，两者独立。`diagnostics.stage_durations` 主要记录 Worker 输出处理细项，当前没有
Gateway 处理细项；完整返回范围见[接口文档的任务诊断](video-gateway-api.md#任务诊断)。

```json
{
  "schema": "minimax_h3.v1",
  "unit": "seconds",
  "stages": {
    "encode_prompt": 0.12,
    "diffuse": 22.5,
    "decode": 1.3,
    "video_decode": 1.2,
    "audio_decode": 0.08
  }
}
```

## 源码定义与导出

唯一契约来源是 `dingo/common/video_timing_schemas.py` 的 `SCHEMAS` 注册表。
`TimingSchema` 定义版本、单位、说明及阶段集合；`StageDefinition` 定义阶段说明和可选父阶段。
Worker 的引擎字段映射单独位于适配模块；当前 H3 映射是
`dingo/vllm/omni/minimax_h3_timings.py` 的 `NATIVE_STAGE_NAMES`。

从仓库根目录导出可供客户端校验或类型生成的 JSON Schema：

```bash
python -m dingo.common.video_timing_schemas minimax_h3.v1 > /tmp/minimax_h3.v1.schema.json
```

导出使用 JSON Schema draft 2020-12；子阶段通过 `x-parent-stage` 标识。
它是生产方的严格契约；客户端遇到未来新增字段应忽略或保留，而不是使整条任务查询失败。

## 版本及客户端规则

- 外层结构固定为 `schema`、`unit`、`stages`。当前单位固定为秒。
- `schema` 是阶段语义版本，不是 Worker 镜像版本。不同引擎只有在计时边界相同的情况下
  才能使用相同 schema，否则应定义独立 schema。
- 已有阶段的含义、单位、名称和父子关系不得原地修改；此类变更新增 `.v2`。
- 同版本可增加可选阶段；所有阶段都允许缺失，零表示有效零值，不表示未采集。
- `stages` 只含有限、非负数值。未知 schema、单位不符或没有有效阶段时，当前 Gateway
  省略整个 `model_execution`；观测数据不兼容不应导致视频任务失败。
- 模型阶段只描述当前/最终 attempt；不是多个重试的累计值。
- 客户端按 `schema` 选择中文标签等展示配置；未知 schema 可按动态字典展示或隐藏详情。
  普通业务逻辑只依赖通用 `metrics`，新增模型不要求它同步升级。
- 父子阶段可能重叠，不能将所有阶段相加作为执行总时间。

## 已注册 Schema

目前仅注册 `minimax_h3.v1`，覆盖 H3 FL2VA/Ref2VA request 模式的原生 profiler 计时。
它表示引擎返回的 profiler 样本，不保证是所有 GPU rank 的最大值或总和。
step 模式交错执行的逐请求隔离尚未验收，不能用此契约宣称该场景具有准确归因。

| 阶段 | 定义 | 父阶段 |
|---|---|---|
| `encode_prompt` | 提示词/条件编码，可能包含视觉输入处理 | — |
| `diffuse` | 去噪阶段 | — |
| `decode` | 输出视频/音频解码总阶段 | — |
| `video_decode` | Video VAE 解码 | `decode` |
| `audio_decode` | Audio VAE 解码 | `decode` |
| `reference_video_prepare` | 参考视频准备方法，包括在该处执行的探测、校验及转码 | — |
| `reference_visual_encode` | 参考图片与视频的视觉/VAE条件编码合计 | — |
| `reference_audio_encode` | 视频内音轨及独立参考音频的条件编码合计 | — |

原生 `_prepare_reference_videos` 仅 rank 0 做实际工作，其他 rank 可快速返回；该数值不是
跨 rank 汇总。方法被调用但没有实际输入时也可能有近零值，不能仅凭键是否存在推断输入类型。
需要开启 `--enable-diffusion-pipeline-profiler` 才能取得这些模型阶段；它与
Gateway 的 `DINGO_VIDEO_MEDIA_TIMING` 日志开关独立。

## 新增模型或引擎

1. 确定计时边界、原始单位、执行模式和多 rank/多请求归属。
2. 在 `SCHEMAS` 中增加独立版本定义，或确认可以复用已有定义。
3. 在后端适配代码中转换原始字段与单位，输出 `model_execution` 对象。
4. 通过统一结果协议传递；Gateway只校验、持久化和返回对象，无需修改任务序列化字段列表。
5. 补充 schema 映射、缺失值、版本不兼容以及任务交接/查询测试，并更新本页。

源码变更后需要将定义随 Worker/Gateway版本发布。建议先更新 Gateway使其认识新 schema，
再更新 Worker；旧 Gateway会省略它不认识的模型计时。既有任务不会自动补采模型计时。
