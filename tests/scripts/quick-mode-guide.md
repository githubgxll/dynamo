# DingoRouter 测试脚本 QUICK_MODE 模式说明

`run-all-tests-oneclick.sh` 支持三档 `QUICK_MODE` 开关，按需选择测试范围。

## 用法

```bash
# 冒烟模式（约 3.5 分钟，二次跑）
QUICK_MODE=smoke bash tests/router/run-all-tests-oneclick.sh

# 中速模式（约 6.5 分钟，二次跑）
QUICK_MODE=standard bash tests/router/run-all-tests-oneclick.sh

# 全量模式（约 11 分钟，二次跑，默认）
QUICK_MODE=full bash tests/router/run-all-tests-oneclick.sh

# 可与 SKIP_MODEL_DOWNLOAD 叠加（full 模式下跳过模型下载+依赖模型的集成测试）
SKIP_MODEL_DOWNLOAD=1 QUICK_MODE=full bash tests/router/run-all-tests-oneclick.sh
```

> 首次全新环境耗时更长（因 Rust 全量编译依赖树）：smoke 约 8-10 分钟、standard 约 25-35 分钟、full 约 35-50 分钟。

## 三档覆盖范围

### smoke（冒烟模式）

只跑核心 Rust 包单元测试 + Python 单元测试，跳过所有集成测试、模型下载。

| 步骤 | 执行内容 | 跳过内容 |
|---|---|---|
| 环境准备 | env-setup-py-integ.sh（链式级联，幂等） | — |
| Rust 单元测试 | **5 个核心包**（见下表） | 其余 18 个 Rust 包 |
| Rust 集成测试 | ❌ 跳过 | 全部 5 个 crate |
| Python 单元测试 | ✅ 全部 3 组 | — |
| Python 集成测试 | ❌ 跳过 | 全部 5 组 |
| 模型下载 | ❌ 跳过（standard/full 之外均不下载） | 1.5G Qwen3-0.6B |

**smoke 的 5 个核心 Rust 包**：

| 包名 | 测试数 | 覆盖的代码路径 |
|---|---|---|
| `dynamo-backend-common` | 132 | 后端通用框架（worker 生命周期、拓扑、模型注册） |
| `dynamo-kv-router` | 587 | KV 感知路由（chooser、scheduling、worker 选择） |
| `dynamo-runtime` | 439 | 运行时（pipeline、transport、discovery、task tracker） |
| `dynamo-llm` | 1507 | LLM 层（session affinity、prefill/decode router、protocols） |
| `dynamo-mocker` | 495 | Mock 引擎（模拟 worker、KV 事件、测试基建） |
| **合计** | **3160** | 覆盖核心路由/调度/运行时约 80% 代码路径 |

加上 Python 单元测试 65 个，smoke 模式共验证 **3225 个测试**。

---

### standard（中速模式）

跑全部 23 个 Rust 包单元测试 + Python 单元测试，跳过所有集成测试和模型下载。

| 步骤 | 执行内容 | 跳过内容 |
|---|---|---|
| 环境准备 | env-setup-py-integ.sh（链式级联，幂等） | — |
| Rust 单元测试 | ✅ **全部 23 包** | — |
| Rust 集成测试 | ❌ 跳过 | 全部 5 个 crate |
| Python 单元测试 | ✅ 全部 3 组 | — |
| Python 集成测试 | ❌ 跳过 | 全部 5 组 |
| 模型下载 | ❌ 跳过（自动设 `SKIP_MODEL_DOWNLOAD=1`） | 1.5G Qwen3-0.6B |

**standard 的 23 个 Rust 包**：

| 序号 | 包名 | 测试数 |
|---|---|---|
| 1 | dynamo-backend-common | 132 |
| 2 | dynamo-bench | 22 |
| 3 | dynamo-codegen | 0 |
| 4 | dynamo-data-gen | 43 |
| 5 | dynamo-ext-proc | 6 |
| 6 | dynamo-kv-hashing | 0 |
| 7 | dynamo-kv-router | 587 |
| 8 | dynamo-llm | 1507 |
| 9 | dynamo-memory | 147 |
| 10 | dynamo-mocker | 495 |
| 11 | dynamo-mocker-backend | 19 |
| 12 | dynamo-rl | 10 |
| 13 | dynamo-runtime | 439 |
| 14 | dynamo-tokens | 53 |
| 15 | dynamo-vllm-rs-backend | 0 |
| 16 | kvbm-common | 0 |
| 17 | kvbm-config | 63 |
| 18 | kvbm-consolidator | 28 |
| 19 | kvbm-engine | 126 |
| 20 | kvbm-kernels | 0 |
| 21 | kvbm-logical | 504 |
| 22 | kvbm-physical | 13 |
| 23 | libdynamo_llm | 3 |
| | **合计** | **4191** |

加上 Python 单元测试 65 个，standard 模式共验证 **4256 个测试**。

---

### full（全量模式，默认）

跑全部测试：Rust 单元 + Rust 集成 + Python 单元 + Python 集成。

| 步骤 | 执行内容 | 跳过内容 |
|---|---|---|
| 环境准备 | env-setup-py-integ.sh + env-setup-integ.sh（含 e2e fixture 重生成） | — |
| Rust 单元测试 | ✅ 全部 23 包（4191 passed） | — |
| Rust 集成测试 | ✅ 全部 5 crate（51 passed, 2 ignored） | — |
| Python 单元测试 | ✅ 全部 3 组（65 passed） | — |
| Python 集成测试 | ✅ 全部 5 组（73 passed, 24 skipped, 54 deselected） | — |
| 模型下载 | ✅ 下载 Qwen3-0.6B（1.5G） | — |
| | **合计** | **4380 passed** |

**full 模式的 Rust 集成测试（5 crate）**：

| crate | 测试文件 | passed |
|---|---|---|
| dynamo-kv-hashing | request_hashing, serde_roundtrip | 20 |
| dynamo-kv-router | standalone_indexer_http | 3 |
| dynamo-runtime | bidirectional_e2e, lifecycle, pipeline, pool, soak | 8 (1 ignored) |
| kvbm-consolidator | chaos_properties, dedup, e2e, kvbm_bridge, lifecycle, output_contract, zmq_ingress | 18 (1 ignored) |
| kvbm-kernels | kernel_roundtrip, memcpy_batch, stub_build | 2 |

**full 模式的 Python 集成测试（5 组）**：

| 序号 | 测试目标 | passed | skipped | deselected |
|---|---|---|---|---|
| 04 | integration_main（tests/ 主组） | 48 | 17 | 30 |
| 05 | slot_tracker_standalone | 0 | 0 | 3 |
| 06 | slot_tracker_e2e | 4 | 0 | 0 |
| 07 | mocker_output_replay | 1 | 0 | 0 |
| 08 | router_e2e_mockers | 20 | 7 | 24 |

---

## 三档对比总览

| 维度 | smoke | standard | full |
|---|---|---|---|
| **二次跑耗时** | ~3.5 分钟 | ~6.5 分钟 | ~11 分钟 |
| **首次跑耗时** | ~8-10 分钟 | ~25-35 分钟 | ~35-50 分钟 |
| Rust 单元测试 | 5 核心包（3160） | 23 包全量（4191） | 23 包全量（4191） |
| Rust 集成测试 | ❌ | ❌ | ✅（51） |
| Python 单元测试 | ✅（65） | ✅（65） | ✅（65） |
| Python 集成测试 | ❌ | ❌ | ✅（73） |
| 模型下载 | ❌ | ❌ | ✅（1.5G） |
| **总 passed** | 3225 | 4256 | 4380 |
| **磁盘占用** | 无额外 | 无额外 | +1.5G 模型 |
| **外网需求** | 仅依赖包 | 仅依赖包 | +模型下载 |
| **适用场景** | 改代码后快速自检 | commit 前验证 | CI / 完整回归 |

## 环境变量组合

| 变量 | 默认 | 作用 |
|---|---|---|
| `QUICK_MODE` | `full` | 选择测试范围：smoke / standard / full |
| `SKIP_MODEL_DOWNLOAD` | `0`（仅 full 模式生效） | `=1` 时跳过模型下载 + Python 集成测试 06/07/08 |
| `REBUILD_WITH_SLOT_TRACKER` | `0` | `=1` 时重新构建含 slot-tracker 的 binding |
| `QUICK_PACKS` | 未设置 | 直接指定 Rust 包名（逗号分隔），onelick 内部使用 |

> `standard` 模式自动设 `SKIP_MODEL_DOWNLOAD=1`，无需手动指定。
> `SKIP_MODEL_DOWNLOAD=1` 可单独与 `full` 叠加，跑 Rust 集成 + Python 单元 + Python 集成 04/05，但跳过 06/07/08 和模型下载。
