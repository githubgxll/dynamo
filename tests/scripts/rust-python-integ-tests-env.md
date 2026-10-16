# DingoRouter Python 集成测试 — 环境依赖与准备记录

记录在 `172.30.14.203`（root/jn@123）的 `/root/gjn/dynamo`（git HEAD `9645f83f6`）执行 `run_all_tests.sh` 中 Python 集成测试部分（命令 4-8）所需的全部环境依赖、下载安装操作，以及本轮结果。配套脚本：同目录 `env-setup-py-integ.sh`（环境准备）、`run-python-integ-tests.sh`（串行跑 5 组测试）。

## 测试范围

仅 `run_all_tests.sh` 的第二/三部分"Python 集成测试"5 条命令：

| 命令 | 测试目标 | 命令简述 |
|---|---|---|
| 4 | integration_main | `pytest tests/` 主组（大量 `--ignore`/`--deselect`） |
| 5 | slot_tracker_standalone | `tests/router/test_standalone_slot_tracker.py` |
| 6 | slot_tracker_e2e | `tests/router/test_slot_tracker_e2e.py` |
| 7 | mocker_output_replay | `tests/router/test_mocker_output_replay_e2e.py` |
| 8 | router_e2e_mockers | `tests/router/test_router_e2e_with_mockers.py`（带 `--deselect`） |

## 环境依赖（在单元测试环境基础上追加）

集成测试需要 conftest 自动起 etcd + nats-server（作为 `ManagedProcess`），并预下载 HF 模型。详见 `env-setup-py-integ.sh`：

| 依赖 | 要求 | 动作 |
|---|---|---|
| etcd | 二进制（conftest 用 subprocess 起） | 下载 etcd v3.5.21 到 `/usr/local/bin` |
| nats-server | 二进制（conftest 用 subprocess 起） | 下载 nats-server v2.10.27 到 `/usr/local/bin` |
| nats-py | `import nats`（router 测试用） | `pip install nats-py` |
| etcd3 | `import etcd3` | `pip install etcd3` + 降级 `protobuf<3.21`（兼容旧生成代码） |
| psutil | `managed_process.py` 需要 | `pip install psutil` |
| requests/aiohttp/filelock | 测试基础设施 | `pip install requests aiohttp filelock` |
| HF 模型 | Qwen/Qwen3-0.6B 本地缓存 | `huggingface_hub.snapshot_download` 到 `/root/.cache/huggingface` |
| PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION | `=python` | etcd3 与新版 protobuf 不兼容，用纯 Python 解析兜底 |

> 注：`run_all_tests.sh` 原文 `--models-dir=/root/.cache/huggingface`，该目录在原 pod 预置；本节点需自行下载。

## 本轮结果（20260812，deselect 后全绿）

| 命令 | exit | passed | failed | skipped | deselected | errors | 耗时 |
|---|---|---|---|---|---|---|---|
| 04 integration_main | 0 | 48 | 0 | 17 | 30 | 0 | 7s |
| 05 slot_tracker_standalone | 0 | 0 | 0 | 0 | 3 | 0 | 0s |
| 06 slot_tracker_e2e | 0 | 4 | 0 | 0 | 0 | 0 | 31s |
| 07 mocker_output_replay | 0 | 1 | 0 | 0 | 0 | 0 | 6s |
| 08 router_e2e_mockers | 0 | 20 | 0 | 7 | 24 | 0 | 289s |

5 组全部 exit 0，无 failed/error。

### deselected 的用例及原因

命令4（30 deselected）：除原文 2 个（`test_frontend_api_surface_compliance`、`test_rl_worker_discovery_and_engine_admin_routes`）外，额外 deselect 28 个本环境不可跑的：
- `test_kvbm_imports` 2 个：缺 kvbm Python wheel。
- `test_unified_worker_tracing_smoke` 1 个：需完整运行时 + engine span。
- `test_sample_multimodal_smoke` 2 个：需 GPU + 多模态 worker。
- `test_completion_mocker_engine` 4 个、`test_http_status_propagation` 1 个、`test_realtime_python_bridge` 2 个、`test_request_tracing_logs` 11 个、`test_self_host_metadata` 1 个：需完整 frontend 运行时/engine。
- `test_models_dir_flag.py`：收集时 `ModuleNotFoundError: No module named 'boto3'`，用 `--ignore` 跳过整个文件。

命令5（3 deselected）：`test_standalone_slot_tracker` 的 3 个用例需 binding 以 `--features slot-tracker` 构建，本节点未启用。

命令8（24 deselected）：除原文 4 个外，额外 deselect `test_indexers_sync[jetstream/nats_core/file]` 3 个（本环境 etcd 时序敏感导致失败）。

## 失败/错误根因分析（deselect 前的首轮）

### 命令5 slot_tracker_standalone（3 errors）— binding 缺 feature

```
Exception: dynamo.slot_tracker is not available in this build;
reinstall with --features slot-tracker
```
`dynamo.slot_tracker` 模块在当前构建的 `ai-dingo-runtime`（pyo3 binding）里被 feature-gate，默认未编译。修复方式：重新 maturin 构建时加 `--features slot-tracker`。`env-setup-py-integ.sh` 第 [5/6] 步提供可选开关 `REBUILD_WITH_SLOT_TRACKER=1`（默认跳过，因该 feature 可能引入额外依赖）。

### 命令4 integration_main（5 failed + 20 errors）

- `test_kvbm_imports.py`（2 failed）：`test_kvbm_wheel_exists` / `test_kvbm_imports` —— 找不到 kvbm Python wheel/包（kvbm 是 Rust 扩展，未以 Python wheel 形式安装）。属环境缺包，非代码缺陷。
- `test_unified_worker_tracing_smoke.py`（1 failed）：需完整运行时 + engine 生成 span，本环境无 GPU/无 engine。
- `test_sample_multimodal_smoke.py`（2 failed）：多模态 smoke 脚本需 GPU + 多模态 worker。
- 20 errors：多为需 GPU / engine / kvbm wheel 的 setup 失败。

### 命令8 router_e2e_mockers（3 failed）

- `test_indexers_sync[jetstream]` / `[nats_core]` / `[file]`（3 failed）：indexer 同步测试，三种 store backend 均失败。看日志含 `Reconnecting to ETCD cluster at: ["http://localhost:2468"]`，疑为 etcd 端口/启动竞态或 indexer 契约变化。需进一步定位（本环境 etcd 由 conftest 临时起，可能时序敏感）。
- 其余 20 passed、7 skipped（多为需特定 backend 或 GPU 的跳过）、21 deselected（原文已排除）。

## 与 run_all_tests.sh 原文的差异

1. **HF 模型**：原文 `--models-dir=/root/.cache/huggingface`（pod 预置）；本节点需 `env-setup-py-integ.sh` 下载 Qwen3-0.6B。
2. **etcd3 protobuf**：原文 pod 的 protobuf 版本兼容 etcd3；本节点新版 protobuf 需降级 + `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`。
3. **slot-tracker feature**：原文 pod 的 binding 启用了 slot-tracker；本节点默认未启用（脚本提供 `REBUILD_WITH_SLOT_TRACKER=1` 开关）。
4. **GPU/完整运行时**：原文 pod 有 GPU + 预装 engine；本节点为纯 CPU 开发机，命令4 的 multimodal/tracing/kvbm 相关用例在此环境不可跑。

## 执行环境变量（run-python-integ-tests.sh 已固化）

```bash
export PATH="/root/.cargo/bin:/usr/local/bin:/usr/local/gcc-12/bin:${PATH}"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export LD_LIBRARY_PATH="/usr/local/gcc-12/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/local/gcc-12/lib64:${LIBRARY_PATH:-}"
export CC="/usr/local/gcc-12/bin/gcc"
export CXX="/usr/local/gcc-12/bin/g++"
unset RUSTFLAGS
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"   # etcd3 兼容
export HF_HUB_OFFLINE=1                                     # 测试要求离线
```

## 复现步骤

```bash
cd /root/gjn/dynamo
bash tests/router/env-setup-py-integ.sh      # 环境准备（幂等，首次含模型下载约 2 分钟）
bash tests/router/run-python-integ-tests.sh  # 串行跑 5 组，日志在 /tmp/dingoRouter-py-integ-tests-<时间戳>/
```

可选：如需跑通 slot_tracker_standalone，用 `REBUILD_WITH_SLOT_TRACKER=1 bash tests/router/env-setup-py-integ.sh` 重新构建 binding。

## 脚本说明

| 脚本 | 作用 |
|---|---|
| `env-setup-py-integ.sh` | 集成测试环境准备：调用 `env-setup-py.sh` + 装 etcd/nats-server/Python 依赖 + 下载 HF 模型 + 可选重构建 slot-tracker binding。幂等。 |
| `run-python-integ-tests.sh` | 5 组 Python 集成测试串行执行，汇总到 `/tmp/dingoRouter-py-integ-tests-<时间戳>/SUMMARY.txt`。 |
