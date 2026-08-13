# DingoRouter Python 单元测试 — 环境依赖与准备记录

记录在 `172.30.14.203`（root/jn@123）的 `/root/gjn/dynamo`（git HEAD `9645f83f6`）执行 `run_all_tests.sh` 中 Python 单元测试部分（命令 1-3）所需的全部环境依赖、下载安装操作，以及本轮结果。配套脚本：同目录 `env-setup-py.sh`（环境准备）、`run-python-unit-tests.sh`（串行跑 3 组测试）。

## 测试范围

仅 `run_all_tests.sh` 的第一部分"Python 单元测试"3 条命令：

| 序号 | 测试目标 | 目录 |
|---|---|---|
| 1 | vllm_unit | `dingo/vllm/tests/` |
| 2 | frontend_unit | `dingo/frontend/tests/` |
| 3 | sglang_unit | `dingo/sglang/tests/` |

集成测试（命令 4-8）、Rust 测试、代码一致性检查不在本脚本范围。

## 环境基线（执行前）

| 项 | 状态 | 说明 |
|---|---|---|
| 源码 | `/root/gjn/dynamo`，HEAD `9645f83f6` | 已存在 |
| `/usr/bin/python3` | 3.6.8（系统） | **过旧，不可用** |
| `/usr/bin/python3.11` | 3.11.11，无 pip/pytest/dynamo | **可用，需 bootstrap** |
| Rust 工具链 | 1.93.1（rustup） | 已就绪（与 Rust 测试同环境） |
| gcc-12 libstdc++ | 已 ldconfig | 已就绪（Rust binding 构建复用） |

## 下载安装 / 配置清单

### 1. python3.11 bootstrap pip

- `/usr/bin/python3.11` 自带 `ensurepip`（pip 22.3.1），运行 `python3.11 -m ensurepip` 完成 bootstrap。
- 升级用到 pip 22.3.1 即可满足。

### 2. 安装 pytest 测试插件

- `pyproject.toml` 的 `[tool.pytest.ini_options]` 配置了 `filterwarnings` 引用 `pytest_benchmark.logger.PytestBenchmarkWarning`，未装 `pytest-benchmark` 会在收集阶段崩溃（`ModuleNotFoundError: No module named 'pytest_benchmark'`）。
- `asyncio_mode = "auto"` 需要 `pytest-asyncio`。
- 安装：`pytest pytest-timeout pytest-benchmark pytest-asyncio pytest-xdist pytest-rerunfailures`。

### 3. 构建 ai-dingo-runtime（Rust Python binding）

- `ai-dingo`（pyproject `name = "ai-dingo"`）依赖 `ai-dingo-runtime==1.3.0`，该包**不在 PyPI**（`No matching distribution found`），需从源码 `lib/bindings/python`（pyproject `name = "ai-dingo-runtime"`）构建。
- 构建：`python3.11 -m pip install ./lib/bindings/python/`（pip 调 maturin 后端编译 pyo3 binding）。
- 依赖：Rust 工具链 1.93.1 + gcc-12 libstdc++（`LIBRARY_PATH`/`LD_LIBRARY_PATH`）+ 保留 `tokio_unstable`（`unset RUSTFLAGS`，复用 Rust 测试环境）。
- 首次构建约 5-8 分钟，产出 `dynamo._core` 扩展模块。
- 安装依赖包：pydantic 2.13 / pydantic-core 2.46 / uvloop 0.22 等（pip 自动拉取）。

### 4. editable 安装 ai-dingo

- `python3.11 -m pip install -e .` 安装 `ai-dingo`（含 dingo 包），拉取 aiohttp / transformers / kubernetes / pyzmq / msgspec 等依赖。
- 完成后 `import dingo` / `import dynamo` 可用。

## 关键差异：sglang_unit 的 2 个 deselect

`dingo/sglang/tests/test_fpm_contract.py` 的 2 个用例直接 `from sglang.srt.observability.forward_pass_metrics import ...`，需要完整 sglang 引擎（`sglang==0.5.17`，PyPI 无此版本且需 GPU 栈）。当前节点无 GPU、未装 sglang，这 2 个用例与 vllm_unit 的 `test_vllm_engine_all` 同理被 deselect：

```
--deselect dingo/sglang/tests/test_fpm_contract.py::test_sglang_fpm_decodes_with_dynamo_schema
--deselect dingo/sglang/tests/test_fpm_contract.py::test_sglang_fpm_field_order_matches_dynamo
```

> 注：`run_all_tests.sh` 原文未 deselect 这 2 个，说明原 pod 装了 sglang。本环境无 sglang，按需 deselect 以等价 vllm 的处理方式。

## 执行环境变量（run-python-unit-tests.sh 已固化）

```bash
export PATH="/root/.cargo/bin:/usr/local/bin:/usr/local/gcc-12/bin:${PATH}"
export NO_PROXY="127.0.0.1,localhost"      # 避免 HF 联网/mockito 被代理
export no_proxy="127.0.0.1,localhost"
export LD_LIBRARY_PATH="/usr/local/gcc-12/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/local/gcc-12/lib64:${LIBRARY_PATH:-}"
export CC="/usr/local/gcc-12/bin/gcc"
export CXX="/usr/local/gcc-12/bin/g++"
unset RUSTFLAGS
```

`HF_HUB_OFFLINE=1` 用于 vllm_unit 命令，避免 HuggingFace 联网。

## 复现步骤

```bash
cd /root/gjn/dynamo
bash tests/router/env-setup-py.sh            # 环境准备（幂等，首次构建 binding 约 8 分钟）
bash tests/router/run-python-unit-tests.sh   # 串行跑 3 组，日志在 /tmp/dingoRouter-py-unit-tests-<时间戳>/
```

## 本轮结果（20260812）

3 组 Python 单元测试全部通过（exit 0）。

| 序号 | 测试目标 | 结果 | passed | failed | deselected | 耗时 |
|---|---|---|---|---|---|---|
| 1 | vllm_unit | 通过 | 42 | 0 | 0 | 0.3s |
| 2 | frontend_unit | 通过 | 19 | 0 | 0 | 0.1s |
| 3 | sglang_unit | 通过 | 4 | 0 | 2 | 0.1s |

### 说明

- vllm_unit 的 `--deselect` 列表与 `run_all_tests.sh` 完全一致（排除需 vllm 引擎的 `test_vllm_engine_all` 及 jinja/connector 等环境相关用例），实跑 42 个。
- frontend_unit 无 deselect，实跑 19 个全过。
- sglang_unit 在原 `--deselect` 基础上额外 deselect 2 个 fpm_contract 用例（缺 sglang 引擎），实跑 4 个。
- 三组均在秒级完成（纯单元测试，无 GPU/无网络）。

## 脚本说明

| 脚本 | 作用 |
|---|---|
| `env-setup-py.sh` | Python 测试环境准备（ensurepip → pytest 插件 → maturin 构建 ai-dingo-runtime → editable 安装 ai-dingo），幂等。 |
| `run-python-unit-tests.sh` | 3 组 Python 单元测试串行执行，用 `/usr/bin/python3.11`，汇总到 `/tmp/dingoRouter-py-unit-tests-<时间戳>/SUMMARY.txt`。 |
