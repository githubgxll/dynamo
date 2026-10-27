# 2026-10-27：Torch 来源指纹跨 Python 版本误报修复

结论：这次构建停在 **HSDP 补丁应用前的 Torch 来源检查**。本机 Python 3.14 与镜像 Python 3.12 对同一 AST 的默认文本表示不同，造成旧指纹误报。使用同一份已锁定的官方源码，本机 Python 3.12 算出的 `Module._apply` 指纹与 CI 报错中的实际值完全相同。本次修复指纹算法和配套验证，继续使用固定成功父镜像；不升级 Torch、Omni、Dynamo 或其他依赖，也不修改 HSDP 迁移 helper。

**修正本次构建检查不等于 CPU/GPU 卸载已经验证成功。** 新配方仍需在 Linux Runner 完成实际镜像构建、CPU 验证及发布，随后执行八卡迁移和固定输入媒体验收。

## 已观察到的失败与根因

失败日志中，父 C14 镜像的依赖、CPU 媒体、服务导入、C14 回归和六组 RMSNorm 数值检查已经通过。下一层 `runtime.py apply` 调用 `torch_contract.verify()`，在检查第一个原生 Torch 方法时停止；新卸载补丁尚未应用，更未执行实际 CPU↔GPU 参数迁移。

错误包含：

```text
Native Torch method differs: torch.nn.modules.module.Module._apply
expected_ast=e1b50250f5d3b8e5f8b08cac6f97a03b2b2794d4ab8b98eb51b8205e284c71e3
actual_ast=b834c61006395b0cea8dc86e0246c653a30d127db722b8941764505d0ea22506
torch_git=cf30153c4c131c8164ee7798e5022d810682e2cb
```

旧算法直接对 `ast.dump(node, include_attributes=False)` 的输出计算 SHA256。Python 3.14 默认省略空列表字段，Python 3.12 则保留。例如同一函数在 3.12 输出中的 `posonlyargs=[]`、`kwonlyargs=[]`、`decorator_list=[]` 等，默认不再出现在 3.14 的输出中。源码行为未改变，指纹输入却发生变化。

下表由**同一份已锁定的 PyTorch v2.13.0 官方源码**在本机 Python 3.14.7 和 Python 3.12.14 分别复算。镜像使用 Python 3.12.3；第一项的 3.12 复算结果精确匹配 CI 的 `actual_ast`。CI 在第一项即停止，后两项是本地复现结果，不能表述为已经在失败镜像中通过验证。

| 方法 | 旧算法：Python 3.14.7 | 旧算法：Python 3.12.14 |
|---|---|---|
| `Module._apply` | `e1b50250f5d3b8e5f8b08cac6f97a03b2b2794d4ab8b98eb51b8205e284c71e3` | `b834c61006395b0cea8dc86e0246c653a30d127db722b8941764505d0ea22506` |
| `FSDPModule._apply` | `d2445abe09c170ddb9a90a774dcb5da5ec2c1213ceea15a1c07387d9c7e7e462` | `455c74a8f73b5da8c8433e3ad7f24d9ad49e96cdff6edf3f812d3fa128c49a3a` |
| `FSDPParam.reset_sharded_param` | `46b9e1b69eacb440fb3c8b2a1f019844617d644c0715b3c565e0c4ac95d24059` | `2a4a7b66042ec447041f4c851fd1a2f0013fb1daff836b9d73b9562be6072f17` |

这说明旧来源检查不能跨本地和镜像解释器直接比较；并不能仅凭这一结论跳过其余实际父源码校验。

进一步通过 GitHub 读取日志中的精确提交 `pytorch/pytorch@cf30153c4c131c8164ee7798e5022d810682e2cb`，`torch/nn/modules/module.py`、`torch/distributed/fsdp/_fully_shard/_fully_shard.py`、`_fsdp_param.py` 三个完整文件的 SHA256 均与原 v2.13.0 参考相同，三个新方法指纹也与当前锁一致。对照记录保存于本地 `.h3-experimental-evidence/hsdp/local/torch-exact-source-cf30153/comparison.json`。这是精确上游提交的来源核对，镜像实际安装文件仍由构建检查。

## 本次修复

`torch-contract.json` 升级为 `h3-hsdp-torch-native-contract/2`，指纹算法标记为 `python-ast-fields-json/1`。指纹输入使用规范 JSON，明确包含 AST 节点类型、全部字段、空列表及 `None`，并区分常量类型；排除行号等源码位置属性。源码格式和注释不会导致误报，条件、调用、参数或常量等 AST 改动仍会产生不同指纹。未知结构或不支持的契约必须停止，不能静默忽略。

三项方法一并更新指纹和来源材料，并以已审核方法摘录进行离线回归。来源记录保留完整官方源文件 SHA256 和摘录 SHA256，便于区分算法变化与源码变化。新测试覆盖跨解释器一致性以及真实 AST 改动必须拒绝的情况；它们属于源码指纹验证，不是 Torch 运算或 GPU 测试。

本次保持以下内容：

- 固定父镜像 `registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:87365073cbdf95fa17ca352a24a93cb74fde144a47c96135e91011d1a03df72e`。
- vLLM 0.31.0、vLLM-Omni 0.31.0rc1、PyTorch 2.13.0+cu130、Cache-DiT 1.5.0 与现有 video-gateway。
- 已准备的 `hsdp_migration.py` 迁移实现和 Omni 定向补丁行为；原始父层来源、依赖和文件保留检查继续执行。
- 不重装依赖、不下载权重、不更换并行方案、不停用 HSDP，也不向原生 Torch 注入补丁。

## 重新提交与验收

已完成本地验证：

- Python 3.12.14 与 3.14.7 各运行 24 项 HSDP 配方及指纹测试，全部通过。三个官方摘录在两种解释器下都匹配同一组锁；方法内容变化仍被拒绝。
- `01-check.cmd` 完整通过：304 项回归中实际执行 288 项，16 项因本机没有 Torch 跳过；19 个 Python 文件及 4 段工作流脚本检查通过。证据目录为 `.h3-experimental-evidence/hsdp/local/checks-20261027T110538.508440Z/`。
- 受限本机环境不能启动 Git Bash，曾在工作流语法检查报 `NtCreateDirectoryObject ... 0xC0000022`；使用正常本机权限重跑同一只读检查入口后通过，没有删掉或放宽语法检查。Python 3.12 的定向测试独立通过；其本地环境没有 PyYAML，因此完整交付入口使用既定 Python 3.14 工具环境。

此后只补充本文证据记录并重新封存，代码未再变更。上述结果不是 Linux 镜像构建、Torch 运算或 GPU 验收结果；本次未自动提交、推送或发布。

本次包含新配方，使用 `02-submit-build.cmd` 提交后构建，不能只用 `03-run-build.cmd` 重跑旧提交：

```bat
cd /d D:\AI\dynamo-h3-experimental031
call delivery\h3-hsdp-offload\02-submit-build.cmd
```

`02` 会先执行本地检查和封存核验，再提交、推送并触发构建。沿用 `Dingorouter-H3-vllm-omni-jirx` 分支。`03` 仅用于相同配方已上传后，确需重新执行同一配方的情况。

远端仍须从实际父镜像导出来源，验证全部三个原生 Torch 方法，完成补丁应用和最终镜像 CPU 检查；发布后使用完整 digest 和对应发布证据。若新指纹仍不匹配，应保留 `parent-source/` 对照实际源码，不应扩大白名单或跳过校验。

真实八卡 HSDP 参数往返、C02 冷/热/连续生成、显存峰值、媒体质量及 Ref7 验收继续按 [README](README.md) 的顺序执行。旧失败 Run 和日志保持不变，后续使用新候选镜像、新 Pod 与新 RUN。构建与 CPU 通过均不能推定 GPU 卸载有效或显存收益成立。
