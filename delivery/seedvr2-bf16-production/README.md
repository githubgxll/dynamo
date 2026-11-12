# SeedVR2 BF16 镜像构筑与交接

本配方从已接受的26d H3/SR共驻镜像薄派生，只安装SeedVR2 DiT-only BF16 helper和三锚点pipeline补丁。框架、依赖、Gateway、H3源码、native库、权重与业务profile基线均继承，详细数值边界见 [生产说明](../../container/seedvr2-bf16-production/README.md)。

`release-lock.json` 固定实际父digest/imageID、目标仓库、分支、唯一workflow marker、两项源码变化、12项真实CPU精度检查及五项镜像身份环境。`parent-source-manifest.json` 来自已发布26d实际证据，完整2037项；`expected-source-manifest.json` 仅改变pipeline并增加helper，完整2038项。不能用当前Git HEAD的完整Dingo树替代父镜像实际安装内容。

## 本地检查和冻结

Windows CMD中使用已有Python环境：

```bat
cd /d D:\AI\dynamo-seedvr2-bf16-20261112
python -B delivery\seedvr2-bf16-production\ops.py check
python -B delivery\seedvr2-bf16-production\ops.py seal
```

检查包括Python语法、回归、实际workflow路由和Bash语法。Windows没有目标Torch时，真实CPU自检明确待最终Linux镜像执行，不下载替代大包或把模拟metadata当实际导入证明。

本地实现、K8S接口和独立评审完成后冻结同一tree／父digest／配置／目标；协调会话将绑定的 `BUILD_REVIEW.json` 放到权威交接的 `release/reviews/`，再由唯一线上提交者导入：

```bat
python -B delivery\seedvr2-bf16-production\ops.py import-review
```

`ops.py` 不提供自动push入口。触发CI的Git push本身就是构筑请求，只能由本次指定唯一提交者在最终冻结复核后执行。其余会话不push、dispatch或rerun。

## 线上构筑合同

指定分支为 `Dingorouter-H3-vllm-omni-jirx`；唯一marker为 `[seedvr2-bf16]`。新workflow仅构筑本生产Dockerfile，旧镜像job显式排除此marker。Actions中额外的skipped工作流行不等于额外镜像构筑。

流程为：冻结及实际父身份验证 → build/load且不发布 → 最终同imageID的UID1000禁网验证 → 单次push → 完整digest回拉并比较imageID → 始终保存真实Harbor结果和制品。

存在的成功tag仅按其既有digest拉取、验收和复用，不覆盖发布。排队／运行／结果未知时只续查原run与产物；明确失败后先核实是否已发布，再在授权范围内修正、复核、重新冻结并顺序重提。没有固定attempt=1限制，也不清空真实历史。发布后回拉未确认保留pending状态，不以新提交替代查证。

## 发布交接

发布后给出完整digest、imageID、源码commit/tree、父digest、精确补丁／配置锁、实际Linux验证、ZIP及逐证据哈希。H3/SR使用同一成品。新profile继续H3 BF16+Cache-DiT、FL无卸载／Ref模块卸载，SR stage显式开启BF16 helper而CLI/VAE/noise/solver保持FP16。

本次两个 `H3_SR_PRODUCTION_*` 镜像身份与两个 `SEEDVR2_BF16_*` 身份都绑定新child commit/26d父镜像，以通过真实startup检查。旧父e5cf/c88及原清单/receipt独立保留；这不表示父文件都重新构筑。

部署方可调用 `python3 -B /opt/seedvr2-bf16/runtime.py verify --output-dir <新可写目录>` 做无GPU源身份自检。首业务后核对所有SR rank的一次性precision receipt，再验收原失败窗、正常1080/W33、1440/W65、H3-only和H3→SR。诊断SP4成功不替代新生产digest/SP8的GPU验收。

上游PR仅另行准备材料，本构筑流程不提交上游PR或issue。
