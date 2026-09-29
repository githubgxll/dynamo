# Dingo Runner 持久化回收

代码、镜像配置和部署来源均为 **DingoRouter-base**。每日调度运行在 Runner Pod 内，不依赖 GitHub 默认分支或 Actions schedule。合并 PR 只更新代码；必须应用生成的配置并重建 Runner Pod 才生效。此方案仅适用于一个 Runner、一个 Docker daemon、一个 PVC 的部署。

## 策略和边界

- 每天北京时间 05:00 触发；每分钟检查一次。遇到构建则等待空闲，Pod 停机错过时间后补跑最近一次，不补跑所有历史日期。首次启动没有成功记录也会补跑。
- `.github/dingo-images.json` 中所有配置仓库（含 enabled=false），合计保留 Created 最新的 5 个不同产物镜像 ID；不是每个仓库 5 个。纳秒时间排序，同时间按 ID 排序。
- 仅匹配生成白名单中的精确仓库、完整标签前缀和 commit_sha_length 指定的 SHA 长度（目前 12 位）。renderer 复用构建流水线的标签生成函数：Dynamo 前缀来自镜像配置，vLLM/SGLang 前缀来自 container/context.yaml 对应 CUDA 的 runtime_image_tag，包含 disabled 系列。前缀按字面量匹配，不当作正则表达式。experimental-deadbeef 等未知前缀一律保护；历史前缀在配置变更后默认保护，不自动扩大匹配范围。缺少生成白名单时拒绝镜像清理，不能直接把原始 .github/dingo-images.json 当作部署后的 GC 配置。
- 只删除上述白名单识别出的 Git SHA 产物标签。builder-*、buildcache-*、基础镜像、未知标签、无标签镜像均保护；一个 ID 有受保护标签则整个 ID 保护。因此 Docker 镜像总数可以超过 5。
- 删除前验证远端 manifest 的 config digest 与本地 ID 一致，保存 inventory、计划、digest/tag 恢复映射。远端不可访问、标签漂移或平台不明确时保留镜像并产生告警；缓存清理继续。删除不用 force，不修改远端仓库。
- 每日清理 7 天未使用的构建缓存。空闲空间低于 150GiB 时逐级尝试旧缓存回收、缓存保留预算 30GB、全部未使用缓存回收，每阶段重新检查磁盘，达到目标即停止。
- Docker daemon 自身启用 BuildKit GC，defaultKeepStorage=80GB。这是缓存策略预算，不是整个 PVC 的硬配额，也不限制受保护镜像和工作目录。
- job started hook 在步骤执行前检查磁盘，低于 150GiB 先回收缓存，仍不足 100GiB 则拒绝执行；job completed hook 在低水位时回收。单次构建仍可能消耗超过剩余空间，需根据最大构建峰值调整阈值，不能承诺永不满盘。
- job hook、每日清理及手工入口共享 flock 和 Worker PID/启动时间标记。GC 超时、进程被杀或异常时保留 maintenance.json，阻止新任务，避免 Docker 后台操作未结束就开始构建。调度进程退出会终止 Runner，交由 Kubernetes 重启。
- 手工 exec 的构建必须经过下述 run 包装入口；绕过包装的命令不受互斥保护。Docker 内置 GC 遵循 BuildKit 自身的缓存引用保护机制。

## 生成与上线

在 DingoRouter-base 合并后的 checkout 中执行。生产应用应安排 Runner 无运行中任务的维护窗口，暂停任务提交，确认没有手工构建。

```bash
set -euxo pipefail
uv run --no-project --with pyyaml python deploy/ci/github-runner/render_gc.py \
  --output /tmp/dingo-gxl-runner-gc.yaml
```

私有仓库建议使用独立、仅有拉取权限的 `kubernetes.io/dockerconfigjson` Secret，生成时增加 `--registry-secret <已有Secret名称>`。该 Secret 的 `.dockerconfigjson` 会只读挂载到独立目录，仅镜像远端验证使用，不受 Actions login/logout 影响。不要将凭据写入 Git 或日志。未指定时使用 Runner 现有 Docker 客户端凭据；若其失效，只会跳过无法验证的镜像并告警，不能视作镜像回收已经有效。

通过 5 区 master 备份现有 Deployment 和启动 ConfigMap；如已安装，还需备份 dingo-gxl-runner-gc ConfigMap。备份应妥善保存。以下命令是上线步骤，不由测试自动执行：

```bash
set -euxo pipefail
tsh ssh --cluster=server.teleport.hd-04.zetyun.cn root@hd04-cci-k8s-master-1 \
  'kubectl -n elm-test get deployment/dingo-gxl-runner configmap/dingo-gxl-runner -o yaml' \
  > /tmp/dingo-runner-before-gc.yaml

tsh ssh --cluster=server.teleport.hd-04.zetyun.cn root@hd04-cci-k8s-master-1 \
  'kubectl apply --dry-run=server -f -' < /tmp/dingo-gxl-runner-gc.yaml

tsh ssh --cluster=server.teleport.hd-04.zetyun.cn root@hd04-cci-k8s-master-1 \
  'kubectl diff -f -' < /tmp/dingo-gxl-runner-gc.yaml
```

`kubectl diff` 有差异返回 1，检查差异后再单独执行应用。确认 Deployment/PVC 名称、卷容量、节点约束、代理与当前线上一致；不能用仓库旧配置覆盖线上无关变更。

```bash
set -euxo pipefail
tsh ssh --cluster=server.teleport.hd-04.zetyun.cn root@hd04-cci-k8s-master-1 \
  'kubectl apply -f -' < /tmp/dingo-gxl-runner-gc.yaml
tsh ssh --cluster=server.teleport.hd-04.zetyun.cn root@hd04-cci-k8s-master-1 \
  'kubectl -n elm-test rollout status deployment/dingo-gxl-runner --timeout=600s'
```

生成包包含启动配置、GC 脚本 ConfigMap、Deployment 与 PVC；校验和触发配置更新后的 Pod 重建。后续更新继续使用 renderer，不直接应用未包含 GC 挂载的基础 YAML。无需修改或同步 main。可选手工 Actions 工作流的 UI 注册仍受 GitHub 默认分支规则约束，下述 kubectl 入口完全不依赖该 UI。

## 验证和日常操作

**Actions UI 不是低水位应急入口。** 手动 GC workflow 同样执行 job-started hook；如果缓存回收后仍不足 100GiB，该作业在镜像清理步骤之前就会失败。即使 apply=false，前置 hook 也可能回收缓存。此限制有意保留，不按工作流名称或 job 环境变量豁免准入，避免普通构建绕过保护。

如果空间主要被旧镜像占用，应在 Runner 已启动且无运行中任务时，使用下文 `kubectl exec ... manager.py manual --apply`。该入口不执行 job-started/磁盘准入，仍检查 Docker root、构建互斥锁及中断标记，允许低于 100GiB 时尝试镜像回收。若 Pod 启动阶段已经因低水位失败，则不能依赖 exec；需按异常恢复流程建立维护窗口处理卷或扩容。清理后达到准入水位才恢复构建。

以下 `kubectl` 命令在 5 区 master 上执行；本地通过上述 tsh ssh 访问。先检查 Pod Running、Runner 在线、启动日志无异常，再验证：

```bash
set -euxo pipefail
kubectl -n elm-test exec deployment/dingo-gxl-runner -c runner -- df -h /runner-data
kubectl -n elm-test exec deployment/dingo-gxl-runner -c runner -- \
  python3 /etc/dingo-gc/manager.py manual
# 检查预览计划后执行一次实际清理
kubectl -n elm-test exec deployment/dingo-gxl-runner -c runner -- \
  python3 /etc/dingo-gc/manager.py manual --apply
kubectl -n elm-test exec deployment/dingo-gxl-runner -c runner -- \
  cat /runner-data/gc/status.json
kubectl -n elm-test exec deployment/dingo-gxl-runner -c runner -- \
  curl --fail http://127.0.0.1:9105/metrics
```

预览目录位于 `/run/dingo-runner-gc/preview-*`；退出码 2 表示存在无法验证而保留的候选。实际回收日志和恢复映射在 `/runner-data/gc/run-*`，保留 14 天；PVC 写入失败时临时保存在 `/run/dingo-runner-gc/run-*`，重建 Pod 前先导出。预览目录需按需导出、删除。

运行一次正常构建和推送，确认 started/completed hook 执行；在构建期间调用清理应提示 busy。次日 05:00 后检查 last_daily_date、last_daily_success、free_bytes 与日志。验证最新 5 个产物及受保护基础镜像仍在。

手工构建用统一入口，例如：

```bash
set -euxo pipefail
kubectl -n elm-test exec deployment/dingo-gxl-runner -c runner -- \
  python3 /etc/dingo-gc/manager.py run -- docker build -t example:local /runner-data/example
```

`deploy/ci/github-runner/gc/alerts.yaml` 提供可选 PrometheusRule：PVC 可用空间低于 100/50GiB、Deployment 不可用、每日 GC 超过 36 小时未成功、GC 中断/失败和镜像验证警告。PVC/Deployment 告警由集群指标提供，Pod 崩溃时仍能检测。安装前必须确认 Prometheus Operator、kube-state-metrics、kubelet 卷指标、规则选择标签、Pod 9105 抓取和告警路由；仅有文件或 scrape annotation 不代表监控已生效。规则需单独应用并在 Prometheus 中验证查询及告警送达。

## 风险、异常和回滚

- 应用部署会重建 Pod，运行中的构建会中断；必须在空闲窗口操作。缓存回收增加后续构建时间及网络/磁盘 I/O；只读仓库凭据故障会导致历史镜像累积。
- 出现 maintenance.json 时先查看其引用的日志与 Docker 状态，导出临时报告，确认无构建后重建整个 Pod，让客户端和 Docker daemon 一起停止。不要直接删除标记后接单；daemon 可能仍在处理超时命令。重启只是解除异常运行状态，不会清空 PVC，也不能代替回收。
- 如果不足 100GiB 且缓存已无可回收内容，应分析受保护镜像、工作目录和日志或扩容 PVC；不要直接删除 overlay2 文件。
- 配置回滚：在维护窗口通过 master 对备份的 Deployment/启动 ConfigMap 执行 `kubectl apply -f -`，恢复已备份的 GC ConfigMap（若有），再 `kubectl rollout restart deployment/dingo-gxl-runner -n elm-test` 并检查 rollout。首次安装回滚后可删除不再挂载的 GC ConfigMap。仅 rollout undo 不能恢复 ConfigMap 内容；PVC 不删除。可选告警规则单独恢复或删除。
- 删除的数据不能通过配置回滚恢复。按 recovery.jsonl 的 digest_reference 执行 docker pull，再 docker tag 恢复标签；前提是远端对应 digest 仍保留。缓存由后续构建重建。

## 本地测试

```bash
set -euxo pipefail
uv --cache-dir /tmp/dingo-gc-uv-cache run --no-project --with pytest --with pyyaml \
  python -m pytest --noconftest -c /dev/null -p no:cacheprovider \
  .github/scripts/test_runner_gc.py .github/scripts/test_runner_gc_manager.py -q
```

测试使用模拟 Docker/registry，不连接生产，不删除本机镜像；包含跨进程文件锁、中断保护、水位、时区、渲染配置及 Bash 语法验证。
