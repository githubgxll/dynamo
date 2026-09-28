# Dingo Runner 每日回收

目标：5 区 elm-test 的 dingo-gxl-runner，Docker data-root=/runner-data/docker。

## 策略

- 每天北京时间 05:00（UTC 21:00，cron `0 21 * * *`）。GitHub 调度可能延迟，Runner 忙时排队。
- `.github/dingo-images.json` 中所有配置仓库（包括 enabled=false 的历史系列）的本地 Dingo 产物，合计保留创建时间最新的 5 个不同镜像 ID。
- 按 Docker inspect 的 Created 排序，不按拉取时间、tag 字典序或最后使用时间；同一时间按 ID 稳定排序。
- 只处理以 Git SHA 结尾的产物 tag。builder-*、buildcache-*、未知标签、外部基础镜像、无标签镜像不参与计数和删除。一个 ID 只要带有受保护标签，整个 ID 都保留。
- 候选的每个 tag 都需要验证远端 manifest 的 config digest 与本地 image ID 相同；支持单镜像 manifest 和明确匹配本地平台的索引。远端不可访问、tag 已漂移、平台不明确等情况跳过并使任务失败，不强制删除。
- 删除前输出完整 inventory、计划及恢复用的 digest/tag 映射。不修改远端仓库。
- 身份校验成功且镜像步骤已启动后，独立清理 7 天未使用的构建缓存；部分镜像无法校验时仍会回收缓存，工作流保留失败状态。缓存回收可能降低保留镜像后续构建的缓存命中率，但不删除这些镜像的标签。
- 不保证 Docker 总共只剩 5 个镜像，保护项和无法验证的候选会额外保留；也不保证释放 100GiB。最终可用空间低于 100GiB 时任务失败告警。

## 上线与互斥

1. 将脚本、配置和工作流合入 `DingoRouter-base`。本仓库默认分支目前是 `main`：还需要将 `.github/workflows/runner-gc.yml` 同步到 `main`，GitHub 才会注册每日 schedule 和 workflow_dispatch。仅合入 DingoRouter-base 不会启动定时任务。工作流会明确 checkout DingoRouter-base 获取脚本及配置，不需要将整个业务分支合入 main，也不要更改仓库默认分支。
   确认目标 Runner 标签包含 self-hosted/linux/x64/dingo/gxl，以及 REGISTRY_USERNAME、REGISTRY_PASSWORD 可用。
2. 单个 Runner 进程一次只接一个 job，因此清理 job 与该进程的构建 job 串行。GC concurrency 仅约束 GC 自身；禁止在 GC 期间手工 exec 构建。若存在共享同一 Docker daemon 的多个 Runner，必须先统一互斥，不能直接启用此方案。
3. 首次 workflow_dispatch 保持 apply=false，source_ref 默认为 DingoRouter-base（预览 PR 时可填可信的 PR 分支），检查 artifact 中 plan.json 的 keep/candidates/protected，以及 recovery.jsonl。预览只读；空间不足仍会报告失败。
4. 确认计划后手动 apply=true 验证实际清理。定时触发自动 apply。
5. 检查清理前后 df、docker system df，运行一次正常构建及推送，随后观察每日磁盘水位。

日志、清单和 Markdown 摘要上传为 Actions artifact，保留 14 天。不要把 registry 密码写入脚本或日志。

## 风险与回滚

- 删除缓存后构建可能变慢，重新拉取历史镜像会增加网络和磁盘 I/O。时间新旧不代表业务重要性，特殊保留需求应通过受保护标签表达。
- 禁用工作流可立即停止后续定时任务；撤销本次新增文件即可回滚配置，无需重启 Runner。
- 删除数据本身不可撤销。使用 recovery.jsonl 中的 digest_reference 执行 `docker pull <digest_reference>`，然后 `docker tag <digest_reference> <tag>` 恢复本地标签；缓存由后续构建重建。远端保留策略必须覆盖所需恢复周期。
- 本次方案不修改 Docker daemon、PVC，也不会自动清理工作目录、基础镜像或远端镜像。

## 本地验证

```bash
set -euxo pipefail
uv --cache-dir /tmp/dingo-gc-uv-cache run --no-project --with pytest \
  python -m pytest --noconftest -c /dev/null -p no:cacheprovider \
  .github/scripts/test_runner_gc.py -q
```

测试模拟 Docker/registry，不连接生产，也不删除本机镜像。

## 当前线上 Pod 一次性执行

无需重启 Pod，也无需修改 Deployment/PVC。以下命令在本地仓库根目录执行，经 5 区 master 中转；临时脚本和日志放在容器 `/tmp`，避免向已满的 `/runner-data` 写入文件。Pod 重建后应重新确认名称。

先暂存本地修复并预览，不删除镜像：

```bash
set -euxo pipefail
tar -cf - .github/scripts/runner_gc.py .github/dingo-images.json |
  tsh ssh --cluster=server.teleport.hd-04.zetyun.cn root@hd04-cci-k8s-master-1 \
    'kubectl exec -i -n elm-test dingo-gxl-runner-864f8d989b-9tblf -c runner -- bash -c "set -euxo pipefail; mkdir -p /tmp/dingo-runner-gc; tar -xf - -C /tmp/dingo-runner-gc; python3 /tmp/dingo-runner-gc/.github/scripts/runner_gc.py --config /tmp/dingo-runner-gc/.github/dingo-images.json --keep 5 --output /tmp/dingo-runner-gc/preview"'
```

检查 `/tmp/dingo-runner-gc/preview/plan.json` 和 `recovery.jsonl`，先导出留档。首次清理需要维护窗口：暂停新任务进入此 Runner，并确认没有 Runner.Worker、docker build/buildx、cargo/rustc 等构建进程。一次 pgrep 只表示瞬时状态，不能代替任务隔离。不要仅凭 docker ps 为空判断没有构建。

在确认维护窗口及删除范围后，执行应用并回收旧缓存：

```bash
set -euxo pipefail
tsh ssh --cluster=server.teleport.hd-04.zetyun.cn root@hd04-cci-k8s-master-1 \
  'kubectl exec -n elm-test dingo-gxl-runner-864f8d989b-9tblf -c runner -- bash -c '\''
    set -euxo pipefail
    test "$(docker info --format={{.DockerRootDir}})" = /runner-data/docker
    gc_run_dir="$(mktemp -d /tmp/dingo-runner-gc-apply.XXXXXX)"
    {
      date -Is
      df -h /runner-data
      gc_status=0
      python3 /tmp/dingo-runner-gc/.github/scripts/runner_gc.py \
        --config /tmp/dingo-runner-gc/.github/dingo-images.json \
        --keep 5 --output "${gc_run_dir}" --apply || gc_status=$?
      docker builder prune --all --force --filter until=168h
      docker system df
      df -h /runner-data
      date -Is
      exit "${gc_status}"
    } 2>&1 | tee "${gc_run_dir}/apply.log"
  '\'''
```

每次 apply 都重新取镜像清单并校验远端，可能与此前预览略有变化，因此维护窗口内不要运行其他构建。若 Docker 因卷完全满而无法完成元数据写入，或清理后仍不足 100GiB，停止恢复构建；单独评估扩大缓存回收范围或扩容 PVC，不直接删除 overlay2 文件。

执行完将 gc_run_dir 的日志和恢复映射导出保存，并检查最新 5 个产物、本地基础镜像、Docker 健康和正常构建/推送。之后恢复 Runner 接单。临时复制脚本只支持本次执行；每日自动回收仍依赖 main 上的定时工作流注册。
