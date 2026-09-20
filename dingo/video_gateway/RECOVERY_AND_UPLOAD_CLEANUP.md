# Result handoff 与上传临时目录回收

## Result handoff 异常处理（M5）

Worker 完成后，Gateway 用 etcd 事务写入结果引用、将任务置为 `finalizing`，并释放执行
slot。遇到超时或断连，必须先重新读取任务，不能根据客户端异常推断事务没有提交。

- 已提交匹配的 handoff：接着完成后处理，不再执行模型。
- CAS 竞争、存储暂时不可用：重新读取、指数退避，加少量抖动，上限 5 秒；停止进程时
  可以中断等待。不会因为重试次数达到某个数就把未确认的结果当作失败。
- 构造或写入 handoff 的内部错误：核对状态后，只允许将同一 revision 的 `in_progress`
  任务置为失败；取消请求同样按这个窄条件写入终态。
- 写失败/取消状态后响应丢失：仍然读回核对。若终态写入本身持续发生程序错误，当前
  Gateway 停止准入并请求进程重启，由现有 owner 接管机制恢复，避免落入通用 Worker
  失败/重试路径。无法解析的持久化记录可能仍需人工修复，不承诺任意数据损坏自动恢复。
- 释放 slot 必须同时匹配 task、owner generation、execution token；不匹配时只结束旧
  任务，不碰当前 slot 占用者。终态 fenced release 不允许修改执行身份。
- 日志、指标异常不改变已提交结果。正常路径不增加额外存储读取。

## 上传临时目录回收（M6）

上传活跃性属于一次 HTTP 请求，不属于整个 Gateway 进程。

每个新 `_uploads/<uuid>` 目录有一个空文件 `.upload-heartbeat`。从开始接收 multipart
到输入校验、任务提交结束，当前请求每 `min(30秒, upload_grace_s / 3)` 更新该文件的
mtime。没有 JSON owner、额外 etcd key、每上传租约或 Gateway owner 查询。

清理器只删除目录和心跳都超过 `lifecycle.upload_grace_s` 的临时上传。默认 1 小时按
**最后一次成功心跳**计算。后台扫描的实际回收时间还取决于 `sweeper_interval_s`
（默认 30 秒）及文件系统响应时间。

- 正常提交：目录原子移动到任务目录，心跳标记移除。
- 正常失败、客户端取消：停止并等待请求与心跳协程，然后主动删除临时目录。
- 删除暂时失败：心跳已经停止，即使 Gateway 不重启，之后仍能超时回收残留。
- Gateway 崩溃：心跳自然停止，其他 Gateway 可按同一规则回收。
- 心跳 I/O 失败或心跳已超龄：中止上传；不会默默停止续期后继续接收请求。
- 心跳通过打开的文件描述符更新，目录提交重命名不会触发误报。取消时会等待已开始的
  文件操作结束后再关闭描述符和删除目录，包括重复取消的情况。

共享文件系统必须能在 Gateway 间及时反映文件 mtime，节点时钟应保持同步。超过宽限
期的进程暂停/存储中断按上传失效处理；此机制不承诺无限期暂停后还能恢复同一上传。
不使用跨 Pod `flock`：当前 DingoFS 实测没有提供需要的锁互斥效果。

### 从旧 owner 版本迁移

新旧清理器会保守跳过不识别的标记。旧 `.upload-owner.json` 或无心跳的历史目录不会
自动删除；确认旧版本全部下线、旧上传均已结束后，再人工清理这些历史目录。
新请求使用 `.upload-heartbeat`，不再有“必须等 Gateway 重启才能回收”的行为。

## 针对性回归

- `tests/video_gateway/test_handoff_recovery.py`：不确定提交、失败/取消回包丢失、CAS、
  中断存储等待、损坏终态写入出口及身份约束。
- `tests/video_gateway/test_early_release_dispatcher.py`：完整派发链路、模型调用次数、
  slot 复用与恢复失败时停止准入。
- `tests/video_gateway/test_upload_lifecycle.py`：真实慢 multipart、取消、心跳失败、
  清理失败残留、重复取消时文件操作收尾。
- `tests/video_gateway/test_artifact_store.py`：跨实例清理、停止续期后的回收、超龄心跳
  不再复活和目录提交重命名。
