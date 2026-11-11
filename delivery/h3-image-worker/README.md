# 独立 H3-Image Worker 镜像

本配方继承固定的官方 H3-Image amd64 镜像，保留 Torch2.14／vLLM0.33／cu130运行环境。新增原生 Images API Worker入口、只读模型校验与两处源代码修复。四卡TP4为默认配置，双卡TP2和单卡逐层卸载使用同一成品镜像。

父镜像为 `docker.io/vllm/vllm-omni@sha256:77c3f76764346918576ba4183daf169240070b659c919c7fff45df31bbf333c9`，发布目标为 `registry.hd-04.alayanew.com:8443/openclaw/vllm-omni`，分支为 `Dingorouter-H3-vllm-omni-jirx`，唯一构筑标记为 `[h3-image-worker]`。

## 修复与实际源码

父镜像复制的源码Git HEAD为 `c958c8f5f628f2f29d5e548af236df2b1f5b59ab`。本次已完整重算其152,195,329-byte源码层SHA256，冻结2138个Python源文件的哈希。最终安装版以实际RECORD和包目录的完整源文件集合为边界，逐文件投影到COPY来源清单校验；COPY中未入wheel的文件显式记录，版本生成文件单独锁定实际父哈希。仅回补以下公开上游修复。

1. `MiniMaxH3ImagePipeline._resolve_text_encoder_root` 优先读取已校验的本地 `text_encoder/`，满足离线组合目录加载。
2. Image VAE decoder转为BF16后重新计算FP32 RoPE频率，并保留重复准备时的CUDA Graph缓存。

修复来源为公开commit `0b74da131f963ab2fd60f7eba813145672cf81e4`。修改前后SHA256、载荷和符号范围保存在 `container/h3-image-worker/patches.lock.json`。其余父源码保持原字节。Omni版本与direct_url保留父安装来源，另记补丁身份，同时更新RECORD中两份实际修复文件的hash和大小。

## 运行入口与模型挂载

最终入口为 `python3 -P /opt/h3-image-worker/worker.py`，UID/GID为1000/1000。默认参数为 `--shared-root /source-models --profile h100-4-tp4`。

将已下载的共享根只读挂到 `/source-models`，将可写emptyDir或平台工作卷挂到 `/work`。hd04来源根为 `/dingofs/data2/mengsz/H3_Safety_Image_models/`，操作者应只读核查目标节点的实际可见性。支持以下有限候选布局。

```text
/source-models/MiniMax-H3-Image
/source-models/MiniMax-H3/text_encoder

/source-models/generation/MiniMax-H3-Image
/source-models/generation/MiniMax-H3/text_encoder

以上两种布局也可位于 /source-models/H3_Image_models/ 下。
```

发现多个有效候选时拒绝启动，操作者可把明确的图像模型根目录作为共享挂载来源。安全模型目录不进入图像资产选择。

Worker按既有83文件交接的真实来源固定Image `fbde899fffb5c5225ee8dd7027680113746af42e` 和Base编码器 `0919fb684102ef115126612f3fbb84d1027e7317`。启动检查53个必需资产的完整SHA256或Git blob SHA1、文件大小、分片index及路径归属。旧README与较新修订不同的部分保留其真实来源。

通过后在 `/work/h3-image-worker/model-view/` 内创建已校验来源的逐文件符号链接视图，原始模型目录继续只读。视图manifest记录源文件和哈希，存在的视图需校验后复用。vLLM实际模型参数替换为本次视图路径，日志和 `model-validation.json` 保存生效路径与argv。原计划 `/opt/models/minimax-h3-image` 是合并目录模式的默认逻辑路径，shared-root模式通过受锁视图解决分离编码器布局。

首次运行直接使用完整入口，避免外层ConfigMap再次构造视图或重复全量哈希。

```text
--shared-root /source-models --profile h100-4-tp4
```

其他配置将profile改为 `h100-2-tp2`、`h100-1-dlo`，或显式选择备选 `h100-1-te-offload`。可用 `--discover-models /source-models` 仅输出识别到的布局，`--shared-root /source-models --verify-assets-only` 只完成资产检查与视图准备。`--output-dir` 默认 `/work/h3-image-worker`。

## 验收与发布

本地使用 `delivery/h3-image-worker/ops.py check` 检查源码、锁、workflow路由和回归。`seal` 生成精确输入树和文件哈希。根协调者对同一tree／recipe／父digest给出 `reviews/BUILD_REVIEW.json` PASS，导入对应冻结回执后，唯一线上提交者才提交触发CI的push。

CI使用现有持久Docker builder，先在固定父imageID上只读执行独立实装probe，保存stdout／stderr和退出状态，再build/load候选。实际最终imageID的UID1000禁网验证通过后发布，并按完整digest回拉。运行时验证保留活动分发及全部metadata实例、COPY来源与实际RECORD完整覆盖、两个精确修复、四种CLI参数／dataclass传递、10项真实模块CPU检查，以及Worker本地回归。CPU用例包含PDD／tile数值、tiny VAE加载／dtype／FP32 RoPE及幂等，不包含完整VAE forward／decode或图像画质。

LMCache0.5.6的正确CUDA扩展是`lmcache.cuda_ops`。最终UID1000禁网环境在新进程严格导入`lmcache.lmcache_native`与`lmcache.cuda_ops`，四个实际原生库按`native.lock.json`核验字节、SHA256、唯一RECORD hash／size行及活动分发路径。保存`native-validation.json`，不吞掉导入／符号／driver错误，不把外层LMCache的CPU回退日志算作原生加载通过。vLLM的CUDA目标为`_C_stable_libtorch`；CPU平台尝试旧`vllm._C`的warning不单独证明GPU运行失败，真实CUDA算子仍待H100验收。

发布后对同一回拉imageID额外执行UID1000／禁网／只读的`tar`可见性与版本观察，保存`tar-validation.json`。缺少tar仅记录交接限制，可改用Python流式导出；不为此另打镜像。该观察不替代运行时或GPU门禁。

继承的 cuDNN `LD_PRELOAD`／`LD_LIBRARY_PATH` 保持有效。依赖检查使用严格pip／uv结果，uv检查使用独立无缓存模式，避免继承root缓存目录权限造成误失败。最终保存安装清单、CycloneDX包声明、源文件清单与发布证据。该声明材料用于实验镜像交接，生产许可审批按组内正式流程另验。

前一轮请求排队、运行或UNKNOWN时续查原run和Harbor。明确失败后先核对是否已有发布，再修复、复核、重新冻结并顺序重提。已有合格digest时复用同一成品。

实际H100驱动、CUDA／NCCL执行、模型加载、首图、图像规格与人工效果由后续K8S实验记录提供。
