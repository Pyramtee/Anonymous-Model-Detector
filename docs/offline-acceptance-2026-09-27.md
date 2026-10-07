# TokenizerDetector v2 离线验收报告

验收日期：2026-09-27（Asia/Shanghai）。实施版本：0.2.0；算法版本：`strict-v2.0`；报告格式：`schema_version: 2`。

## 验收结论

完整测试 **159 passed in 29.50s**，无跳过。包含124项不依赖完整词表的单元/模拟测试，以及35项需要完整固定参考的离线集成测试。测试运行时封锁真实 socket 连接，HTTP 使用 `httpx.MockTransport`。

本轮没有调用 OpenRouter，没有查询账户或消耗 credits。公开 tokenizer 文件的准备与推理请求分开；全部参考现已在本地通过固定大小和 SHA-256 校验。离线验收验证的是实现与明确测量假设的一致性，不是线上识别准确率。

## 固定输入与版本

| 项目 | 本轮值 |
|---|---|
| Python | 3.12.7 |
| tokenizers | 0.23.2 |
| tiktoken | 0.14.0 |
| 目标参考 | Qwen3.8、MiMo2.6、DeepSeek V4.1、Step5、Kimi K3、GLM5.3、MiniMax M3、腾讯 Hy4 |
| 外部对照 | Mistral v0.3、GPT-NeoX |
| 固定工件 | 33个不同内容哈希，共87,237,074字节 |
| 主矩阵 | 72条完整包裹探针 × 16个目标参考策略 |
| 独立真值 | 83条静态 token ID 样例；20个目标/对照策略 × 72条静态计数 |
| 探针版本 | `2026-09-27.3` |
| 探针 SHA-256 | `093c11719f213f2b7c9b2d9c0709846ded8bdd7209542ecd0bcb3e6d367a46e0` |
| 参考清单 SHA-256 | `039b1a9b0c54dee160f79ee73c74d7eb4e23d424f6293d0c6d64b12c0019f363` |

完整仓库、commit、文件、编码后端和已核查别名保存在 [固定清单](../data/reference-manifest.json)，同时快照进入实验日志。Step5 按用户认可的可信固定镜像处理。

静态真值由 [独立生成器](../scripts/generate_reference_fixtures.py) 直接加载官方 HF JSON，或通过 AST 提取 Kimi 官方配置常量后交给 tiktoken 生成，不调用被测适配器产生期望值。生成器源码哈希随 fixture 保存；测试不会自动重生成期望值。模拟 usage 使用明确的固定偏移17，不冒充真实服务端数据。

## 端到端结果

以下均经过固定观测 → 模拟 HTTP → 判定 → JSONL → 离线回放；20个报告与回放逐字段一致。计数是模拟传输尝试，实际网络请求为0。

| 参考 | literal_special | plain | 每种策略的尝试数 |
|---|---|---|---|
| Qwen3.8 | compatible：Qwen | compatible：Qwen | 8 |
| MiMo2.6 | compatible：MiMo | compatible：MiMo | 8 |
| DeepSeek V4.1 | compatible：DeepSeek | ambiguous：DeepSeek、Step | 12 |
| Step5 | compatible：Step | ambiguous：DeepSeek、Step | 12 |
| Kimi K3 | compatible：Kimi | compatible：Kimi | 8 |
| GLM5.3 | compatible：GLM | compatible：GLM | 8 |
| MiniMax M3 | compatible：MiniMax | compatible：MiniMax | 8 |
| 腾讯 Hy4 | compatible：Hy | compatible：Hy | 8 |
| Mistral v0.3（对照） | no_reference_match | no_reference_match | 6 |
| GPT-NeoX（对照） | no_reference_match | no_reference_match | 6 |

所有兼容/歧义结论均完成必要锚点、选择样本与两条冻结确认。外部对照由严格残差反例拒识，报告保留 `confirmation.complete=false`，不宣称建立任何参考兼容关系。外部对照不参与候选排名或选择器优化。

另做8项逐家留出实验，每项遍历该家的两种策略：移除 Qwen、MiMo、Kimi、GLM、MiniMax 或 Hy 后，原家族观测均被拒识。DeepSeek/Step 基础管线共享，移除其中一家时不能借缺失的竞争参考完成双向标记协议，保持弃权或拒识，不补造被移除的候选。

还检查了16个目标策略连续三轮的本地调度与确认可分性：普通家族累计8/11/14次；DeepSeek/Step 累计12/15/18次。普通家族和 literal_special 的结论符合预期，DeepSeek/Step plain 始终保留歧义。这是协议覆盖检查，不额外构成独立识别准确率数据。

## 关键验收覆盖

| 范围 | 主要覆盖 | 测试位置 |
|---|---|---|
| 文件与适配器 | 固定 commit/hash、缺文件明确失败、HF全部added matcher绕过、BOS/EOS、Kimi正则与特殊ID、原生字节 | test_references、test_adapters、test_reference_integration |
| 指纹与策略 | 原普通文本计数、Step/DeepSeek双向标记、组件差异、缓存失效、外部对照隔离、静态独立oracle | test_reference_integration、test_pinned_detection |
| 判定 | 大恒定偏移、严格残差、重复锚点漂移、冻结偏移、不同选择文本、留出信息量、共享策略歧义 | test_engine |
| 输出增强 | 原生bytes还原、非规范生成切分、缺bytes降级、无usage时不做多余GET、两条holdout独立复核排除 | test_engine、test_transport、test_runner |
| 预算和故障 | 第12/40次后禁止发送；重试/诊断/续跑/孤立预留记账；POST不确定结果不重发；锁与fsync失败 | test_transport、test_run_store、test_runner |
| 日志与路由 | route变化、未知事件、缺bank、无对应请求的观测、阶段/实际文本错配；损坏时不修改日志、不再联网 | test_runner、test_run_store |
| 命令与报告 | 六个命令、旧参数迁移、参考预检、无凭证回放、中文/JSON同源、历史报告不伪造迁移 | test_cli |

普通确认池在验收中补齐了独立 Unicode、连字和空白边界内容，使 GLM 的已选区分证据能够由留出样本复核。选择池、严格匹配条件及阈值没有为测试结果放宽。

## 本地交付与复现

完整指纹矩阵与组件差异：`data/cache/fingerprint-v2.json`。

本轮20个可回放实验：`data/runs/offline-acceptance-20260926T173626Z/`。目录名使用UTC，对应本地2026-09-27。汇总：[offline-acceptance.json](../data/cache/offline-acceptance.json)。这些文件属于项目本地缓存/实验，不纳入Git。

可直接进行无需凭证的回放：

```bash
uv run --offline --extra dev tokenizer-detector replay \
  data/runs/offline-acceptance-20260926T173626Z/step5--plain
```

准备好依赖和全部参考后重跑验收：

```bash
uv run --offline --extra dev pytest -m "not integration" -q
uv run --offline --extra dev pytest -q
uv run --offline --extra dev tokenizer-detector list
uv run --offline --extra dev tokenizer-detector fingerprint -o data/cache/fingerprint-v2.json
```

旧 `data/tokenizers`、厂商配置和历史校准JSON保留；旧 softmax/nonce/融合信号实现已由新流程替代，旧在线校准入口明确停用。没有提交、部署、Web服务或真实模型调用。

本轮没有验证实际端点的usage、模板、路由和高级参数行为。工具遇到不满足本地假设的情况会保留反例并弃权；tokenizer兼容本身也不能证明权重或开发商身份相同。
