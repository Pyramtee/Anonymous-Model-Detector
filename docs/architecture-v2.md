# TokenizerDetector v2 实施架构

本文件落实用户批准的架构计划。冻结参考集为 2026-09-26 核查版本；Step5 镜像按用户认可的可信来源处理。代码在项目内通过 `uv run` 使用，面向 Linux/WSL，另在 macOS 上实测通过；原生 Windows 因缺少 `fcntl` 不受支持。项目根目录由包路径 `parents[2]` 推导，参考清单、工件、缓存和 `.env` 都从仓库内的 `data/` 读取，因此只支持源码目录内运行（editable 安装），不支持把 wheel 装到 site-packages 后全局调用。准备公开 tokenizer 文件不需要模型权重；检测与下载是不同命令。本轮验收不运行真实 OpenRouter 请求。

## 职责和依赖

| 部分 | 实现 | 职责 |
|---|---|---|
| 数据契约 | models | Probe、Profile、ProbeBank、Observation、Action、Conclusion；schema v2 |
| 固定参考 | references | 人工维护 manifest、固定 commit/hash、内容寻址缓存、完整性预检 |
| 编码适配 | adapters | HF/Kimi 两种策略、组件摘要、可靠字节映射 |
| 指纹矩阵 | probe_bank | 固定完整包裹、选择/确认/输出池、版本缓存、候选对见证 |
| 判定 | engine | 纯计算状态机、所有合法假设、严格偏移、确认、弃权 |
| HTTP | transport | 每次 send 只发送一次；持久预算；不内置重试或重定向 |
| 日志 | run_store | JSONL/fsync、独占锁、崩溃与损坏处理 |
| 编排 | runner | 调度、有限重试、能力降级、续跑、无网络回放 |
| 表达 | cli/report | 六个命令、中文 Markdown、JSON 同源报告 |

判定和编码层不能导入网络客户端；回放仅用日志中的 bank、请求观测及决策，不读取凭证或下载文件。外部对照不进入正常矩阵或选择器。

## 固定参考与编码策略

唯一维护入口为 `data/reference-manifest.json`。每份工件明确仓库、完整 revision、路径、大小、SHA-256 和用途。目标包含 Qwen、MiMo、DeepSeek、Step、Kimi、GLM、MiniMax、Hy；Mistral/GPT-NeoX 为外部对照。文件存于 `data/artifacts/<sha256>/<filename>`，不纳入 Git。旧缓存保留，只允许内容验证后复用，不能自动回退旧模型。

`literal_special` 保留全部 added matcher 属性；`plain` 绕过全部 added matcher，保留基础 normalizer、pre-tokenizer、BPE 和 decoder。二者都禁用自动 BOS/EOS。Kimi 由固定官方数据和审阅后的本地实现复现，不执行远程代码。bytes 无法正确还原的后端返回不可用，不将词表展示字符串冒充 token bytes。

## 推断流程

完整用户文本是 `Data:\n{payload}\nEnd.`。单 user、非流式、无 nonce/system/tools。每个参考×策略维护自身偏移；同实验组所有有效选择样本 `y - T(x)` 必须相同。保留全部合法假设；别名不投票。路由/配置变化不得按更高拟合度重新分组；当前实现选择停止为不稳定，保存原证据。

选择器确定性最小化最坏剩余家族，再比较策略数、预测间距、请求成本和文本长度。标记/损坏对照作为组合。稳定锚点与至少三个选择文本完成后，冻结全部假设与偏移，执行两条未使用确认样本。无法区分和无法测量是不同结果；softmax 与概率置信度已取消。

快检默认12次实际HTTP尝试，正常安排3次间隔锚点、3次普通选择、最多4次标记区分、2次确认。深检继承同一实验账本，默认累计40次。输入不能区分时深检可最多使用4次输出字节尝试，可靠输出词元只是词表兼容证据；关键输出排除也必须确认。logit_bias 不实现。

## 持久化和故障

`data/runs/<run_id>/events.jsonl` 是唯一事实源，report.json/report.md 为派生报告。请求发送前先追加 attempt_started 并 fsync，之后执行一次 HTTP send，再记录结果。预算由 started 数量恢复；孤立 started 不退还、不自动重发。进程间用 fcntl 独占锁；日志损坏拒绝在线续跑，写入失败拒绝继续发送。

每个实验保存完整 bank、固定参考清单快照、版本身份、实际请求、脱敏响应、usage 来源、路由、能力与确认决策。API Key 不写入日志，响应中反射的凭证也脱敏。输入原生计数字段缺失时只允许预算内查询 generation 原生字段，不采用估算替代值；输出 bytes 验证不查询无关的 prompt usage。response cache 命中不作为有效计数。

输出首次排除若导致家族集合缩小，两条新留出样本的交集必须从输入阶段候选出发独立复现该排除，不能借用首次输出结果。没有区分信息时保留已确认的输入歧义。输出路由变化返回不稳定状态。报告在同一实验锁内写入，避免续跑期间覆盖另一版本的报告。

回放除检查 JSONL 完整性外，还拒绝未知事件和不关联成功请求的观测；探针、阶段和实际发送文本必须一致。语义损坏不会触发网络续跑或静默修复。

串行发送间隔3.1秒，connect/write/pool为10/10/5秒，read为60秒；这是阶段超时，不是服务器执行时间保证。单逻辑请求最多一次暂时性错误重试，最多自动等待30秒；POST读取超时保守停止。输入输出上限32，输出字节上限128；声明的最小值最多适配至128，永不删除输出上限。

## 命令与兼容性

### OpenCode Zen 后端

`detect --backend opencode` 使用真实 OpenCode v2 的 `opencode run --standalone --format json`，每次私有服务、新建会话，探针以原始 UTF-8 标准输入传入。规范化客户端版本，在线拒绝 v1；已知 v1/v2 桥接记录均可离线回放，但不能跨客户端或桥接版本续跑。`--backend` 缺省为 OpenRouter；旧 v2 日志缺省后端亦为 OpenRouter，保留历史回放报告结构。`resume` 从日志恢复后端，不允许切换后端、模型或 provider。

`opencode` 模块管理独立工作目录、PWD、OpenCode home、临时目录与 XDG 配置/数据/缓存/状态，并启动仅监听 loopback 的随机路径桥接。原生 v2 配置关闭辅助请求，使用 `@opencode/ai/providers/openai-compatible` 与 `compatibility.maxTokensField: "max_tokens"`。实际请求保持客户端系统提示、工具定义、固定消息与身份头；转发只允许 `https://opencode.ai/inference/v1/chat/completions`。发送前走统一的持久预算、节流和 fsync。客户端额外请求不转发；重试统一由 runner 控制。桥接向客户端返回不含工具调用的固定 SSE 结束响应，检测器仅使用真实上游结果。网络 POST 超时仍保守记录为结果未知。

首个请求将唯一 user 文本替换为占位符后冻结完整请求上下文及 SHA-256。后续请求、续跑和离线回放都检查原始探针与上下文，不能通过重新分组吸收变化。日志新增可选 `backend_initialized`、`request_context_frozen`、`client_request_blocked` 事件，保留原始 SSE 与归一化响应并在回放时交叉核验计数、路由和实验组。API Key 和本地配置内容不写入日志。

客户端版本或固定上下文（含工作目录、日期、工具定义）变化为不稳定；运行目录缺失要求新实验。此后端只实现输入计数通道，不调用 OpenRouter generation 诊断，不使用客户端统计替代原始 usage，不从 prompt_tokens 中扣除缓存 tokens。deep 模式保持累计预算但不启动输出字节探针。服务端未暴露底层路由时明确记录可见性限制。

`fetch/list/detect` 保留；增加 `fingerprint/resume/replay`。旧 confidence/S1/S2/S3 和阈值接口取消，旧参数给迁移提示。CLI不在导入时读取.env。环境变量优先于.env，仅有联网意图的命令加载凭证。

旧校准记录保留为历史材料，不进入新算法；旧线上校准脚本停用。旧报告因缺少原始实际文本，不支持伪造迁移为v2回放记录。同版本回放必须产生相同结论；版本不兼容时拒绝原地续跑，保留历史数据。

## 验收

单元测试使用手写计数矩阵和 HTTP MockTransport；参考集成测试要求完整十份工件，缺失明确失败。`opencode_native` 测试调用真实 v2 客户端，覆盖完整检测、复杂文本、运行隔离、同版本续跑及回放，上游始终使用 MockTransport；缺少 v2 客户端时明确跳过。静态 golden 使用独立官方加载路径产生，测试运行时不重写期望值。所有测试默认封锁 Python socket 出站连接；真实客户端仅连接本地桥接，关闭模型下载。

验收覆盖：参考哈希/策略/ID、外部对照、共享词表、确认隔离、模板反例、缺失usage与缓存、路由漂移、非规范生成切分、12/40预算、日志失败/崩溃/锁、CLI迁移、续跑和无凭证回放。离线结果不宣称为线上识别准确率。
