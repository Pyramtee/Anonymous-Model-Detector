# TokenizerDetector v0.2

面向 OpenRouter 和 OpenCode Zen 匿名模型的 tokenizer 指纹检测工具。以输入原生计数、差分和有条件的特殊标记为主，输出可解释的兼容候选；不以模型自报身份、文风或未校准概率判断厂商。

固定参考包含 Qwen3.8、MiMo2.6、DeepSeek V4.1、Step5、Kimi K3、GLM5.3、MiniMax M3 和腾讯 Hy4。Step5 使用用户认可的固定镜像。Mistral、GPT-NeoX 仅作离线外部对照。

## 安装和离线准备

运行环境为 Python 3.11+ 的 Unix 类系统（面向 Linux/WSL，另在 macOS 上实测通过；原生 Windows 不支持运行锁所依赖的 `fcntl`）。请克隆仓库后在**源码目录内**使用下面的 `uv run`：项目根目录由包路径推导，参考清单、工件、缓存和 `.env` 都从仓库内的 `data/` 读取，因此不支持把 wheel 装到任意位置后全局调用（那样会找不到清单和参考文件）。

```bash
uv sync --extra dev
uv run --extra dev tokenizer-detector fetch --include-controls
uv run --extra dev tokenizer-detector list
uv run --extra dev tokenizer-detector fingerprint -o data/cache/fingerprint.json
```

fetch 只下载人工清单锁定的 tokenizer/config/编码资料，不下载权重、不调用推理 API。文件按 SHA-256 校验并存入 data/artifacts。完整八家参考缺失时检测会在联网前停止，不能自动退回旧模型。

## 检测、续跑和回放

以下 detect/resume 命令会发起真实模型请求，只在你决定在线使用且拥有相应额度时运行。

```bash
# API Key 通过 OPENROUTER_API_KEY 环境变量或项目 .env 提供。
# 快检：默认最多12次实际HTTP尝试，自动保存完整实验。
uv run --extra dev tokenizer-detector detect --model "vendor/anonymous" --mode quick

# 指定provider时使用完整endpoint slug。
uv run --extra dev tokenizer-detector detect --model "vendor/anonymous" --provider "endpoint-slug"

# 继续已有实验；深检是累计最多40次，不是再追加40次。
uv run --extra dev tokenizer-detector resume data/runs/RUN_ID --mode deep

# 回放不联网、不加载API Key、不依赖当前词表文件。
uv run --extra dev tokenizer-detector replay data/runs/RUN_ID -o replay.json
```

可用 --max-requests 显式修改累计上限，--run-dir 指定新实验目录，-o 额外导出 JSON。失败、重试、诊断 GET、参数降级和结果不确定的请求都占次数。已存在实验目录不会被新 detect 覆盖。

## OpenCode Zen 匿名模型

先安装 OpenCode v2，并确保 `opencode --version` 可用；当前验证版本为 2.0.28。后端调用本机真实客户端，经本地桥接访问 Zen，不需要安装插件。v1 在线调用返回 `opencode_unsupported_version`，历史 v1 桥接记录仍可离线回放。

```bash
uv run --extra dev tokenizer-detector detect --backend opencode --model big-pickle
uv run --extra dev tokenizer-detector detect --backend opencode --model opencode/space-bunny-free
uv run --extra dev tokenizer-detector detect --backend opencode --model exo-free --mode deep

# 免费模型单次连通性验证；public 明确选择免费访问，覆盖环境变量中的个人 Key。
uv run --extra dev tokenizer-detector detect --backend opencode --model mimo-v2.6-flash-free --api-key public --max-requests 1

# 续跑自动恢复原后端；预算仍为累计上限。
uv run --extra dev tokenizer-detector resume data/runs/RUN_ID --mode deep
```

支持 Zen 的 Chat Completions 模型 ID，包括 `big-pickle`、`space-bunny-free`、`exo-free`、`mimo-v2.6-flash-free`；模型可用性以 [官方模型列表](https://opencode.ai/v2/docs/console/models/) 为准。`opencode/` 前缀可省略。此后端不支持 Responses、Anthropic Messages、OpenCode Go 或自定义 API 地址。

凭证可通过 `OPENCODE_API_KEY`（环境变量或项目 `.env`）或 `--api-key` 提供。未提供时使用客户端的原生免费访问方式；不复制个人 OpenCode 配置或凭证。`--provider` 只适用于 OpenRouter。服务端拒绝访问时会保存错误并停止。

每个探针通过 UTF-8 标准输入传入新会话，保留空格、引号和换行。客户端通过 `opencode run --standalone` 启动私有服务，运行在独立临时工作目录，PWD、OpenCode home、临时目录及 XDG 配置/数据/缓存/状态均隔离，关闭插件、共享、快照、标题生成、压缩和预热。使用原生 v2 OpenAI-compatible provider，并显式配置 `maxTokensField: "max_tokens"`，保持输出上限校验。桥接只将计划内请求转发至 `https://opencode.ai/inference/v1/chat/completions`；客户端收到固定结束响应，模型工具调用不会传递给客户端执行。检测证据来自原始上游响应，客户端结束响应中的计数不参与判定。

OpenCode 的系统提示、工具定义和固定消息作为固定上下文保存并校验，由原有偏移假设与重复锚点验证计数。固定上下文、客户端或桥接版本、可见路由变化返回 `unstable_endpoint`。工作目录路径与日期也会进入上下文，因此跨日期续跑可能需要新建实验；临时运行目录的位置见报告 `opencode.runtime_dir`，需要续跑时请保留它。升级后的新实验使用 `opencode-bridge-v2`，旧实验不能跨版本继续请求。

OpenCode 后端的 quick/deep 均只使用输入计数证据，输出字节通道标为 `unsupported`。直接保存 `usage.prompt_tokens`，不从中减去缓存计数，也不把隐藏 reasoning 当成可见输出。Zen 未返回底层路由身份时，报告明确限制路由可见性。当前 tokenizer 参考库仍为上述八家；接入成功不保证能匹配任意匿名模型。

本机 OpenCode 2.0.28 已通过真实客户端连接模拟上游的完整检测、复杂 UTF-8 文本、同版本续跑和离线回放测试。历史 1.18.15 的 2026-10-07 Big Pickle 实验累计 10/12 次请求（含一次修复前的解析失败），9 次有效观测、四次相同锚点及两条留出确认得到与 MiMo2.6 tokenizer 参考兼容的结果，不证明权重或开发商身份。Zen 在 `[DONE]` 后追加的费用记录不会被当成 token 证据。单次连通性验证只核对请求和原生 usage，不足以产生 tokenizer 判定。

2026-10-14 使用 2.0.28 和 `mimo-v2.6-flash-free` 完成一次真实请求：HTTP 200、有效 SSE、原生 `prompt_tokens=5389`、`completion_tokens=32`，有效观测及离线回放一致，未发生额外请求或客户端错误。实验记录为 `data/runs/opencode-v2-mimo-flash-smoke-20261014-01`。报告的 `insufficient_evidence / budget_exhausted` 来自单次预算，不代表连接失败。

## 如何解读结果

- compatible：本次观测和留出确认下，与一个家族兼容。
- ambiguous：多家仍兼容，例如目标将 Step/DeepSeek 的特殊标记当普通文本处理。
- no_reference_match：当前参考及编码假设不能解释观测；不等于证明发现新词表。
- insufficient_evidence：缺少必要测量或确认。
- unstable_endpoint：计数、路由或请求配置不能稳定比较。
- reference_unavailable / request_failed：参考或请求条件不满足。

tokenizer 兼容不证明模型权重、开发商或具体参数规模相同。输出 token 路径也不必等于对最终字符串重新 encode 的结果。completion_tokens 可能包含隐藏 reasoning，因此不作为严格重编码等式。

两种本地策略都关闭自动 BOS/EOS：literal_special 使用完整 added-token 匹配规则；plain 绕过全部 added matcher。plain 是明确的本地测量假设，不是对所有服务端行为的断言。

## 数据与可复现性

- data/reference-manifest.json：唯一人工维护的固定版本、文件大小和哈希。
- data/artifacts/：校验后的公开参考文件，忽略提交。
- data/cache/：派生指纹矩阵，可重建。
- data/runs/RUN_ID/：events.jsonl、report.json、report.md、run.lock。
- tests/fixtures/：固定 token IDs 及独立官方编码 oracle 的模拟计数。

events.jsonl 是唯一事实源；发送前 fsync 请求预留事件，崩溃后的未完成尝试不会退款。日志损坏或版本不兼容拒绝联网续跑。报告保留候选、偏移、残差、排除证据、停止原因与版本；不保存 API Key。

参考、算法或探针版本变化后，重新创建实验；历史记录仍可用原版本回放。不自动追踪新模型。人工更新 manifest 后重新准备文件、构建矩阵并执行全部离线验收。

## 离线验收

```bash
# 小型单元/模拟请求测试，不要求完整下载或安装 OpenCode。
uv run --extra dev pytest -m "not integration and not opencode_native" -q

# 真实 v2 客户端连接本地桥接及模拟上游；不会调用在线模型。
uv run --extra dev pytest -m opencode_native -q

# 先fetch --include-controls，再运行完整验收；测试阶段禁止真实网络。
uv run --extra dev pytest -q
```

参考集成测试缺少任何必需参考会明确失败，不会跳过伪装通过。`opencode_native` 测试要求安装 v2，缺少客户端时明确跳过；本机验收必须实际通过。HTTP 使用 MockTransport，固定观测来自独立 oracle，不以被测适配器临时生成“远端真值”。离线通过不代表线上准确率已经验证。

## 从v0.1迁移

fetch/list/detect 保留，但参考清单、参数和报告采用新版结构。--max-calls 改为 --max-requests；--unknown-threshold 和 --no-completion 取消。confidence、unknown_score、S1/S2/S3 字段不再输出。

旧 data/tokenizers、data/vendors.yaml 和 calibration JSON 保留为历史材料，不参与v2检测。旧在线校准脚本停用；不把缺少完整实际文本的旧报告转换为伪造的v2实验。

详细原理见 [技术方案](docs/technical-plan-2026-09-26.md)，实施架构见 [架构文档](docs/architecture-v2.md)。

OpenCode v2 适配后的完整本地验收 **201 项通过**，包含真实 v2 客户端连接模拟上游的两项测试。原 v2 基线的 159 项验收详情见 [基线验收报告](docs/offline-acceptance-2026-09-27.md)。词表文件、矩阵缓存和实验记录保存在项目 `data` 目录内且不纳入 Git；验收报告里提到的那批模拟实验目录与指纹矩阵是当时本地生成的产物，克隆仓库后并不存在——指纹矩阵可用 `fingerprint` 命令重建，但那批实验目录没有随仓库分发、也没有一键重建入口（回放链路本身由测试套件覆盖）。未执行真实 OpenRouter 请求。
