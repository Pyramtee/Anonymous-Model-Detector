# TokenizerDetector v0.2

面向 OpenRouter 匿名模型的 tokenizer 指纹检测工具。以输入原生计数、差分和有条件的特殊标记为主，输出可解释的兼容候选；不以模型自报身份、文风或未校准概率判断厂商。

固定参考包含 Qwen3.8、MiMo2.6、DeepSeek V4.1、Step5、Kimi K3、GLM5.3、MiniMax M3 和腾讯 Hy4。Step5 使用用户认可的固定镜像。Mistral、GPT-NeoX 仅作离线外部对照。

## 安装和离线准备

运行环境为 Python 3.11+、Linux/WSL，在项目目录使用：

```bash
uv sync --extra dev
uv run --extra dev tokenizer-detector fetch --include-controls
uv run --extra dev tokenizer-detector list
uv run --extra dev tokenizer-detector fingerprint -o data/cache/fingerprint.json
```

fetch 只下载人工清单锁定的 tokenizer/config/编码资料，不下载权重、不调用推理 API。文件按 SHA-256 校验并存入 data/artifacts。完整八家参考缺失时检测会在联网前停止，不能自动退回旧模型。

## 检测、续跑和回放

以下 detect/resume 命令会发起真实 OpenRouter 请求，只在你决定在线使用且拥有相应额度时运行。本项目重构验收没有执行这些在线命令。

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
# 小型单元/模拟请求测试，不要求完整下载。
uv run --extra dev pytest -m "not integration" -q

# 先fetch --include-controls，再运行完整验收；测试阶段禁止真实网络。
uv run --extra dev pytest -q
```

集成测试缺少任何必需参考会明确失败，不会跳过伪装通过。HTTP 使用 MockTransport，固定观测来自独立 oracle，不以被测适配器临时生成“远端真值”。离线通过不代表线上准确率已经验证。

## 从v0.1迁移

fetch/list/detect 保留，但参考清单、参数和报告采用新版结构。--max-calls 改为 --max-requests；--unknown-threshold 和 --no-completion 取消。confidence、unknown_score、S1/S2/S3 字段不再输出。

旧 data/tokenizers、data/vendors.yaml 和 calibration JSON 保留为历史材料，不参与v2检测。旧在线校准脚本停用；不把缺少完整实际文本的旧报告转换为伪造的v2实验。

详细原理见 [技术方案](docs/technical-plan-2026-09-26.md)，实施架构见 [架构文档](docs/architecture-v2.md)。

本轮完整离线验收 **159 项通过**，详情与可直接回放的模拟实验见 [验收报告](docs/offline-acceptance-2026-09-27.md)。词表文件、矩阵缓存和实验记录保存在项目 data 目录内；未执行真实 OpenRouter 请求。
