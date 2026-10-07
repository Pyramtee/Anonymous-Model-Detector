# TokenizerDetector 宏观设计审计

审计日期：2026-09-26。对象：MiMo 草稿 `.mimocode/plans/1790415846309-cosmic-forest.md`、当前工作区实现、现有校准数据与测试。执行了离线测试和 tokenizer 实验，查阅公开官方文档；未调用付费模型 API。本报告区分设计结论、代码实证与尚未验证的生产行为。工作区原有改动未作修改。

**1. 总体判断**

方向可行，但当前方案适合定位为“tokenizer 行为指纹匹配原型”，尚不能作为可靠的匿名模型厂商归属检测器。首要工作是修正证据边界、测量方法和拒识机制，而不是增加更多厂商名称或调整融合权重。

OpenRouter 当前官方文档说明 usage 中的 prompt/completion counts 使用模型原生 tokenizer，因此输入计数是有依据的主要观测量。[Usage Accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting)

必须分开以下结论：

| 层次 | 实际可回答的问题 | 不能直接推出的结论 |
|---|---|---|
| 有限探针计数匹配 | 本次输入集合上的计数是否兼容某些参考 tokenizer | 完整词表或所有输入行为完全一致 |
| tokenizer 行为家族 | 哪些公开实现能解释观测 | 模型权重、规模、架构和训练来源相同 |
| 模型发布者/厂商线索 | 哪些已知模型使用这类 tokenizer | 匿名模型必定由其中某一厂商训练或发布 |
| API 服务提供方 | 哪个推理服务处理了此次请求 | 该服务商就是基础模型开发者 |

建议产品表述为：“对 OpenRouter 匿名模型进行 tokenizer 指纹检测，返回兼容的 tokenizer 版本、已知使用模型、无法区分的候选，以及带证据等级的归属线索。”

**2. 应保留与应调整的功能边界**

| 原设计 | 审计判断 | 建议 |
|---|---|---|
| CLI、只做 OpenRouter、先做文本 | 合理 | 保留，先保证观测可靠 |
| 本地公开 tokenizer 对照 | 合理 | 改成版本化参考快照，并验证官方行为 |
| 输入计数为主信号 | 合理 | 固定请求结构，建立重复测量与差分模型 |
| 直接报告厂商 family | 过度推断 | tokenizer 家族与厂商线索分别输出，支持跨厂商等价组 |
| 一家一个 tokenizer | 不合理 | 一厂多版本、多模型与 tokenizer 多对多映射 |
| 5–15 次调用 | 可作为快速模式目标 | 不能承诺每次足够；证据不足应停在 inconclusive |
| 手工固定探针 | 可作为第一批候选池 | 离线评估判别力，按剩余候选选择探针 |
| 不做自动判别串挖掘 | 可以暂缓复杂搜索 | 不应同时排除廉价的本地探针筛选 |
| 不本地复现 chat template | 有条件合理 | 必须验证残差稳定；模板变动不能一概当小噪声 |
| S1/S2/S3 固定权重融合 | 当前依据不足 | S2/S3 合并为输出侧辅助证据，按可用性启用 |
| softmax 置信度 | 不可解释为真实性概率 | 先输出匹配误差、候选集合和证据覆盖率 |
| 不做 calibrate 子命令 | CLI 形式可以简化 | 可复现的校准与独立验证不能省略 |
| 不做 Web UI、调度、多模态 | 合理 | 当前目标不需要扩展这些范围 |

**3. P0：tokenizer 相同不能证明厂商或模型相同**

草稿将“同词表策略”定义为报到厂商，隐藏了关键的跨厂商复用情况。同一家厂商也可能发布采用不同基础模型 tokenizer 的产品。DeepSeek 官方说明 R1 的蒸馏模型包含基于 Qwen 和 Llama 的系列，这说明发布者与基础模型谱系不能合并为一个字段。[DeepSeek-R1 官方说明](https://huggingface.co/deepseek-ai/DeepSeek-R1)

本项目 `data/vendors.yaml` 把 `DeepSeek-R1*` 等通配模式放进单一 DeepSeek tokenizer family；这不是经过逐版本验证的共享关系。类似地，某一历史模型的 tokenizer 不能自动代表厂商未来发布的所有模型。

“词表相同”也不足以描述完整 tokenizer：预分词正则、normalizer、merge ranks、added tokens、special-token 策略、post-processor 均可改变行为。有限输入上计数相同，只能称为“本轮观测不可区分”。即使对所有输入的计数相同，也未必识别 token ID 的编号方式。

要求：分别记录 tokenizer 实现、模型版本、发布者、基础模型谱系、服务提供方；结果必须容许跨厂商候选集合。禁止把 tokenizer 匹配直接写成“确定是某家匿名新模型”或“与公开模型权重相同”。

**4. P0：随机 nonce 污染了最重要的测量信号**

`client.py:100` 发送的是 `probe.text + "\n\n[nonce:随机UUID]"`，而 `detect.py:122` 本地只编码 `probe.text`。`signals/prompt_count.py` 假定不同请求之间的额外开销近似常量，但每次 nonce 不同，其 token 长度并非常量。

离线实验：固定输入 `Hello世界123`，固定随机种子 20260926，生成 200 个 32 位十六进制 nonce，计算完整输入减去原输入的 token 数。结果只代表当前本地后端，不是在线模型实测：

| 本地 family | nonce 后缀增量范围 | 均值 | 标准差 |
|---|---:|---:|---:|
| Qwen | 29–37 | 33.09 | 1.66 |
| DeepSeek | 20–30 | 24.43 | 1.96 |
| Kimi（当前后端） | 21–30 | 24.46 | 1.77 |
| GLM | 22–31 | 26.34 | 1.62 |
| MiMo | 29–37 | 33.09 | 1.66 |
| Step | 20–30 | 24.43 | 1.96 |
| MiniMax | 19–29 | 23.34 | 1.94 |

这里的波动会进入跨探针差分；固定的数十 token 增量还会影响“模板开销应较小”的残差先验。对于仅相差几个 token 的特殊串，影响不可忽略。当前 mock 仅添加固定 15 token，无法暴露该问题。

官方文档确认 response cache 命中会清零 usage，但这种缓存默认关闭，并支持 `X-OpenRouter-Cache: false`。应使用该开关并检查响应缓存状态，不必为了防缓存修改文本。普通 prompt cache 与 response cache 是两种机制，不能把所有 usage 异常都解释成缓存。[Response Caching](https://openrouter.ai/docs/guides/features/response-caching)

如果特定实验确实要添加 nonce，必须保存实际发送的完整内容，并对每个候选编码该完整内容；不能把 UUID 注释称作模型看不见的“隐藏注释”。

**5. P0：softmax 排名不是可信归属概率，unknown/ambiguous 语义错误**

`fuse.py` 对手工得分乘 8 后进行 softmax。它回答的是“在当前候选池中谁相对靠前”，并不回答“此候选真实正确的概率”。更换候选数量、添加相同 tokenizer 的多个别名，都可能改变数值。

当前实现中 unknown 需要“绝对得分低”与“相对置信度低”同时成立。其后果是：只要所有错误选项里有一个相对领先者，也可能输出确定 family。极端例子是只加载一个候选时，softmax 永远为 1，连匹配分 0 都能被接受。

使用仅 prompt 信号权重进行离线直接调用：

| 输入分数 | 当前结果 |
|---|---|
| 仅 Qwen=0 | Qwen，confidence=1，同时 unknown_score=1 |
| Qwen=1、MiMo=1，未声明等价 | Qwen，confidence=0.5 |
| Qwen=0.4、MiMo=0 | Qwen，confidence≈0.9608 |

在默认阈值下，ambiguous 分支要求 top.combined_score < 0.35，且前两名概率差小于 0.08。这样的低分近似平局已被前面的 unknown 分支截获；高分平局反而直接返回首个 family。结果可能依赖注册顺序。

建议采用分开的决策步骤：

1. 观测是否可靠；否则 `insufficient_observability` 或 `unstable_endpoint`。
2. 是否有候选达到独立校准的绝对误差标准；没有则 `unknown_tokenizer`。
3. 多个候选是否仍能解释观测；是则返回 `ambiguous` / 兼容集合。
4. 某一行为家族是否经过留出探针确认；是则返回 `compatible`，再单列归属线索。

在没有足够验证数据前，展示每条探针的实际/预测计数、稳健残差、最接近竞争者、重复稳定性、参考库覆盖范围，比展示“83% 是某厂商”更有意义。`unknown_score` 不能作为候选 softmax 的简单补数。

**6. P0：参考库按厂商压缩，且缺少完整 tokenizer 行为定义**

`fetch.py:71` 遍历仓库并取第一个可下载文件，所有版本落到同一个 family 目录；`resolve/main` 未锁定 commit；后续发现文件存在就直接复用。当前 hash 只描述本地单个文件，不能追溯下载时的仓库版本，也没有落实草稿所说的按 hash 自动归并。

参考库应保存：精确 repo、commit revision、下载时间、相关文件哈希、实现类型与版本、tokenizer 配置、special-token 策略、已验证模型关系。不同版本应并列存在，只有经过验证的等价实现才能合并；网络下载失败不得在语义上把“新型号参考”静默替换成“旧型号参考”。

Kimi 当前只读取 `tiktoken.model`，并在 `tokenizer_backends.py:31` 硬编码一套通用预分词正则，`special_tokens={}`。BPE 文件本身不足以定义 Kimi tokenizer；官方的预分词和特殊 token 定义也必须复现。该问题足以使整个 Kimi 参考列偏离真实行为，调权重无法补救。

独立离线对照使用相同本地 BPE 文件，仅将正则换成官方配置；未执行下载的远端 Python，使用 AST 提取常量：

| 长探针 | 当前 Kimi 后端 | 官方预分词规则 |
|---|---:|---:|
| long_zh_article | 309 | 307 |
| long_en_tech | 346 | 327 |
| long_mixed_code | 319 | 281 |
| long_zh_en_mix | 301 | 300 |

代码探针相差 38 token，约为正确规则计数的 13.5%。这也说明只比较 vocab 文件哈希不能验证完整行为。[Kimi 官方实现，固定 commit ac6c49f](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/ac6c49f04883bd0a0598b790693a72061c676629/tokenization_kimi.py)

另一个独立文件比较结果是：本地 Qwen/MiMo 的 151643 项 vocab、经序列化格式统一后的 151387 条 merges、normalizer/pre-tokenizer/post-processor 均相同；DeepSeek/Step 的整个 model 和这些处理配置也相同。两对都在 added tokens 上有差别。四条普通长探针分别给出 Qwen/MiMo `[341,347,300,279]`、DeepSeek/Step `[316,334,292,289]`。这些是当前快照的结果，不能外推到两个厂商的所有模型。

HF 后端也应明确 `add_special_tokens` 等设置。不要把“能加载且能 encode”当作“与厂商官方 tokenizer 等价”。

**7. P1：探针设计应依据可辨识性，而不是按类别凑数**

长文本在某些条件下有利于放大候选差异，但长度并不等于判别力。同一总 token 数可能由不同切分得到；几个长段落刚好同计数，不代表 tokenizer 相同。

当前默认 12 次调用，实际分配为：1 次能力探测、3 次长文本、6 次特殊串、2 次边界串；没有任何 completion 探针。前 6 个特殊串只覆盖 MiMo/Step 相关标记，分配与当前竞争候选无关。低预算时甚至会挤掉全部长文本。

应在本地计算候选 × 探针矩阵，衡量每条探针能分开哪些候选、差异是否超过实际噪声，选择覆盖难分候选对的小集合。未知偏移存在时，应比较差分签名或去除偏移后的签名，不能只看单次总计数的方差。

推荐分两阶段：普通文本粗筛；针对剩余候选选择高区分串。复杂自动字符串搜索可以延后，但现有探针池的离线筛选成本低，应纳入第一版。新增 Unicode、代码、数字等样本前先确认其带来的候选区分收益。

`detect.py:210` 使用全部长/边界/特殊探针的本地指纹计算等价组，但实际远程只执行其中一部分。这会把“理论上可区分”误当成“本次已经区分”。报告中的本轮兼容集合必须由本次真实有效观测确定。

**8. P1：特殊 token 是有条件的部署行为证据**

`<|...|>` 在公开 tokenizer 文件里是 added/special token，不意味着匿名服务必然允许用户文本触发这个 token。服务端可能采用普通文本编码、转义、过滤，或者因模板、模态协议产生额外处理。

这不是否定特殊串价值，而是要求先做控制实验：原串、破坏标记语法的对照串、带固定普通文本包裹的版本，以及已知模型在同一 provider 上的表现。服务端不把标记当特殊 token 处理时，不应直接排除该 tokenizer 家族。

“某标记独有”只能限定为当前版本化参考库内的差异，不能上升为全生态唯一标识。特殊串应在普通文本候选缩小后使用，并单独报告是否验证了服务端识别策略。

**9. P1：S2/S3 同源，并受隐藏输出与非唯一编码影响**

S2 对比复述文本的本地 token 数与 API completion count；S3 对比同一个输出的 chars/token。设字符数为 c，API token 数为 a，本地为 b，则相对比值误差为 `|(c/a)-(c/b)|/(c/b) = |b-a|/a`。它仍然是同一计数差的变换，没有增加独立观测信息。可以研究组合特征是否改善预测，但不能把二者当作两份独立证据增强置信度。

草稿“忽略 reasoning details”需要反转：官方说明 reasoning token 纳入 completion token 统计，且隐藏 reasoning 文本不等于关闭推理。因此可见 `content` 的重编码通常不能直接对应 completion 总数。还应考虑结束 token、工具输出、截断及服务端后处理。[Reasoning Tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)

更基础的限制是：可见字符串不唯一决定生成时的 token 路径。本地 Qwen tokenizer 可以将 `[71,68,75,75,78]` 解码为 `hello`（5 token），但重新 encode `hello` 得到 `[14990]`（1 token）。这是合法编码路径的离线反例，不表示在线模型通常逐字符生成；它证明严格等式并不成立。

输出侧建议降级为可选交叉验证：确认没有不可见 reasoning、完成状态正常、输出处理可解释后启用；支持 token/byte 级 logprobs 的端点可进一步核验。仅有文本与总 completion count 时采用经校准的容差。

当前 `_copy_fidelity` 会把子串或含有相同字符集合的错误复述判为高质量；报告还将 `fidelity_ok` 固定写为 True。这会进一步夸大输出证据可靠性。应对严格复述使用实际逐字相等判定，或直接把实际输出作为新的输入计数探针。

**10. P1：差分法需要稳定的测量条件**

推荐的工作模型是 `y_i = T_k(x_i) + b_g + ε_i`，其中 `x_i` 是实际计数文本，`g` 是相同 provider/模板/请求模式，`b_g` 是待估计偏移。这是需要验证的近似模型，边界合并、模板分支或处理链改写可能使误差随输入变化。

建议先在关闭 response cache 的情况下重复固定请求，估计基线噪声；固定消息数量、role、system、reasoning 模式、工具及多模态设置。普通文本的差分可优先使用固定包裹和控制串，但不能未经验证假定拼接 token 数可加。

OpenRouter 支持 provider routing 和 fallback；固定模型 ID 不等于全部请求的服务实现完全相同。可用时固定 provider、限制 fallback，并始终保存返回的 model/provider/generation ID。无法固定时应按可见路由分层分析；发现不稳定只说明测量环境不稳定，不能单凭计数波动声称幕后一定混用了多个模型。[Provider Routing](https://openrouter.ai/docs/guides/routing/provider-selection)

不要把较大的恒定偏移先验上判坏：隐藏 system/template 可能有较大开销。更应该关心跨样本残差是否稳定，且新的留出样本是否仍能被解释。

**11. P1：原生计数、请求预算与证据记录需要统一管理**

应读取并保存完整 usage details、response cache headers、generation ID、实际输入、请求参数、finish_reason、路由信息和时间。官方提供 `/generation` 查询 native token 及请求元数据，可作为按需诊断路径。[Generation Metadata](https://openrouter.ai/docs/api/api-reference/generations/get-generation)

不要把 `native_tokens_prompt` 缺失时的 `tokens_prompt` 静默当作同一种测量。计数来源与语义需要明确标记；拿不到可信原生计数时，本轮应降级而不是凭相近数值继续评分。

`max_calls` 当前统计逻辑探针数，而 `client.py:114` 允许每个探针进行多次 HTTP 尝试；因此它不是实际请求或费用硬上限。应分别限制与记录实际尝试数、成功生成数、输入/输出 token、金额、墙钟时间和重复测量开销。超时后可能已产生费用，重试也应记账。

当前 JSON 丢弃了 nonce、完整输出、完整 usage、provider/请求 ID 等关键资料，无法对历史结果换算法重算。保存脱敏的完整实验记录和文件版本；不保存 Authorization/API key。报告应该能回答“为什么排除第二候选”，而不是只有分数。

**12. 校准数据未达到验收标准**

现有 `data/calibration_real.json`：7 个已知模型的最终 verdict 全部为 unknown；其中 5 个被标记 hit=true。`scripts/calibrate_real.py:40` 将候选排名第一也算命中，因此这个 hit 指标不能说明最终结论成功。Kimi 的 top 为 DeepSeek；MiMo 七家均分（top confidence 为 1/7），首个候选是 Qwen。

这份记录缺少代码版本、原始观测、参考版本与运行时间，不能据此量化当前工作区的线上准确率，更不能推断修复后表现；它只能说明现有证据不足以验收。

`data/calibration_free.json` 的 5 条记录全部为 429，没有成功的库外模型拒识验证。429 不是 unknown tokenizer 的检测成功。

本次执行 `.venv/bin/python -m pytest -q`，结果 11 passed。mock 使用本地被测后端产生“真值”，给 nonce/template 固定增加 15 token，且 completion 被设为规范重编码长度。它验证了理想条件下流程能工作，但无法证明本地后端正确、真实 nonce 可消除、路由稳定或 unknown 拒识可靠。

第一版必要验证集：官方 tokenizer 独立对照、同厂多版本、跨厂复用、库外 tokenizer、少候选/单候选、样本缺失、近似平局、provider 切换、特殊串普通文本处理、hidden reasoning、预算不足。探针选择和阈值拟合所用数据应与最终评估分开。

分别报告 Top-1 排名准确率、被接受结论的准确率、覆盖率/弃权率、库外误接纳率与同家族跨厂商误归因率；不要把这些揉成一个 hit。

**13. 建议的第一版流程**

1. 建立版本化的 tokenizer 参考快照，与官方实现离线比对；生成候选 × 探针矩阵，列出本来无法区分的实现。
2. 对目标端点做能力与稳定性检查：原生 prompt count、缓存状态、参数支持、可见 provider、重复基线。
3. 执行少量普通文本探针，按稳健残差筛选候选；允许拟合分层模板偏移。
4. 对剩余候选选择新增信息最多的探针，必要时启用经过控制验证的特殊串。
5. 保留少量独立探针做确认；候选选择后不能只用同一批样本宣称高可靠性。
6. 输出本轮兼容 tokenizer 集合、被排除候选及理由、库外/不可观测/不稳定状态；单列厂商归属线索及其局限。
7. 按需开启输出侧或 token/logprobs 交叉验证；增加的是明确证据，不是固定权重的“第三信号”。

12 次快速模式的示例预算：2 次重复基线、4 次粗筛、4 次针对性探针、2 次留出确认。这是一个可调整的初始分配，不是保证达到可识别性的公式。实际额度需计入重试；候选无法区分时应停止并返回集合。

可以先保持现有 fetch/list/detect；研究场景更有价值的后续能力是离线 `profile/compare`、独立 `evaluate`、基于保存记录的 `replay`。是否实现为新子命令不是首要问题，功能应存在。

**14. 建议的输出示例（示意，非本次检测结果）**

```json
{
  "status": "ambiguous",
  "tokenizer_match": {
    "compatible_profiles": ["profile-A@revision", "profile-B@revision"],
    "scope": "observed probes only",
    "accepted_observations": 10,
    "residual_summary": "within calibrated measurement noise",
    "unresolved_reason": "profiles indistinguishable under current endpoint policy"
  },
  "attribution": {
    "known_users": ["model-A", "model-B"],
    "vendor_conclusion": "not established",
    "weight_identity": "not tested"
  },
  "provenance": {
    "reference_manifest": "manifest-hash",
    "probe_set": "version-hash",
    "raw_observations": "saved locally"
  }
}
```

**15. “各种方式识别”的功能补充**

建议按端点能力启用以下方法，而非对每个模型固定执行全部方法：

| 方法 | 可获得的证据 | 使用条件与限制 |
|---|---|---|
| 输入 count 差分 | 普通文本的分词长度指纹 | 主路径；要求原生计数和稳定的处理链 |
| 针对候选生成判别串 | 放大候选间可测差异 | 本地计算真实串的编码，不能假定重复串 token 数线性相加 |
| 特殊 token / added-token 对照 | 特殊标记处理行为 | 需验证用户输入能否触发特殊识别 |
| logprobs / top_logprobs | 输出 token 内容、字节与候选词元支持情况 | 端点实际返回时才可用；分块不是 token，采样概率不是 tokenizer 唯一指纹 |
| logit_bias 的 token ID 探测 | 指定 ID 与输出文本的对应关系 | 选择在候选中映射不同的普通词元 ID；先确认参数被支持且生效，未生成目标词元不能直接排除候选 |
| 可见输出重编码 | 与候选 tokenizer 的辅助一致性 | 处理 reasoning、隐藏控制 token、非规范 token 路径与截断 |
| model / endpoint metadata | 平台声明的 tokenizer 分类、模板类别、支持参数 | 用于排序和请求配置，不作为独立归属证明 |
| 能力与风格行为 | 训练/后训练的弱旁证 | 单独报告；自称厂商、口癖、答题风格容易受 system prompt 和微调影响 |

OpenRouter 官方支持按 token ID 的 `logit_bias` 以及 `logprobs/top_logprobs` 参数，但具体 provider 能力不同。可用时使用 `provider.require_parameters=true` 并检查响应，避免参数被静默忽略。新信号应有独立控制实验与预算，不应仅因接口接受参数就认为测量成功。[API Parameters](https://openrouter.ai/docs/api_reference/parameters)、[Provider Routing](https://openrouter.ai/docs/guides/routing/provider-selection)

**16. 改进优先级与验收门槛**

| 优先级 | 工作 | 完成标准 |
|---|---|---|
| P0 | 拆分 tokenizer 匹配与归属 | 同词表跨厂商例子不会输出唯一厂家 |
| P0 | 移除 nonce 测量污染 | 本地对照对应真实发送内容，固定输入可重复 |
| P0 | 参考 tokenizer 官方一致性 | Kimi 等后端通过独立官方实现对照 |
| P0 | 重写接受/拒识/歧义规则 | 单候选 0 分拒识，高匹配平局返回集合 |
| P1 | 版本化参考与原始实验记录 | 结果可追溯、可离线重算 |
| P1 | 选择有信息量的探针 | 每次额外请求有明确区分目标 |
| P1 | 校准与留出评估 | 单独报告误接纳与弃权，不以调低阈值代替验证 |
| P2 | 输出侧、logprobs 与行为辅助线索 | 明确前提、依赖与弱证据级别 |

应先完成 P0 再重新做小规模在线校准。当前证据支持继续开发这个研究工具，但不支持将它的现有 confidence 或 family verdict 用作匿名模型归属的可靠结论。
