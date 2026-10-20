# 2026 LLM 位置编码总总结

整理日期：2026-08-29

本总结基于本目录中的 Technical Report、Model Card、System Card 和官方 model config。最重要的阅读原则是：长上下文能力不等于某一种 RoPE；同一个模型的文本、视觉、音频和压缩 attention 也可能使用不同的位置机制。

## 一、先理解标准 RoPE

Transformer 的 self-attention 本身对 token 顺序没有感知。如果输入 token 发生置换，只看内容的 attention 可能得到相似结果，所以需要把位置注入 Q/K 或 attention score。

RoPE 不把位置向量直接加到 token embedding，而是对 Q 和 K 的二维通道对做旋转。对位置 `m` 的 query 和位置 `n` 的 key：

```text
q'_m = R(m * theta_i) q_m
k'_n = R(n * theta_i) k_n
```

其中每个二维通道对有自己的频率 `theta_i`。旋转后做点积时，相位包含 `m-n`，因此模型可以学习相对位置。

常见的实现流程是：

```text
Q, K projection
      ↓
split rotary dimensions and pass-through dimensions
      ↓
apply cos(position * frequency), sin(position * frequency)
      ↓
attention score = Q_rot K_rot^T + Q_nope K_nope^T
```

工程上需要同时保持三件事一致：position IDs、rotary channel layout、KV cache 中的位置状态。

## 二、模型对比表

| 模型 | 显式位置机制 | 关键公开参数/做法 | 长上下文路线 | 证据强度 |
|---|---|---|---|---|
| Gemma 4 | local full RoPE + global p-RoPE；vision 用 axial 2D-RoPE/2D absolute PE | global `p=0.25`、theta 1M；local theta 10K | local/global attention 交替、KV sharing、key-as-value | Technical Report 明确 |
| Qwen3.5-Omni | TM-RoPE + temporal/spatial IDs + timestamp tokens | 音频每 160ms 一个 temporal ID；视频按真实时间；H/W IDs | 通过真实时间坐标和显式 timestamp 提升多模态长程建模 | Technical Report 明确 |
| Qwen3.8 | partial mRoPE，主要由 Gated Attention 使用 | rotary dim 64、theta 10M、partial factor 0.25、sections `[11,11,10]` | 原生 262K；YaRN factor 4 扩到 1M | Model Card/config 明确 |
| DeepSeek V4 | partial RoPE + learnable positional bias | Q/KV 最后 64 维；core output 用位置 `-i`；RoPE 维度 BF16 | CSA/HCA 压缩 KV，RoPE 与压缩共同设计 | Technical Report 明确 |
| Kimi K3 | MLA 显式 NoPE；KDA 递归隐式位置 | KDA 的 gate/decay/state 携带顺序；MLA 不加 RoPE | 不需要为 MLA 调 RoPE 或 YaRN；仍受状态和系统限制 | Technical Report 明确 |
| GLM-5 | MLA 的 192 维 NoPE + 64 维 RoPE | theta 1M、max position 202752、interleave；indexer 也有 RoPE 配置 | DSA 稀疏选择 + 32K/128K/200K 长上下文训练 | config 明确，论文部分明确 |
| Phi-4-Reasoning-Vision-15B | 语言主干 full RoPE | theta 500K、partial factor 1.0、rope scaling null | long-context training；视觉位置构造未披露 | config 明确 |
| Seed1.8 / Seed2.0 | 未披露 | Model Card 没有 theta、rotary dim 或 position ID 规则 | 报告能力和评测，不足以证明底层位置方案 | 未披露 |
| Claude Opus 4.8 | 未披露 | System Card 主要是安全与能力评测 | 长上下文机制未展开 | 未披露 |
| GPT-5.6 | 未披露 | 官方资料主要是 API、缓存、工具和能力 | context window 与 prompt caching 不是位置编码 | 无架构报告 |

## 三、最值得比较的四条路线

### 1. 全 RoPE，但改变频率范围：Phi

Phi 的做法接近传统 decoder-only Transformer：完整 rotary factor 加到 Q/K，使用更大的 `rope_theta=500K`，再通过 8K/16K 长上下文训练适应更长序列。它简单、容易复现，但没有公开视觉 token 的独立空间位置处理。

### 2. 部分 RoPE：Gemma、Qwen、DeepSeek、GLM

partial RoPE 的基本思想是把 Q/K 拆成两组：一组旋转、提供位置关系；另一组不旋转、保留更稳定的内容相似度。这样可以降低旋转相关的缓存和相位负担，也允许模型把部分通道用于语义信息。

不同模型的差异在于：

- Gemma 用层类型区分 local 与 global，并在 global 层使用 p-RoPE；
- Qwen3.8 用 mRoPE 处理多维视觉位置，并只旋转一部分通道；
- DeepSeek V4 把 partial RoPE 延伸到压缩 KV 和 output 的负位置旋转；
- GLM-5 在 MLA 中显式拆出 NoPE 192 维和 RoPE 64 维，并让 DSA indexer 也保留位置线索。

### 3. 多维位置：Qwen3.5-Omni、Qwen3.8、Gemma vision

多模态 token 的“位置”不是单一序列下标：视频需要时间、行、列，图像需要二维坐标，音频需要真实时间。多维 RoPE 的抽象是：

```text
position_ids = [time_id, height_id, width_id]
```

不同位置维度被分到不同 rotary sections。Qwen3.5-Omni 还使用 timestamp text token 辅助学习真实时间；Qwen3.8 的 config 则直接公开 mRoPE sections 和 interleaving。

### 4. NoPE 与隐式位置：Kimi K3

Kimi K3 的关键不是“删除位置编码”，而是重新安排位置职责：KDA 的递归状态和衰减天然依赖 token 顺序，MLA 只负责全局内容交互并使用 NoPE。这个方案避免了给 MLA 重新设置 RoPE frequency base，但代价是必须正确实现 KDA 的递归/扫描和衰减稳定性。

## 四、theta、partial factor 和 YaRN 分别是什么

### theta

`rope_theta` 是 RoPE 频率基数。增大它通常会让部分频率的相位变化更慢，从而减少超长位置上的快速相位漂移。但增大 theta 不是自动获得长上下文能力，仍需要长序列训练、合适的数据和稳定的缓存实现。

本目录中可直接对比：Phi `500K`、GLM `1M`、Qwen3.8 `10M`；Gemma 还按 local/global 层使用 `10K/1M`。

### partial factor

partial factor 决定多少 Q/K 通道进行旋转。它不是把 position ID 或 theta 乘一个比例。例如 factor `0.25` 通常意味着 rotary 子空间只占 head dimension 的一部分，剩余通道保持 NoPE。

实现时必须明确是“前若干维”还是“后若干维”、是否交错排列，以及 Q/K 是否使用相同切分。Qwen、DeepSeek 和 GLM 的官方实现布局并不应互相猜测替换。

### YaRN

YaRN 是长上下文扩展方法，不等于普通 RoPE。它会重新缩放/插值位置频率，使推理长度超过训练长度。Qwen3.8 model card 给出原生 262K 到 1M 的 factor 4 配置，并提醒静态 YaRN 可能影响短文本，因此只应在需要长上下文时启用。

## 五、从这些报告得到的工程规律

1. 先定义位置的语义，再选编码：文本是一维顺序，图像是二维坐标，视频/音频还需要真实时间。
2. 位置编码必须和 attention 结构共同设计：稀疏 attention、线性 attention、压缩 KV 和 MLA 都会改变位置状态如何保存。
3. 长上下文的瓶颈不只是相位：KV cache 大小、attention FLOPs、压缩误差、量化误差和长序列训练同样重要。
4. 旋转维度可以和内容维度分开：partial RoPE/NoPE 子空间让模型保留更稳定的语义通道。
5. 不能从 benchmark 或 context window 反推 RoPE：Seed、Claude、GPT-5.6 的公开资料就是这种情况。
6. cache 复用时必须保存位置语义：直接复用带有旧位置的 K/V，可能导致相位错位；任何 cache relocation 都要检查 position IDs 和 RoPE 重算规则。

## 六、建议你的阅读和复现顺序

1. 先用自己的 `transformer_block.py` 实现标准一维 RoPE，并用小张量验证 Q/K 在不同位置的相位变化。
2. 增加 partial RoPE：比较 rotary/pass-through 两组维度。
3. 复现 GLM 的 `[NoPE, RoPE]` 分块，理解 attention score 如何同时包含内容项和相对位置项。
4. 再读 Gemma 的 local/global p-RoPE，观察层间不同 theta。
5. 最后研究 Qwen 的 mRoPE 和 DeepSeek 的压缩 KV output rotation；这些需要多维 position IDs、缓存和特殊 attention。
6. 把 Kimi K3 作为另一条路线单独实现，不要把它简化为“把 RoPE 删除”。

## 七、逐篇笔记

- [Gemma 4](google/gemma-4-position-encoding.md)
- [Phi-4-Reasoning-Vision-15B](microsoft/phi-4-reasoning-vision-15b-position-encoding.md)
- [Qwen3.5-Omni](alibaba/qwen3.5-omni-position-encoding.md)
- [Qwen3.8](qwen/qwen3.8-position-encoding.md)
- [DeepSeek V4](deepseek/deepseek-v4-position-encoding.md)
- [Kimi K3](kimi/kimi-k3-position-encoding.md)
- [GLM-5 / GLM-5.3](glm/glm-5-position-encoding.md)
- [Seed1.8](bytedance/seed1.8-position-encoding.md)
- [Seed2.0](bytedance/seed2.0-position-encoding.md)
- [Claude Opus 4.8](anthropic/claude-opus-4.8-position-encoding.md)
- [GPT-5.6](openai/gpt-5.6-position-encoding.md)
