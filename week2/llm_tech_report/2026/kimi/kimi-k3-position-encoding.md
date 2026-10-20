# Kimi K3：位置编码分析

## 资料与结论

- 主要资料：`kimi-k3-technical-report.pdf`。
- 原文：[Kimi K3 Technical Report](https://arxiv.org/abs/2607.24653)；[官方技术博客](https://www.kimi.com/en/blog/kimi-k3)。
- 结论：Kimi K3 没有采用每层都给 Q/K 加 RoPE 的统一方案，而是把位置处理分工给 KDA 和 MLA：KDA 通过递归状态、forget gate 和 decay 隐式编码顺序，Gated MLA 层明确使用 NoPE。

## Hybrid Attention

每个 block 的结构是：

```text
3 x Kimi Delta Attention (KDA)
1 x Gated MLA
```

KDA 是线性/递归 attention，负责沿序列传递状态；Gated MLA 保留全局 token-to-token interaction，负责跨远距离读取内容。

## KDA 如何隐式表示位置

单个 KDA head 的核心递归可以抽象为：

```text
S_t = (I - beta_t * k_t k_t^T) * S_{t-1}
      + beta_t * k_t v_t^T
o_t = S_t^T q_t
```

其中 `S_t` 是处理到位置 `t` 的序列状态，`alpha_t`/`beta_t` 控制旧信息保留和新信息写入。每一步都基于前一步状态，所以 token 顺序已经进入计算路径；衰减越多，越早 token 对当前状态的影响越小。

报告明确称，Kimi K3 不使用显式 positional embedding，而是通过 KDA 的 recurrent gating 和 decay mechanism 隐式编码 positional information。这不是说 KDA 没有位置信息，而是位置信息不再由固定的正弦旋转注入。

## MLA 为什么使用 NoPE

报告明确写出：Kimi K3 对所有 MLA layers 使用 No Position Encoding；其 query 和 key 不施加显式位置编码。

全局 MLA 的主要作用变成根据内容建立全局交互，而 KDA 层负责根据顺序、recency 和递归状态进行位置敏感的信息流动。这样扩展 context 时不需要重新调 RoPE frequency base，也不需要应用 YaRN。

## NoPE 的准确含义

NoPE 不是给所有模型层都删除位置信息。Kimi K3 的准确表述是：

```text
KDA : implicit position through recurrent state and decay
MLA : explicit NoPE
```

如果把 KDA 错误实现成普通无位置 softmax attention，模型会失去顺序；如果把 MLA 擅自改成 RoPE，又不再是论文描述的架构。

## 长上下文含义

RoPE 长上下文扩展通常要考虑训练长度、推理长度、theta、频率插值和相位外推。Kimi K3 绕开了 MLA 部分的这条路径：KDA 的递归状态不依赖固定 RoPE 频率，MLA 不存显式位置相位，context 增长主要受递归状态、全局 attention、cache 和系统实现限制。

但这不等于天然无限上下文。递归状态有信息压缩和遗忘，MLA 仍有计算/缓存成本，模型依然要通过长上下文训练和评测验证实际能力。

## 对自己实现的启发

你当前的 `transformer_block.py` 是标准 Transformer block，默认每层 self-attention 使用 RoPE。若要模拟 Kimi K3，至少需要按层选择：

```python
if layer_type == "kda":
    x = kimi_delta_attention(x)  # 顺序来自递归状态
elif layer_type == "gated_mla":
    x = mla(x)                   # 不对 q/k 施加 RoPE
```

不能只把 `rope_theta` 改成很大的值来复现 K3；核心变化是 attention 机制和位置表达方式同时变化。

## 披露边界

报告明确说明了 KDA 的递归/衰减位置机制和 MLA 的 NoPE，但 KDA 的所有 kernel、并行扫描细节和完整 checkpoint 配置仍属于实现层信息。
