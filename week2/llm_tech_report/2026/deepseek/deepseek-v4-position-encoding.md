# DeepSeek V4：位置编码分析

## 资料与结论

- 主要资料：`deepseek-v4-technical-report.pdf`，补充资料为 `deepseek-v4-technical-documentation.pdf`。
- 原文：[DeepSeek 官方公告](https://deepseek.com/en/news/v4-preview/)；[arXiv Technical Report](https://arxiv.org/abs/2606.19348)。
- 结论：V4 仍使用 RoPE，但只旋转最后 64 个维度，并为压缩 KV 和 core attention output 设计了额外的位置处理。

## 为什么普通 RoPE 不够

DeepSeek V4 的目标是百万 token 上下文。它采用 CSA 和 HCA：先把多个 token 的 KV 压缩成 entry，再执行稀疏或稠密 attention。压缩 entry 不再简单对应一个原始 token 位置，因此位置相位、压缩区间和输出位置必须一起处理。

## 部分 RoPE 的具体实现

报告的 Partial Rotary Positional Embedding 小节明确给出：

1. CSA 和 HCA 的 query vector、KV entry vector 只在最后 64 个维度上应用 RoPE。
2. KV entry 同时作为 key 和 value，core attention output 会携带由 KV 加权和带来的绝对位置成分。
3. 对每个 core attention output 的最后 64 维再应用位置 `-i` 的 RoPE。
4. 这样 output 中的位置信号重新表现为 query 与 KV entry 的相对距离。

抽象地表示为：

```text
q_rot = RoPE(q[..., -64:], position=t)
k_rot = RoPE(kv[..., -64:], position=i)
o_rot = RoPE(o[..., -64:], position=-i)
```

`-i` 不是普通的“给第 i 个 token 加位置”，而是用于抵消压缩 KV entry 带入的绝对相位。

## 压缩中的另一种位置信号

KV 压缩本身还使用 learnable positional biases `B`。因此 V4 的位置处理分成两层：

- RoPE：在 Q、KV 和 output 的指定子空间中提供相对相位。
- learnable positional bias：在把一个区间压成一个 KV entry 时区分区间内位置。

标准 attention 可写成：

```text
score(t, i) = RoPE(q_t, t) · RoPE(k_i, i)
```

V4 则更接近：

```text
compressed_kv_i = Compress(KV interval_i, positional bias)
score(t, i) = RoPE(q_t, t) · RoPE(compressed_kv_i, i)
output_rot(t, i) = RoPE(output(t, i), -i)
```

## 存储和硬件实现

报告明确说明：RoPE dimensions 用 BF16 保存，其余 KV dimensions 用 FP8 保存。旋转维度承载相位，量化误差会直接影响位置关系；内容维度则可以进一步压缩。由此可见，位置编码策略已经和 KV cache 的数据布局绑定在一起。

## 长上下文的关键点

V4 支持 1M context，但主要突破来自 CSA/HCA 的压缩和稀疏选择，而不是简单把 RoPE theta 调大。分析时应把三件事分开：RoPE 的相对位置、positional bias 的 KV 压缩、CSA/HCA 的计算和缓存压缩。

## 对自己实现的启发

部分 RoPE 不是把整个 hidden state 乘一个旋转矩阵，而是拆分 rotary/pass-through 子空间：

```python
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
q_rot = apply_rope(q_rot, position_ids)
k_rot = apply_rope(k_rot, position_ids)
q = torch.cat([q_rot, q_pass], dim=-1)
k = torch.cat([k_rot, k_pass], dim=-1)
```

复现 V4 还需要在压缩 attention output 上执行负位置旋转，并为每个压缩窗口保存 entry position。

## 披露边界

报告公开了最后 64 维、`-i` output rotation、learnable positional bias 以及 BF16/FP8 混合存储；压缩权重、具体 kernel 和所有 layer 的完整配置仍应以官方实现为准。
