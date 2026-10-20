# Gemma 4：位置编码分析

## 资料与结论

- 主要资料：`gemma-4-technical-report.pdf`，尤其是 Architecture 部分。
- 原文：<https://arxiv.org/abs/2607.02770>
- 结论：Gemma 4 不是在所有层、所有维度上使用同一种 RoPE，而是把注意力层类型、RoPE 频率和旋转维度比例一起设计。

## 文本主干的实现

报告明确写出：

1. local sliding-window attention 层使用普通 RoPE。
2. global attention 层使用 `p-RoPE`，其中 `p=0.25`。
3. local 层的 RoPE 频率 base 为 `10k`，global 层为 `1M`。
4. E2B 的 local:global attention block 比例为 4:1，其余模型大多为 5:1。

标准 RoPE 可以理解为对 Q、K 的二维坐标对做旋转。对于位置 `m` 的一对维度：

```text
[q'_{2i}, q'_{2i+1}] = [q_{2i}, q_{2i+1}] R(m * theta_i)
[k'_{2i}, k'_{2i+1}] = [k_{2i}, k_{2i+1}] R(n * theta_i)
```

其中 `R` 是二维旋转矩阵。Q 和 K 做点积后，角度差主要依赖 `m-n`，所以注意力可以感知相对位置。

`p-RoPE` 的核心不是改变 token embedding，而是只保留一部分维度的旋转，把其余维度作为 NoPE/近似位置无关的语义通道。这里 `p=0.25` 应理解为只保留约四分之一的 rotary 维度；报告将它用于 global attention，以减少 global KV cache 的位置相关存储开销。

报告给出的工程组合是：

```text
local attention  : full RoPE, theta = 10,000
global attention : p-RoPE, p = 0.25, theta = 1,000,000
```

同时，global 层复用 key 作为 value，并在部分模型上共享 KV cache。报告称这些组合最多可以把 global KV cache footprint 降低 37.5%。这里的节省不应全部归因于 p-RoPE：它还来自 K/V 复用、KV sharing 和 local/global 层比例。

## 多模态位置

Gemma 4 还有三套位置处理逻辑：

- vision encoder：axial 2D-RoPE 加 2D absolute positional embedding；
- 12B encoder-free vision：把图像 patch 的二维坐标 embedding 直接加到 patch 表示上；
- audio：原始 40ms 音频块直接投影到语言模型空间，报告称不额外添加 positional encoding。

因此不能把 Gemma 4 的位置编码简单概括成“使用 RoPE”。文本、视觉、无 encoder 的图像和音频路径使用了不同的位置信号。

## 为什么这样设计

长上下文的矛盾是：global attention 能看到远处 token，但 KV cache 和 attention 计算昂贵；local attention 便宜，却不能直接建立远距离联系。Gemma 4 用层间交替解决范围问题，再用不同 RoPE 配置解决位置和内存问题：

- local 层保留完整 RoPE，负责近邻和高分辨率的序列位置关系；
- global 层用更大的 theta，减慢相位变化，适合更长距离；
- global 层只旋转部分维度，给语义内容留下不旋转通道；
- KV sharing 和 key-as-value 进一步压缩缓存。

## 对你实现的启发

你在 `transformer_block.py` 中如果实现基础 RoPE，可以先做“所有层、全部维度”的版本。Gemma 4 的扩展需要额外传入：

```text
attention_type = local | global
rope_theta = 10_000 | 1_000_000
partial_rotary_factor = 1.0 | 0.25
```

并且必须保证 Q/K 的旋转维度、KV cache 保存格式和推理时的位置索引完全一致。p-RoPE 不是把所有角度乘 0.25，而是只对指定的 rotary 子空间应用旋转。

## 披露边界

报告公开了层类型、theta 和 p 值，但没有公开完整的生产 kernel、每个 head 的具体频率布局以及所有 multimodal encoder 的源码。因此“p=0.25 的精确维度排列”应以官方 checkpoint/config 或实现代码为准，不能仅凭报告复刻全部细节。

