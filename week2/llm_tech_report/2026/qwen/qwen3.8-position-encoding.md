# Qwen3.8：位置编码分析

## 资料与结论

- 主要资料：`qwen3.8-flash-next-technical-report.pdf`。
- 补充资料：`qwen3.8-27b-model-card.md`。
- 原文：[Qwen3.8-Flash-Next repository](https://github.com/QwenLM/Qwen3.8-Flash-Next)；[Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)；[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/config.json)。
- 结论：Qwen3.8 的混合架构把 Gated DeltaNet 和 Gated Attention 结合起来；公开配置显示 attention 侧使用部分、多维、交错的 RoPE，`rope_theta=10,000,000`，rotary factor 为 0.25。1M context 不是原生长度，而是基于 YaRN 的扩展配置。

## 混合 attention 与位置编码

Qwen3.8-27B model card 给出的 layout 是：

```text
16 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))
```

Gated DeltaNet 是线性/递归 attention，顺序由 recurrent state 和 gate 进入计算；Gated Attention 是标准全局 attention，依赖显式 position IDs/RoPE 处理 token 的位置关系。

因此它不是每一层都使用相同的 softmax attention + full RoPE，而是让显式 attention 子层主要承担全局位置关系。

## 官方配置中的 RoPE 参数

官方配置公开了以下字段：

```text
max_position_embeddings = 262,144
rope_theta              = 10,000,000
partial_rotary_factor   = 0.25
mrope_interleaved       = true
mrope_section           = [11, 11, 10]
rope_type               = default
```

Qwen3.8-27B model card 还明确列出 Rotary Position Embedding Dimension 为 64。`partial_rotary_factor=0.25` 表示只有部分 Q/K 子空间进行旋转，剩余维度不做 rotary；不是把 position ID 乘以 0.25。

`mrope_section=[11,11,10]` 表明多维 position IDs 被分配到不同 rotary 子空间。结合 `mrope_interleaved=true`，实现时必须遵循官方模型代码对多个位置维度和通道交错方式的定义。

## 长上下文与 YaRN

Qwen3.8-27B 原生上下文是 262,144 tokens，model card 建议超过该长度时使用 YaRN 扩展到 1M。公开示例给出的扩展配置包括：

```json
{
  "rope_type": "yarn",
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25,
  "factor": 4.0,
  "original_max_position_embeddings": 262144,
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10]
}
```

这体现了三层区别：

1. 原生模型配置：`rope_type=default`，支持 262K；
2. 长上下文推理配置：`rope_type=yarn`，通过 factor 4 扩展；
3. 实际效果：静态 YaRN 可能影响短文本，因此 model card 建议只在确实需要长上下文时覆盖配置。

## mRoPE 的直观理解

对于视频或图像 token，单个 token 不只拥有一条文本序列位置，还可能有时间、行、列三个坐标：

```text
position_ids = [temporal_id, height_id, width_id]
```

模型把这些坐标映射到不同 rotary sections，再把旋转后的 Q/K 用于 attention。这样同一个视觉 token 可以同时表达“第几帧、第几行、第几列”。

这和 Qwen3.5-Omni 的 TM-RoPE 属于同一类多模态位置问题，但 Qwen3.8 的公开 config 进一步给出了 mRoPE 的 section 和 interleaved 字段。

## 对自己实现的启发

你的基础 Transformer 当前可以先实现一维 RoPE。要接近 Qwen3.8，需要逐步增加：

```text
1. partial rotary：只旋转指定维度；
2. mRoPE：接受多个 position ID 维度；
3. interleaved sections：把位置维度映射到指定 rotary 通道；
4. YaRN：只在长上下文部署时覆盖 scaling config；
5. hybrid layer types：区分 Gated DeltaNet 与 Gated Attention。
```

最常见的错误是把 `mrope_section` 当成三个独立的 sequence length，或者把 YaRN factor 固化到所有输入。它们分别会破坏多模态位置布局和短文本性能。

## 披露边界

Qwen3.8 model card/config 对 rotary dimension、theta、partial factor、mRoPE sections 和 YaRN 参数有明确记录；Flash-Next technical report 重点讲新架构和效率。具体每个 layer 的 rotary 调用路径、视觉 token 的 position ID 构造和 kernel 实现，应以官方模型代码为最终依据。
