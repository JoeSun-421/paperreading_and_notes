# Seed1.8：位置编码分析

## 资料与结论

- 主要资料：`seed1.8-model-card.pdf`。
- 原文：[Seed1.8 Model Card](https://arxiv.org/abs/2603.20633)；[ByteDance 官方发布页](https://seed.bytedance.com/en/blog/official-release-of-seed1-8-a-generalized-agentic-model)。
- 结论：这是一份能力、数据、安全和部署导向的 Model Card，不是完整架构 Technical Report。公开文档没有明确披露 RoPE、absolute position、relative position 或具体 position ID 实现。

## 能确认的内容

Seed1.8 支持文本和图像输入，并强调 generalized agent、搜索、代码、GUI 和真实世界工作流。Model Card 讨论了视觉 token 数量和推理效率，但没有给出语言主干 attention 的 head dimension、RoPE theta、rotary dimension 或长上下文 scaling 参数。

这意味着我们只能确认模型需要处理多模态序列，不能从“支持图像/视频”推出它使用 2D RoPE、mRoPE 或视觉 absolute embedding。

## 位置编码问题应该如何记录

对 Seed1.8，正确的研究记录是：

```text
text position encoding : 未披露
vision position encoding: 未披露
RoPE theta              : 未披露
long-context scaling    : 只披露能力/部署层面，未披露公式
```

不能把 GLM、Qwen 或 ByteDance 其他视觉模型的配置复制到 Seed1.8。模型卡的 benchmark 和上下文描述只能说明结果，不足以证明底层位置编码。

## 对分析和复现的启发

如果希望进一步研究 Seed1.8，需要寻找官方 checkpoint/config 或代码；拿到模型配置后优先检查 `rope_theta`、`max_position_embeddings`、`rope_scaling`、`position_ids` 和 multimodal processor。当前公开 Model Card 不足以复现其位置处理。

## 披露边界

本文把“未披露”作为结论，而不是猜测。Seed1.8 的公开资料适合分析产品能力和部署取舍，不适合证明 RoPE 的具体原理。
