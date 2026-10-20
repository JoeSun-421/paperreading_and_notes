# Phi-4-Reasoning-Vision-15B：位置编码分析

## 资料与结论

- 主要资料：`phi-4-reasoning-vision-15b-technical-report.pdf`。
- 原文：[Microsoft Technical Report](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/03/Phi-4-reasoning-vision-15B-Tech-Report.pdf)；[官方 model config](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B/blob/main/config.json)。
- 结论：语言主干使用标准全维度 RoPE，`rope_theta=500,000`，`partial_rotary_factor=1.0`，没有在 config 中设置 rope scaling。视觉部分采用 SigLIP-2 + MLP projector 的 mid-fusion，公开 Technical Report 没有展开视觉 encoder 的位置编码细节。

## 语言主干

官方 config 给出：

```text
max_position_embeddings       = 32,768
tokenizer_model_max_length     = 16,384
rope_theta                     = 500,000
partial_rotary_factor          = 1.0
rope_scaling                   = null
num_attention_heads            = 40
num_key_value_heads            = 10
```

`partial_rotary_factor=1.0` 表示 Q/K 的 rotary 子空间覆盖完整 head dimension，而不是像 DeepSeek V4 或 GLM-5 那样只旋转最后 64 维。`rope_theta=500,000` 比经典 10,000 更大，通常意味着更慢的相位变化和更适合长序列的频率范围。

不过 config 同时出现 32,768 的 max position 和 16,384 的 tokenizer model max length。前者是模型位置模块上限，后者是 tokenizer/部署默认值，不能简单把两者当成同一个数。

## 多模态 token 的位置

Phi-4-Reasoning-Vision-15B 是 mid-fusion：SigLIP-2 把图像转换成 visual tokens，MLP projector 把它们投影到语言 embedding space，再注入语言模型序列。语言模型随后对混合序列执行 causal processing。

从公开资料可以确认 visual tokens 被插入语言模型，但 Technical Report 没有明确给出：

- visual token 是否复用语言 RoPE 的连续 position IDs；
- 图像 patch 是否有单独的二维 RoPE/absolute embedding；
- 多图输入的 image boundary 和 position reset 规则。

所以不能因为它使用 SigLIP-2 就自动断言视觉侧也使用某一种 RoPE。

## 长上下文训练

报告的训练 recipe 包含 Long Context & RAI 阶段，max sequence length 从 8,192 扩展到 16,384。这个阶段说明模型通过长序列训练适应更长输入，但公开 config 没有启用 YaRN 或其他显式 rope scaling。

因此它的长上下文策略更接近：

```text
larger rope_theta + continued long-context training
```

而不是推理时临时把 RoPE factor 改成更大的值。

## 对自己实现的启发

基础版本可按完整 RoPE 写成：

```python
q = apply_rope(q, position_ids, theta=500_000)
k = apply_rope(k, position_ids, theta=500_000)
```

多模态实现的关键不在 RoPE 函数本身，而在 visual tokens 拼入文本序列后如何生成 position IDs。需要在 processor、projector 和 language model 三处保持一致。

## 披露边界

语言主干的 theta、rotary factor 和位置上限有官方 config 证据；视觉 patch 的位置编码机制和完整 position ID 构造未在公开 Technical Report 中说明，应标记为未披露。
