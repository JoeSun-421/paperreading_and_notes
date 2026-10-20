# GLM-5 / GLM-5.3：位置编码分析

## 资料与结论

- 主要资料：`glm-5-technical-report.pdf`。
- 补充资料：`glm-5.3-flash-model-card.md`。
- 原文：[GLM-5 Technical Report](https://arxiv.org/abs/2602.15763)；[GLM-5.3 官方发布页](https://z.ai/blog/glm-5.3)；[GLM-5 官方配置](https://huggingface.co/zai-org/GLM-5/blob/main/config.json)。
- 结论：GLM-5 论文正文重点讲 DSA、MLA 和长上下文训练，没有详细推导 RoPE；官方 checkpoint configuration 则明确显示采用 MLA 的“部分 NoPE + 部分 RoPE”布局。

## 配置中公开的参数

官方 `config.json` 给出：

```text
max_position_embeddings = 202,752
qk_head_dim              = 256
qk_nope_head_dim         = 192
qk_rope_head_dim         = 64
rope_theta               = 1,000,000
rope_interleave          = true
indexer_rope_interleave  = true
```

Q/K 的 head 维度可以拆成：

```text
qk = [NoPE 192 dimensions | RoPE 64 dimensions]
```

这和完整 RoPE 不同：只有 64 维携带显式旋转位置信息，192 维保留内容相似度通道。

## MLA 中的 NoPE/RoPE 分工

对位置 `t` 的 query 和位置 `i` 的 key，可以抽象成：

```text
q_t = [q_nope_t, RoPE(q_rope_t, t)]
k_i = [k_nope_i, RoPE(k_rope_i, i)]
```

点积可以理解为两部分：

```text
score(t, i)
  = q_nope_t · k_nope_i
  + RoPE(q_rope_t, t) · RoPE(k_rope_i, i)
```

第一项只比较内容，第二项通过旋转后的相位差提供相对位置信号。这样模型不会让所有 Q/K 维度都承受长上下文相位变化。

`rope_interleave=true` 表示旋转维度采用交错布局，具体 tensor slicing 仍应以官方 Transformers 实现为准，不能只根据字段名手写一个不同的布局。

## DSA 与位置编码的关系

GLM-5 使用 DeepSeek Sparse Attention，通过 indexer 选择重要 token，再对选中的内容做稀疏 attention。位置编码在这里有两个角色：

1. 主 attention 的 64 维 RoPE 保持 query/key 的相对位置线索；
2. indexer 也有自己的 `indexer_rope_interleave` 配置，让重要性检索不完全丢失序列位置。

DSA 解决的是“看哪些 token”，RoPE 解决的是“这些 token 位于什么相对位置”。稀疏选择减少计算量，但本身不是位置编码。

## 长上下文训练策略

报告描述 GLM-5 的 context extension 训练阶段：

```text
32K  -> 1T tokens
128K -> 500B tokens
200K -> 50B tokens
```

模型配置的最大位置为 202,752，与论文中的 200K stage 基本对应。把 `rope_theta` 调大只是参数变化，真正获得长上下文能力还需要在长序列数据上继续训练。

## GLM-5.3 的关系

GLM-5.3 官方发布说明称它沿用 GLM-5.2 的 base model，GLM-5.3-Flash 则是 GLM-5 系列中的混合稀疏/线性 attention 模型。当前公开的 GLM-5.3-Flash model card 重点说明架构能力和部署信息，并没有重新公开一套位置编码公式。

因此本笔记的证据层级是：GLM-5 的具体 RoPE 维度和 theta 来自官方 checkpoint config；GLM-5 的长上下文阶段和 DSA 来自 Technical Report；GLM-5.3 的基座关系来自官方 release/model card；不把 GLM-5 的所有参数自动断言为 GLM-5.3-Flash 的每层实现。

## 对自己实现的启发

标准实现可以改为：

```python
q_nope, q_rope = q.split([192, 64], dim=-1)
k_nope, k_rope = k.split([192, 64], dim=-1)
q_rope = apply_rope(q_rope, position_ids, theta=1_000_000)
k_rope = apply_rope(k_rope, position_ids, theta=1_000_000)
q = torch.cat([q_nope, q_rope], dim=-1)
k = torch.cat([k_nope, k_rope], dim=-1)
```

实际工程还要实现 DSA indexer、KV cache 和 interleave layout；只修改你项目的 `rope_theta` 不能复现 GLM-5。

## 披露边界

论文正文没有完整展开 RoPE 公式，因此本篇把 `qk_nope_head_dim`、`qk_rope_head_dim`、`rope_theta` 和 interleave 作为官方配置证据，其余实现细节应查看官方代码。
