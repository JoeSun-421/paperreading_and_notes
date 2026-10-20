# Qwen3.5-Omni：位置编码分析

## 资料与结论

- 主要资料：`qwen3.5-omni-technical-report.pdf`。
- 原文：<https://arxiv.org/abs/2604.15804>
- 结论：Qwen3.5-Omni 的公开创新重点不是提出一个全新的文本 RoPE，而是处理多模态流中的时间、空间和模态边界。其核心是 TM-RoPE、显式 timestamp token、动态 temporal ID 和连续的 modality position numbering。

## TM-RoPE 要解决什么

视频和音频不是普通的 token 序列。两个视觉 patch 在 token 序列中相邻，不代表它们在真实时间上相邻；不同帧率和音频采样率也会使 token 数量与真实时间不一致。

Qwen3.5-Omni 延续 TM-RoPE，让音频/视频 token 使用 temporal position ID。报告指出，单纯把绝对时间直接变成 position ID 有两个问题：

1. 长视频会得到很大且稀疏的 temporal ID，长程时间关系变难学习；
2. 为了覆盖不同 frame rate，需要大量且均匀的训练样本，数据构造成本高。

## 公开的具体实现

报告给出的设计可以整理为：

```text
audio : 每 160 ms 对应一个 temporal ID
video : 根据真实 timestamp 生成单调递增的 temporal ID
image : 使用 height ID 和 width ID 表示二维空间
modalities : 后一个模态从前一个模态最大 position ID + 1 开始编号
```

这说明它不是简单给所有 token 使用一个一维 `position_ids`。对多模态输入，位置至少包含：

```text
text/audio/video token -> temporal or sequence position
image/video patch      -> height position + width position
```

在实现层面，这类位置通常需要多维 position IDs，并由不同维度的旋转频率分别作用在 Q/K 的子空间；但报告没有把完整 RoPE kernel 或频率切分表全部展开，所以不能从论文 alone 推断所有实现参数。

## 从绝对时间到显式 timestamp

Qwen3.5-Omni 还在每个 video 或 audio-video temporal patch 前插入格式化的秒数文本，例如把时间编码成类似“某秒”的文本字符串。音频还会在随机间隔插入 timestamp。

这相当于同时使用两种位置信号：

1. TM-RoPE/temporal IDs：给注意力提供结构化的时间坐标；
2. timestamp token：让语言模型用它熟悉的文本 token 语义学习“时间码”。

第二种信号很重要，因为纯 RoPE 是连续旋转信号，模型不一定天然学会把稀疏绝对时间 ID 解释为“几分几秒”。显式 timestamp 把时间变成可读、可预测的离散文本表示。

## 为什么适合 streaming

报告称这种设计支持任意时长的 streaming input。原因不是 RoPE 本身可以无限外推，而是 temporal ID 根据真实时间动态产生，模态之间的编号保持连续，并且时间含义由 timestamp token 辅助表达。

要注意：这并不等于模型不受上下文长度限制。Qwen3.5-Omni 报告仍给出 256K 输入上下文，长时间输入最终仍需要上下文窗口、缓存和吞吐方面的工程控制。

## 对你实现的启发

基础语言模型只需要：

```python
position_ids = torch.arange(seq_len)
```

多模态模型则应考虑：

```text
position_ids.shape = [num_position_dimensions, batch, seq_len]
```

并在 attention 中让不同 position 维度作用于不同 RoPE 子空间。工程上要特别检查：padding、packed sequence、模态拼接和 FlashAttention 是否正确理解三维 position IDs。

## 披露边界

报告明确披露了 TM-RoPE 的用途、160ms 时间分辨率、H/W IDs、模态连续编号和 timestamp 策略；但没有在该报告中完整列出语言主干的 `rope_theta`、rotary dimension、每个 head 的频率切分和对应源码。因此本文只对这些已披露内容下结论，不把 Qwen3 或其他 Qwen 版本的 RoPE 参数移植到 Qwen3.5-Omni。

