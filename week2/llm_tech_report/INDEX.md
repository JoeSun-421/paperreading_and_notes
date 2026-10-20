# 2026 LLM Technical Reports

整理日期：2026-08-29

本目录收录目前找到的、由模型厂商或其研究团队公开的一手资料。这里的“所有大厂”无法严格穷尽：许多闭源模型只公开模型卡或安全 System Card，并不公开完整架构报告；因此不把安全报告、博客或营销页面冒充 Technical Report。

## 已下载

| 公司 | 模型/文档 | 文档类型 | 本地文件 | 原始来源 | 位置编码阅读优先级 |
|---|---|---|---|---|---|
| Google DeepMind | Gemma 4 | Technical Report | `2026/google/gemma-4-technical-report.pdf` | [arXiv: Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770)；[Google model card](https://ai.google.dev/gemma/docs/core/model_card_4) | 高：搜索 `position`、`rotary`、`RoPE`、`context length` |
| Microsoft Research | Phi-4-reasoning-vision-15B | Technical Report | `2026/microsoft/phi-4-reasoning-vision-15b-technical-report.pdf` | [Microsoft Research PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/03/Phi-4-reasoning-vision-15B-Tech-Report.pdf) | 中：重点看文本/视觉 token 对齐及长上下文部分 |
| Alibaba / Qwen | Qwen3.5-Omni | Technical Report | `2026/alibaba/qwen3.5-omni-technical-report.pdf` | [arXiv: Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804) | 高：重点看多模态序列、时间/空间位置和文本位置编码 |
| Alibaba / Qwen | Qwen3.8-Flash-Next | Technical Report | `2026/qwen/qwen3.8-flash-next-technical-report.pdf` | [Qwen official repository](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) | 最高：重点看 Gated DeltaNet/Gated Attention、长上下文及位置处理 |
| Alibaba / Qwen | Qwen3.8-27B | Model Card | `2026/qwen/qwen3.8-27b-model-card.md` | [Qwen official model card](https://huggingface.co/Qwen/Qwen3.8-27B) | 高：与 Flash-Next 报告对照上下文长度和位置相关配置 |
| DeepSeek | DeepSeek V4 | Technical Report | `2026/deepseek/deepseek-v4-technical-report.pdf` | [DeepSeek official announcement](https://deepseek.com/en/news/v4-preview/)；[arXiv: DeepSeek V4](https://arxiv.org/abs/2606.19348) | 最高：重点看 CSA/HCA、百万 token 上下文和位置处理 |
| DeepSeek | DeepSeek V4 | Technical Documentation / Model Card | `2026/deepseek/deepseek-v4-technical-documentation.pdf` | [DeepSeek transparency](https://www.deepseek.com/en/transparency/)；[官方 PDF](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf) | 高：作为报告之外的官方规格补充 |
| ByteDance Seed | Seed 2.0 | Model Card | `2026/bytedance/seed2.0-model-card.pdf` | [arXiv: Seed2.0 Model Card](https://arxiv.org/abs/2607.00248) | 中：模型卡可能只给高层结构，搜索 `position`、`long context` |
| ByteDance Seed | Seed1.8 | Model Card | `2026/bytedance/seed1.8-model-card.pdf` | [arXiv: Seed1.8 Model Card](https://arxiv.org/abs/2603.20633) | 中：与 Seed 2.0 对比位置编码/上下文长度描述 |
| Anthropic | Claude Opus 4.8 | System Card | `2026/anthropic/claude-opus-4.8-system-card.pdf` | [Anthropic transparency](https://www.anthropic.com/transparency)；[官方 PDF](https://cdn.sanity.io/files/4zrzovbb/website/c886650a2e96fc0925c805a1a7ca77314ccbf4a6.pdf) | 低到中：主要是安全评测，架构和位置编码通常不会展开 |
| Moonshot AI / Kimi | Kimi K3 | Technical Report | `2026/kimi/kimi-k3-technical-report.pdf` | [arXiv: Kimi K3](https://arxiv.org/abs/2607.24653)；[Kimi official technical blog](https://www.kimi.com/en/blog/kimi-k3) | 最高：重点看 Kimi Delta Attention、Attention Residuals 和长序列位置处理 |
| Z.ai / GLM | GLM-5（GLM-5.3 的基座） | Technical Report | `2026/glm/glm-5-technical-report.pdf` | [arXiv: GLM-5](https://arxiv.org/abs/2602.15763)；[GLM-5.3 official release](https://z.ai/blog/glm-5.3) | 最高：重点看混合稀疏/线性注意力、长上下文和位置编码 |
| Z.ai / GLM | GLM-5.3-Flash | Model Card | `2026/glm/glm-5.3-flash-model-card.md` | [official Hugging Face model card](https://huggingface.co/zai-org/GLM-5.3-Flash) | 高：与 GLM-5 报告对照其混合注意力和上下文实现 |

## 逐篇位置编码笔记

- [Gemma 4](2026/google/gemma-4-position-encoding.md)
- [Phi-4-Reasoning-Vision-15B](2026/microsoft/phi-4-reasoning-vision-15b-position-encoding.md)
- [Qwen3.5-Omni](2026/alibaba/qwen3.5-omni-position-encoding.md)
- [Qwen3.8](2026/qwen/qwen3.8-position-encoding.md)
- [DeepSeek V4](2026/deepseek/deepseek-v4-position-encoding.md)
- [Kimi K3](2026/kimi/kimi-k3-position-encoding.md)
- [GLM-5 / GLM-5.3](2026/glm/glm-5-position-encoding.md)
- [Seed1.8](2026/bytedance/seed1.8-position-encoding.md)
- [Seed2.0](2026/bytedance/seed2.0-position-encoding.md)
- [Claude Opus 4.8](2026/anthropic/claude-opus-4.8-position-encoding.md)
- [GPT-5.6](2026/openai/gpt-5.6-position-encoding.md)
- [跨模型总总结](2026/position-encoding-summary.md)

## 已核验但没有独立架构 PDF

| 公司 | 模型 | 官方公开资料 | 位置编码结论 |
|---|---|---|---|
| OpenAI | GPT-5.6（Sol / Terra / Luna） | [官方发布页](https://openai.com/index/gpt-5-6/)；[工程说明](https://openai.com/index/gpt-5-6-frontier-intelligence-efficiency/)；[API model guidance](https://developers.openai.com/api/docs/guides/latest-model) | 未披露内部位置编码，官方资料主要讨论能力、推理、缓存、工具调用和效率 |

## 明天的阅读顺序

1. 先读 Kimi K3、GLM-5、Qwen3.8-Flash-Next、DeepSeek V4，这几份最可能包含架构或长上下文实现线索。
2. 对每份文档搜索：`positional encoding`、`position embedding`、`RoPE`、`rotary`、`relative position`、`YaRN`、`ALiBi`、`context length`、`interleaving`。
3. 记录四件事：位置编码施加在 Q/K 的哪一部分、是否使用 RoPE 及其变体、训练长度与推理长度是否不同、长上下文是否使用插值/缩放/注意力结构改造。
4. 对没有公开细节的文档写成“未披露”，不要根据模型名称或博客猜测实现。

## 范围说明

截至 2026-08-29，OpenAI 已公开 GPT-5.6 的官方发布页和工程说明，但没有公开可下载的模型架构 Technical Report；xAI 也没有找到可核验的 2026 年模型架构 Technical Report。GLM-5.3 官方说明其沿用 GLM-5 基座，因此使用 GLM-5 Technical Report 作为架构资料，并额外保存 GLM-5.3-Flash 模型卡。DeepSeek V4 的正式报告已通过 DeepSeek 官方公告关联的 arXiv 页面下载，同时保留官方技术文档/模型卡作为规格补充。
