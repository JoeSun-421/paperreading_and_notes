# paperreading_and_notes

Robotics learning papers and Chinese translations.

按周存放的论文 PDF，以及对应的中文译文和精读笔记。远程仓库：<https://github.com/JoeSun-421/paperreading_and_notes>

## 目录

顶层是阅读周次：`week1`、`week2`、`week3`、`week5`–`week10`。没有 `week4`。

同一周目录里，PDF 是原文。中文 Markdown 与原文放在一起：

- 译文：`*_zh.md`、`*-全文中文翻译.md`、`*-full-translation.md`
- 笔记：`literature-reading-notes.md`、`*_notes_zh.md`、`*_精读笔记.md`、`*_summary.md`

两个专题子目录：[`week2/llm_tech_report/`](week2/llm_tech_report/INDEX.md)（2026 大模型技术报告与位置编码笔记）、[`week3/feifei_li_atlas/`](week3/feifei_li_atlas/README.md)（Atlas）。

## 各周文件

### week1

只有 PDF。

- Neural Machine Translation of Rare Words with Subword Units（arXiv:1508.07909）：[v2](week1/1508.07909v2.pdf)、[v5](week1/1508.07909v5.pdf)
- [SPADE: Self-Play in Adaptive Synthetic Executable Environments](week1/2608.19197_SPADE.pdf)

### week2

PDF：

- [Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning](week2/1702.03118v3.pdf)（arXiv:1702.03118）
- [Decoupled Weight Decay Regularization](week2/1711.05101_Decoupled_Weight_Decay_Regularization.pdf)
- [On Layer Normalization in the Transformer Architecture](week2/2002.04745v2.pdf)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](week2/2104.09864_roformer_enhanced_transformer_with_rotary_position_embedding.pdf)
- [The Llama 3 Herd of Models](week2/2407.21783v3.pdf)
- [Improving Language Understanding by Generative Pre-Training](week2/language_understanding_paper.pdf)
- [Transformers without Tears: Improving the Normalization of Self-Attention](week2/transformer_without_tears.pdf)

笔记：[Week Two Notes](week2/mynote.md)（Pre-LN 等）、[Markdown 常用语法速查](week2/markdown_syntax_cheatsheet.md)。

YaRN：原文 [2309.00071.pdf](week2/llm_tech_report/2309.00071.pdf)，译文 [YaRN：高效扩展大语言模型上下文窗口](week2/llm_tech_report/yarn-context-extension.md)。

2026 技术报告的总索引是 [INDEX.md](week2/llm_tech_report/INDEX.md)，位置编码总总结是 [2026 LLM 位置编码总总结](week2/llm_tech_report/2026/position-encoding-summary.md)。各篇笔记：

| 模型 | 笔记 | 原文 |
| --- | --- | --- |
| Claude Opus 4.8 | [位置编码](week2/llm_tech_report/2026/anthropic/claude-opus-4.8-position-encoding.md) | [system card](week2/llm_tech_report/2026/anthropic/claude-opus-4.8-system-card.pdf) |
| DeepSeek-V4 | [位置编码](week2/llm_tech_report/2026/deepseek/deepseek-v4-position-encoding.md) | [technical report](week2/llm_tech_report/2026/deepseek/deepseek-v4-technical-report.pdf)、[technical documentation](week2/llm_tech_report/2026/deepseek/deepseek-v4-technical-documentation.pdf) |
| Gemma 4 | [位置编码](week2/llm_tech_report/2026/google/gemma-4-position-encoding.md) | [technical report](week2/llm_tech_report/2026/google/gemma-4-technical-report.pdf) |
| GLM-5 / GLM-5.3 | [位置编码](week2/llm_tech_report/2026/glm/glm-5-position-encoding.md)、[GLM-5.3-Flash 模型卡](week2/llm_tech_report/2026/glm/glm-5.3-flash-model-card.md) | [GLM-5 technical report](week2/llm_tech_report/2026/glm/glm-5-technical-report.pdf) |
| GPT-5.6 | [位置编码](week2/llm_tech_report/2026/openai/gpt-5.6-position-encoding.md) | 无 PDF |
| Kimi K3 | [位置编码](week2/llm_tech_report/2026/kimi/kimi-k3-position-encoding.md) | [technical report](week2/llm_tech_report/2026/kimi/kimi-k3-technical-report.pdf) |
| Phi-4-Reasoning-Vision-15B | [位置编码](week2/llm_tech_report/2026/microsoft/phi-4-reasoning-vision-15b-position-encoding.md) | [technical report](week2/llm_tech_report/2026/microsoft/phi-4-reasoning-vision-15b-technical-report.pdf) |
| Qwen3.5-Omni | [位置编码](week2/llm_tech_report/2026/alibaba/qwen3.5-omni-position-encoding.md) | 无 PDF |
| Qwen3.8 | [位置编码](week2/llm_tech_report/2026/qwen/qwen3.8-position-encoding.md)、[Qwen3.8-27B 模型卡](week2/llm_tech_report/2026/qwen/qwen3.8-27b-model-card.md) | [Qwen3.8-Flash-Next technical report](week2/llm_tech_report/2026/qwen/qwen3.8-flash-next-technical-report.pdf) |
| Seed1.8 | [位置编码](week2/llm_tech_report/2026/bytedance/seed1.8-position-encoding.md) | [model card](week2/llm_tech_report/2026/bytedance/seed1.8-model-card.pdf) |
| Seed2.0 | [位置编码](week2/llm_tech_report/2026/bytedance/seed2.0-position-encoding.md) | [model card](week2/llm_tech_report/2026/bytedance/seed2.0-model-card.pdf) |

### week3

| 文献 | 原文 | 中文 |
| --- | --- | --- |
| π0: A Vision-Language-Action Flow Model for General Robot Control | [pi0.pdf](week3/pi0.pdf) | [π0 精读笔记（含译文）](week3/literature-reading-notes.md) |
| Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | [2101.03961.pdf](week3/2101.03961.pdf) | |
| DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | [deepseek-v2.pdf](week3/deepseek-v2.pdf) | |
| FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | [flashattention-2.pdf](week3/flashattention-2.pdf) | |
| Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality | [mamba-2.pdf](week3/mamba-2.pdf) | |
| Atlas | | [目录](week3/feifei_li_atlas/README.md)、[Atlas：面向空间智能的统一世界模型](week3/feifei_li_atlas/atlas_technical_report.md) |

### week5

只有 PDF。

- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](week5/2503.14476.pdf)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](week5/DeepSeek-R1_2501.12948v2.pdf)
- [Understanding R1-Zero-Like Training: A Critical Perspective](week5/Dr.grpo.pdf)（`Dr.grpo.pdf`）
- [Reinforcement Learning with Human Feedback: Learning Dynamic Choices via Pessimism](week5/RLHF.pdf)
- [OpenAI o1 System Card](week5/openai_o1.pdf)

### week6

只有 PDF。

- [Maximum Likelihood Reinforcement Learning](week6/maxRL.pdf)
- [MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement](week6/mimo_rl.pdf)

### week7

只有 PDF。

- [In-Context Learning for Robots: Methods and Applications](week7/iclrobot.pdf)

### week8

| 文献 | 原文 | 中文 |
| --- | --- | --- |
| π0.5: a Vision-Language-Action Model with Open-World Generalization | [pi05.pdf](week8/pi05.pdf) | [精读笔记（含全文译文）](week8/literature-reading-notes.md) |
| FAST: Efficient Action Tokenization for Vision-Language-Action Models | [fast.pdf](week8/fast.pdf) | [全文译文](week8/fast-full-translation.md) |
| π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities | [pi07.pdf](week8/pi07.pdf) | [全文译文](week8/pi07-全文中文翻译.md) |
| RobotWorld: Benchmarking Multimodal Agents for Robot Use Across Diverse Tasks and Embodiments | [2610.10409.pdf](week8/2610.10409.pdf)、[robotmodel.pdf](week8/robotmodel.pdf)（同一份 PDF） | |

`literature-reading-notes.md` 的索引还覆盖 FAST 与 π0.7；上面两篇另有独立译文。

### week9

| 文献 | 原文 | 中文 |
| --- | --- | --- |
| ASPIRE | [ASPIRE.pdf](week9/ASPIRE.pdf) | [译文](week9/ASPIRE_zh.md)、[精读笔记](week9/ASPIRE_精读笔记.md) |
| CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation | [CaP-X.pdf](week9/CaP-X.pdf) | |
| DrEureka: Language Model Guided Sim-To-Real Transfer | [DrEureka.pdf](week9/DrEureka.pdf) | |
| ENPIRE: Agentic Robot Policy Self-Improvement in the Real World | [ENPIRE.pdf](week9/ENPIRE.pdf) | |
| EUREKA: Human-Level Reward Design via Coding Large Language Models | [Eureka.pdf](week9/Eureka.pdf) | [译文](week9/Eureka_zh.md) |
| Recursive Harness Distillation across Agents for Robot Manipulation | [Recursive_Harness_Distillation.pdf](week9/Recursive_Harness_Distillation.pdf) | [精读笔记](week9/Recursive_Harness_Distillation_notes_zh.md) |
| RegenHarness: A Robot Agent Harness with Evidence-Gated Recursive Self-Improvement | [RegenHarness.pdf](week9/RegenHarness.pdf) | [精读笔记](week9/RegenHarness_notes_zh.md) |
| Self-Improving Embodied Foundation Models | [Self_Improving_EFM.pdf](week9/Self_Improving_EFM.pdf) | |
| ARC: A Reasoning Recipe for Robot Foundation Models | [arc.pdf](week9/arc.pdf) | [译文](week9/arc_zh.md)、[小结](week9/arc_summary.md) |
| π\*0.6: a VLA That Learns From Experience | [pi_star_0.6_RECAP.pdf](week9/pi_star_0.6_RECAP.pdf) | |

阅读清单（不是额外下载的 PDF）：[Robotic Harness / RSI 相关论文与技术报告](week9/robotic_harness_rsi_papers.md)、[机器人 RSI / harness 短名单](week9/rsi_shortlist_zh.md)。

### week10

| 文献 | 原文 | 中文 |
| --- | --- | --- |
| Real-Time Execution of Action Chunking Flow Policies | [RTC.pdf](week10/RTC.pdf) | 见精读笔记 |
| Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection | [cam.pdf](week10/cam.pdf) | 见精读笔记 |
| Tutorial on Diffusion Models for Imaging and Vision | [diffusion.pdf](week10/diffusion.pdf) | [全文译文](week10/diffusion-全文中文翻译.md) |
| Scaling Online Exploration for Robot Agents | [rax.pdf](week10/rax.pdf) | 见精读笔记 |
| LoopWAM: Closed-Loop Generalist Policy Improvement from Deployment Rollouts | [loopwam.pdf](week10/loopwam.pdf) | [译文](week10/loopwam.zh.md) |
| World Action Models: The Next Frontier in Embodied AI | [wam.pdf](week10/wam.pdf) | [译文](week10/wam.zh.md) |

[文献精读笔记](week10/literature-reading-notes.md) 覆盖 RTC、Code-as-Monitor、扩散模型教程、Scaling Online Exploration（RAX）。
