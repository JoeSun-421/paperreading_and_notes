# Robotic Harness / RSI 相关论文与技术报告

范围：这里把 **harness** 理解为连接模型、机器人技能/API、执行反馈、验证、记忆和更新策略的运行时系统；同时收录通过自主练习、在线 RL、奖励/仿真代码生成来实现 robotic self-improvement 的工作。排序优先考虑与 harness/RSI 的直接相关性，再考虑影响力和系统完整度。

> 注：2026 年条目是截至 2026-10 的新近 arXiv 预印本，主题相关性很高，但学术影响力仍在形成。

## 清单

1. **ASPIRE: Agentic /Skills Discovery for Robotics**（Runyu Lu、Yubo Wu、Ethan Kou、Letian Fu、Wenli Xiao、Ajay Mandlekar、Yinzhen Xu、Guanya Shi、Ken Goldberg、Ang Chen、Mosharaf Chowdhury、Yuke Zhu、Linxi “Jim” Fan、Guanzhi Wang；NVIDIA GEAR、UMich、UIUC、UC Berkeley、CMU 等）
   - **年份**：2026（arXiv 预印本）
   - **链接**：[arXiv](https://arxiv.org/abs/2607.00272) · [项目页](https://research.nvidia.com/labs/gear/aspire/)
   - **相关性**：目前最直接对应“robotic harness + RSI”的工作之一。它让代码代理读取多模态执行轨迹、诊断失败、修复 code-as-policy 程序，并把验证过的修复沉淀为可迁移技能库，再用演化搜索持续探索。

2. **RegenHarness: A Robot Agent Harness with Evidence-Gated Recursive Self-Improvement**（Kailin Wang、Haoxiang Jie、Yaoyuan Yan、Zhiyou Heng、Zhaosong Li）
   - **年份**：2026（arXiv 预印本）
   - **链接**：[arXiv](https://arxiv.org/abs/2609.27612)
   - **相关性**：标题和协议都直接提出 embodied robotic RSI。其 harness 将提议、调度、观察、验证、提交和受限恢复分开，用版本化记忆、回归检查、发布/回滚门控来更新任务模板、路由和恢复策略，而不更新模型权重。

3. **Recursive Harness Distillation across Agents for Robot Manipulation**（Seungyeon Kim、Junhoo Lee、Minkyu Kim、Baekseung Kim、Nojun Kwak）
   - **年份**：2026（arXiv 预印本）
   - **链接**：[arXiv](https://arxiv.org/abs/2609.33378)
   - **相关性**：研究“经验如何在 harness 中累积”。强代理把干预经验蒸馏成 playbook，轻代理根据真实执行反馈递归修订；经验跨任务和代理复用，且不需要更新 VLA 参数。

4. **AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents**（Michael Ahn、Debidatta Dwibedi、Chelsea Finn、Sergey Levine 等；Google DeepMind）
   - **年份**：2024（ICRA 2024 VLMNM Workshop）
   - **链接**：[arXiv](https://arxiv.org/abs/2401.12963) · [项目页](https://auto-rt.github.io/)
   - **相关性**：把 VLM 场景理解、LLM 任务提议、constitution 安全筛选、人工协助和机器人执行串成大规模数据采集 harness。论文报告在 20 多台机器人上收集约 77k 个真实 episode，是“让机器人自己提出并执行练习任务”的重要系统范例。

5. **Practice Makes Perfect: Planning to Learn Skill Parameter Policies**（Nishanth Kumar、Tom Silver、Willie McClinton、Linfeng Zhao、Stephen Proulx、Tomás Lozano-Pérez、Leslie Pack Kaelbling、Jennifer Barry；The AI Institute、MIT CSAIL、Northeastern University）
   - **年份**：2024（RSS 2024）
   - **链接**：[arXiv](https://arxiv.org/abs/2402.15025) · [项目页](https://ees.csail.mit.edu/)
   - **相关性**：提出 EES（Estimate–Extrapolate–Situate），机器人估计技能熟练度、预测练习收益并主动选择要练的技能，在无需环境重置的情况下反复计划、实践和学习；这是“自主练习调度”这一 RSI 子问题的代表工作。

6. **SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning**（Jianlan Luo、Zheyuan Hu、Charles Xu、You Liang Tan、Jacob Berg、Archit Sharma、Stefan Schaal、Chelsea Finn、Abhishek Gupta、Sergey Levine；UC Berkeley、华盛顿大学、Stanford、Intrinsic）
   - **年份**：2024（ICRA 2024）
   - **链接**：[arXiv](https://arxiv.org/abs/2401.16013) · [项目页](https://serl-robot.github.io/)
   - **相关性**：它本身不是 RSI 算法，但提供了很实用的 robot-learning harness：样本高效离策略 RL、奖励计算、环境重置、机器人控制器和任务脚本。开放实现及 25–50 分钟级真实机器人训练，使在线自改进实验更容易复现和扩展。

7. **SELFI: Autonomous Self-Improvement with Reinforcement Learning for Social Navigation**（Noriaki Hirose、Dhruv Shah、Kyle Stachowicz、Ajay Sridhar、Sergey Levine；UC Berkeley、Toyota Motor North America）
   - **年份**：2024（CoRL 2024 Oral）
   - **链接**：[arXiv](https://arxiv.org/abs/2403.00991) · [项目页](https://sites.google.com/view/selfi-rl/)
   - **相关性**：直接研究机器人部署后的自主改进。SELFI 用在线 model-free RL 微调离线 model-based 策略，在真实社会导航中减少碰撞和人工干预，展示了“预训练策略 + 安全在线更新”的闭环。

8. **Self-Improving Autonomous Underwater Manipulation**（Ruoshi Liu、Huy Ha、Mengxue Hou、Shuran Song、Carl Vondrick；Columbia University、Stanford University、University of Notre Dame）
   - **年份**：2024（arXiv 预印本）
   - **链接**：[arXiv](https://arxiv.org/abs/2410.18969) · [项目页](https://aquabot.cs.columbia.edu/)
   - **相关性**：AquaBot 先从人类示范得到行为克隆策略，再用自主 self-learning 超越遥操作性能；在抓取、垃圾分拣和救援检索等水下任务上做真实机器人实验，是跨复杂动力学环境的 self-improvement 案例。

9. **DrEureka: Language Model Guided Sim-To-Real Transfer**（Yecheng Jason Ma、William Liang、Hung-Ju Wang、Sam Wang、Yuke Zhu、Linxi “Jim” Fan、Osbert Bastani、Dinesh Jayaraman；UPenn、NVIDIA、UT Austin）
   - **年份**：2024（RSS 2024）
   - **链接**：[arXiv](https://arxiv.org/abs/2406.01967) · [项目页](https://eureka-research.github.io/dr-eureka/)
   - **相关性**：把 LLM 放进 sim-to-real 设计环：自动生成奖励函数和 domain-randomization 分布，并用仿真评估筛选候选。它把“奖励/仿真参数代码的迭代搜索”变成可复用的机器人训练 harness，并明确指向用真实执行失败作为后续反馈。

10. **Eureka: Human-Level Reward Design via Coding Large Language Models**（Yecheng Jason Ma、William Liang、Guanzhi Wang、De-An Huang、Osbert Bastani、Dinesh Jayaraman、Yuke Zhu、Linxi “Jim” Fan、Anima Anandkumar；NVIDIA、UPenn、Caltech、UT Austin）
    - **年份**：2024（ICLR 2024；arXiv 首发 2023）
    - **链接**：[arXiv](https://arxiv.org/abs/2310.12931) · [项目页](https://eureka-research.github.io/)
    - **相关性**：奠定了许多后续“代码型 RSI”机器人的基本范式：LLM 生成奖励代码，RL 训练并评估，再把结果反馈给 LLM 做进化式改写。覆盖 29 个环境和 10 种机器人形态，影响了 DrEureka、Text2Reward 等后续路线。

11. **Text2Reward: Reward Shaping with Language Models for Reinforcement Learning**（Tianbao Xie、Siheng Zhao、Chen Henry Wu、Yitao Liu、Qian Luo、Victor Zhong、Yanchao Yang、Tao Yu；香港大学、南京大学、CMU、Microsoft Research、University of Waterloo）
    - **年份**：2024（ICLR 2024 Spotlight；arXiv 首发 2023）
    - **链接**：[arXiv](https://arxiv.org/abs/2309.11489) · [项目页](https://text-to-reward.github.io/)
    - **相关性**：将自然语言目标转成可执行的 dense reward 程序，并允许根据失败模式和人类反馈迭代重写；在 ManiSkill2、MetaWorld 等机器人任务上验证，是“LLM 生成工具/奖励 + 执行反馈”的早期代表。

12. **Self-Improving Robots: End-to-End Autonomous Visuomotor Reinforcement Learning（MEDAL++）**（Archit Sharma、Ahmed M. Ahmed、Rehaan Ahmad、Chelsea Finn；Stanford University）
    - **年份**：2023（CoRL 2023）
    - **链接**：[arXiv](https://arxiv.org/abs/2303.01488) · [项目页](https://architsharma97.github.io/self-improving-robots/)
    - **相关性**：较早系统化讨论 reset-free self-improvement：从少量专家示范开始，前向策略执行任务、后向策略 undo 任务，机器人自主收集训练数据并同时学习奖励。真实 Franka 操作实验中相对行为克隆提升 30–70%，是后续 autonomous practice 工作的重要基线。

## 按主题快速归类

- **直接的 harness RSI**：ASPIRE、RegenHarness、Recursive Harness Distillation。
- **自主练习与在线策略改进**：Practice Makes Perfect、SELFI、AquaBot、MEDAL++。
- **训练/数据/奖励设计基础设施**：AutoRT、SERL、DrEureka、Eureka、Text2Reward。
