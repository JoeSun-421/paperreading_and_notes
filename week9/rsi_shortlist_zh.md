# 机器人 RSI / harness 短名单

> 整理日期：2026-10-11（北京时间）。筛选标准：头部公司/顶级实验室优先 → 不知名大学先丢 → 国内只看清华/北大/浙大/复旦 → 必须有「多轮迭代曲线」或「真机数据」。
> 核实方式：arXiv 摘要/HTML 正文、项目主页、WebSearch。标「⚠未本次核实」的数字来自既有认知，没有在本轮逐字查到原文，引用前请再看一眼原文。
> CaP 方向四篇（CaP-X / Uni-Skill / Code Evolution for Control / Lifelong Robot Library Learning）直接采用 Agent Expert 的筛选结论，本文件未重复核实。

## 推荐入选（满足标准）

### 第一梯队：大厂 tech report + 真机 + 迭代/自我改进数字

**1. ENPIRE: Agentic Robot Policy Self-Improvement in the Real World**（2026，CoRL 2026）
- 链接：https://arxiv.org/abs/2606.19980 ｜ 主页：https://research.nvidia.com/labs/gear/enpire/
- 单位：NVIDIA（GEAR）+ CMU + UC Berkeley（Jim Fan、Yuke Zhu、Guanya Shi、Ken Goldberg）
- 一句话：给 coding agent（Codex / Claude Code / Kimi Code）搭一套「自动复位 + 自动判成功 + 并行 rollout + 自我演化」的物理 harness，让 agent 自己改训练代码/算法、在真机上爬坡。8 台双臂 YAM 真机组成 fleet；PushT、插针、GPU 插槽、剪扎带等任务成功率爬到 99%（插针要求连续 50 次成功）；有成功率随研究时间变化的爬坡曲线和「idea tree」（如 BC 正则化一项 +10.8pp）；1→8 台机器人把插针达到近满分的时间从 1.5 小时以上降到约 40 分钟。
- 为什么过筛：NVIDIA tech report；真机 + 迭代曲线都有；正好是「harness + 机器人 RSI」，和 ASPIRE / CaP-X 同一个团队，是这条线最新、最贴题的一篇。

**2. π\*0.6: a VLA That Learns From Experience（RECAP）**（2025.11，RSS 2026）
- 链接：https://arxiv.org/abs/2511.14759 ｜ 博客：https://www.physicalintelligence.company/blog/pistar06
- 单位：Physical Intelligence
- 一句话：用 RECAP（优势条件化的 RL）把示范、自主 rollout、人工纠正一起用来做 VLA 的自我改进；全部是真机（叠衣服、组装纸箱、做意式咖啡）。加入真机自主经验后，最难的任务吞吐量提升超过 2 倍、失败率约降一半；做两轮 RECAP 迭代：T 恤/短裤吞吐 +50%，纸箱第二轮后达 2× 吞吐；咖啡、纸箱最终成功率超过 90%；可连续做咖啡 13 小时。
- 为什么过筛：头部机器人公司 tech report；大规模真机；有多轮迭代的数字。

**3. Self-Improving Embodied Foundation Models**（2025.09，NeurIPS 2025）
- 链接：https://arxiv.org/abs/2509.15155 ｜ 主页：https://self-improving-efms.github.io/
- 单位：Google DeepMind
- 一句话：两阶段后训练：SFT（行为克隆 + steps-to-go 预测），然后用 steps-to-go 自动生成奖励和成功检测，让机器人 fleet 在几乎无人监督下自主练习。真机 LanguageTable：只用 20%/80% 示范数据时，成功率从约 62–63% 升到 87–88%，只多用约 3% 的 episode；每次实验约 20 小时，1 个人看管整个 fleet；新技能 BananaTable 约 8 小时从 63% 升到 85%。
- 为什么过筛：DeepMind；真机 + 自我改进前后数字。

**4. Eureka: Human-Level Reward Design via Coding LLMs**（2023，ICLR 2024）
- 链接：https://arxiv.org/abs/2310.12931
- 单位：NVIDIA + UPenn + Caltech + UT Austin
- 一句话：LLM 写奖励函数代码 → 在 Isaac Gym 训练 → 根据训练统计反思再改奖励，多轮进化。仅仿真；有逐轮奖励进化曲线；29 个环境中 83% 的任务超过人类专家写的奖励，平均归一化提升 52%（摘要数字）。
- 为什么过筛：NVIDIA；有明确的多轮迭代曲线（无真机，按「迭代曲线」一条过筛）。

**5. DrEureka: Language Model Guided Sim-To-Real Transfer**（2024，RSS 2024）
- 链接：https://arxiv.org/abs/2406.01967
- 单位：UPenn + NVIDIA + UT Austin
- 一句话：在 Eureka 基础上让 LLM 同时设计奖励和 domain randomization 参数，实现 sim-to-real；真机演示：Unitree 四足在瑜伽球上行走、四足步态、灵巧手转方块。⚠未本次核实具体成功率数字。
- 为什么过筛：NVIDIA 参与；有真机。但迭代仍然在仿真里做，真机只是部署目标。

**6. CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation**（2026）
- 链接：https://arxiv.org/abs/2603.22435
- 单位：NVIDIA + Berkeley / Stanford / CMU（Fei-Fei Li、Ken Goldberg、Jim Fan 等）
- 一句话：Code as Policies 方向的基准框架 / harness，研究多轮反馈、技能合成和集成采样对 coding agent 操控可靠性的影响；它是 ASPIRE、ENPIRE、RPG、RoboRSI 的共同基线。
- 为什么过筛：采纳 Agent Expert 的结论（保留）。本人未单独核实它的真机/曲线细节。

**7. ASPIRE: Agentic /skills Discovery for Robotics**（2026）—— 对照，已在用户文档（第 10 条）
- 链接：https://arxiv.org/abs/2607.00272
- 单位：NVIDIA + Berkeley 等（Linxi Fan、Yuke Zhu、Ken Goldberg、Guanya Shi、Guanzhi Wang）
- 一句话：coding agent 修复失败程序，把修好的技能存进不断增长的技能库。RPG 在 22 个任务上复现的 ASPIRE 成功率是 75.5%（参见下面的 RPG）。
- 为什么过筛：已收录，作为对照列出。

### 第二梯队：顶级学校 / 大厂合作 + 真机或迭代曲线

**8. Reconstruct, Practice, Go Real (RPG): Guided Self-Improvement for Embodied Agents**（2026.10）
- 链接：https://arxiv.org/abs/2610.02204 ｜ 主页：https://rpg-robot.github.io/
- 单位：UC Berkeley + Amazon FAR（Pieter Abbeel、Rocky Duan）+ MIT + UChicago
- 一句话：不改模型权重，只改技能库和 system prompt：根据离线数据在仿真里构造练习任务，诊断失败后改技能/prompt，跨任务验证通过才保留。22 个任务的成功率从第 1 轮 28.6% 升到第 15 轮 95.0%（超过 ASPIRE 75.5%、CaP-Agent0+GPT-6 60.0%）；冻结后上真机 30/30 次成功（基线 14/30）；技能库从 15 个增长到 38 个。消融：去掉 Video Analyzer 或 Privileged Agent，成功率从 78.2% 掉到约 51%。
- 为什么过筛：Berkeley + Amazon；逐轮迭代曲线 + 真机都有；直接和 ASPIRE 对比，非常贴题。

**9. SOAR: Autonomous Improvement of Instruction Following Skills via Foundation Models**（2024，CoRL 2024）
- 链接：https://arxiv.org/abs/2407.20635
- 单位：UC Berkeley（Sergey Levine）
- 一句话：VLM 提出任务、自主采集、自动判成功，再用这些数据改进语言条件策略；多台 WidowX 真机自主运行。⚠未本次核实具体提升数字。
- 为什么过筛：顶级学校；真机自主自我改进。没有大厂署名，所以放第二梯队。

**10. SERL / HIL-SERL**（2024 / Science Robotics 2025）
- 链接：https://arxiv.org/abs/2401.16013 （SERL）、https://arxiv.org/abs/2410.21845 （HIL-SERL）
- 单位：UC Berkeley（+ Stanford、UW 等）
- 一句话：真机样本高效 RL 软件栈；PCB 插件、走线等任务在几十分钟到一两小时内训到接近 100%，有训练曲线。⚠未本次核实具体数字。注：ENPIRE 的真机 RL 部分直接基于 SERL 的异步架构。
- 为什么过筛：顶级学校；真机 + 学习曲线。但它是 RL 基础设施，不是 agent/harness 式 RSI，算偏背景阅读。

**11. Practice Makes Perfect: Planning to Learn Skill Parameter Policies（EES）**（2024，RSS 2024）
- 链接：https://arxiv.org/abs/2402.15025
- 单位：MIT（Kaelbling、Lozano-Pérez）+ Boston Dynamics AI Institute
- 一句话：机器人自己决定练哪个技能（Estimate-Extrapolate-Situate），在 Spot 真机上自主练习，提升技能参数策略，有能力随练习时间变化的曲线。⚠未本次核实具体数字。
- 为什么过筛：MIT + BD AI Institute；真机自主练习。

**12. Code as Policies: Language Model Programs for Embodied Control**（2022，ICRA 2023）—— 奠基对照
- 链接：https://arxiv.org/abs/2209.07753
- 单位：Google（Robotics at Google）
- 一句话：LLM 直接写机器人策略代码，在多个真机平台上演示；本身没有自我改进或迭代曲线。
- 为什么过筛：Google + 真机；是整条 CaP → CaP-X → ASPIRE/ENPIRE/RPG 线的起点，适合作为背景。

## 明确淘汰（并写原因）

| 论文 | 单位 | 淘汰原因 |
|---|---|---|
| RegenHarness | — | 用户已判弱：自我改进闭环没有任何量化结果（没有基线对比、没有消融），真机只有一次四足巡检演示 |
| Recursive Harness Distillation (RHD) | — | 用户已判弱：本质是 prompt / playbook 蒸馏；首版 playbook 让弱 agent 从 43.8% 降到 31.3%，弱 agent 自己改写只有 22.9%（低于不用 playbook） |
| Code Evolution for Control (2601.06845) | 深圳寓意科技 + 城大/深大/电子科大 | 无大厂/顶级实验室署名，学校不在名单（采纳 Agent Expert 结论） |
| Lifelong Robot Library Learning (Tziafas & Kasaei, ICRA 2024) | 格罗宁根大学 | 非名单内学校，无大厂署名（采纳 Agent Expert 结论） |

## 边界待定（可问用户再定）

| 论文 | 单位 | 情况 / 待定理由 |
|---|---|---|
| Uni-Skill (2603.02623) | 中科院计算所 + 国科大 | 有仿真 + 真机，但单位不在「清北浙复」名单内（Agent Expert 建议放边界） |
| AutoRT (2401.12963) | Google DeepMind | 大厂 + 大规模真机（20 多台机器人、7 个月、约 7.7 万条真机 episode ⚠未本次核实），但它是「LLM 编排数据采集」，不是策略自我改进闭环，没有迭代后成功率曲线 |
| RoboCat (2306.11706) | Google DeepMind | 大厂 + 多款真机臂，有「自生成数据 → 再训练」的 self-improvement 循环；但只做了一轮这样的循环，⚠本次未核实具体提升数字。偏 2023 年的通用 agent，和 harness / CaP 线关系较远 |
| MEDAL++ / Self-Improving Robots (2303.01488) | Stanford（Chelsea Finn） | 顶级学校 + 真机自主 RL 自我改进，但没有大厂署名；⚠未本次核实数字 |
| SELFI (2403.00991) | UC Berkeley + Toyota（⚠单位未逐一核实） | 真机在线自我改进，但是导航任务，偏离操控 / harness 主线 |
| RoboRSI (2610.12424) | 上海交大（卢策吾）+ Noematrix（初创） | 名字最贴题：移动操作真机 104 轮技能演化 + LIBERO 等四个基准最优；一天内自迭代 5 轮，LIBERO 覆盖任务数从 32 升到 71（共 120 个）。但上交不在用户名单内；真机进展图是事后挑选的 10 次运行，没有严格曲线 |
| Beyond Imitation: Q-Planning (2608.21204) | Animesh Garg 组（⚠单位未核实，可能是 Georgia Tech / NVIDIA） | 真机 5 轮迭代：叠杯 40%→90%，插钱包 25%→80%；单位未核实，暂放边界 |
| PLD: Self-Improving VLA via Residual RL (2511.00091) | NVIDIA（GEAR）等 | ENPIRE 的前置工作，NVIDIA 署名；⚠本次未打开核实真机/曲线细节，值得补查 |

## 建议阅读顺序
ENPIRE → RPG → π\*0.6/RECAP → Self-Improving EFM → CaP-X（基线）→ Eureka/DrEureka → 其余第二梯队。
前两篇和用户已收录的 ASPIRE 是同一条「coding agent + harness 自我改进」线，并且都直接拿 ASPIRE 做对比或建立在它上面。
