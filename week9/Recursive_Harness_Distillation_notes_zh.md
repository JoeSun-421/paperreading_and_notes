# Recursive Harness Distillation across Agents for Robot Manipulation 精读笔记

## 一句话概括
强 agent（GPT-6 Astra）先与冻结的 VLA 交互，把干预经验蒸馏成一份 playbook.md，再根据弱 agent（GPT-5.6 Luna）实际使用 playbook 的执行反馈递归修订。整个过程不更新任何模型参数。SimplerEnv 上 Luna 从 43.8% 提到 66.7%，真机从 37.3% 提到 64.0%。

## 自我改进闭环拆解

1. **谁当评判者？**
   仿真里，成功由环境按原始任务目标判定，agent 拿不到仿真器的物体状态和成功信号。修订时由强 agent Astra 当“老师”，复盘 Luna 的观测、干预和结果，并提出候选修订。接受规则（Eq.5）是：候选 playbook 让 Luna 在同一批 n 个开发实例上的经验成功率 ≥ η（Astra 自己在这批实例上的成功率），第一个达标的候选即被接受。评判误差的控制手段是同一轮复用相同实例、开发集和评测集分离、每版 playbook 冻结后再评测。小样本（每轮 8+8 个案例）带来的统计误差**未讨论**。真机成功由谁判定**未详述**。

2. **人工介入量？**
   真机前期需要人工：每个任务 50 条遥操作示范，用来微调 π0.5。真机每次试验前随机摆放物体（reset 方式**未详述**，推测由人完成）。playbook 的生成和修订全部由 LLM 完成，无人工标注。GR00T 版 playbook 迁移到真机时，由 Astra “adapt”，具体流程**未详述**。在 playbook 这一层可以算作“agent 间自我改进”，但模型权重和老师都是固定的，属于知识蒸馏 + 迭代 prompt 优化，不是严格意义上的 RSI。

3. **改进的是什么？**
   改进对象是 **playbook（文本 harness 指南）**，权重不动（VLA、Astra、Luna 都冻结）。干预接口有三类：改指令（≤512 字符）、对选定图像区域的视觉 token 加 attention-logit 偏置（强度 0–4）、修改末端和夹爪动作（每次 ≤5 步）。论文报告了三个版本：Initial → Intermediate → Refined。Luna 的成功率为：无 playbook 43.8%，Initial 31.3%（**初始版反而更差**），Refined 66.7%。精修版在 4 个任务上都提升了成功率，多数任务每回合 token 数下降。总轮数和是否已饱和**未详述**。对照实验中，Luna 给 Luna 当老师自我精修只有 22.9%，比不用 playbook 还低，说明弱模型自我迭代会走坏，需要强老师。

4. **安全与约束？**
   约束包括：接受阈值 η、每个修订版本单独存档（可追溯、可回退）、playbook 长度上限、开发 rollout 和修订尝试次数上限、动作修改步数上限。作者承认接受规则**不保证终止**。物理安全层（碰撞、力限制等）、回归测试、显式 rollback 机制都**未详述**。

5. **评测怎么做？**
   - 仿真：SimplerEnv Bridge 的 4 个 WidowX 任务，冻结 GR00T-N1.7，开发集和评测集各 48 个实例，留出评测。成功率：GR00T 41.7%，+Luna 43.8%，+Luna+Refined playbook 66.7%（+22.9 pp），Astra+playbook 79.2%。Luna+playbook 超过了不带 playbook 的 Astra。
   - 消融：去掉指令编辑掉得最多（成功率降到约 47.9%），去 attention 约 64.6%，去 action 约 62.5%（数值读自 Fig.3，标签对应关系以原图为准）。
   - 真机：Franka Panda + 微调 π0.5，3 个任务各 25 次。π0.5 单独 15/3/10（37.3%），Luna 无 playbook 全部 0/25（0%），Luna+playbook 18/13/17（64.0%）。
   - 成本：Luna $0.63/回合，Astra $3.43/回合（约 5.5 倍）。
   - 新任务泛化只体现在新初始配置和仿真到真机的 playbook 迁移，没有跨任务类别的测试。

## 对 Zhaoyuan 的启示
1. **playbook 的“WHEN / GUIDE / 事后检查”三段式**很适合用在 Code-as-Policies 的 RSI 里：把失败经验写成条件化规则，不去改代码或权重，便宜，也可版本化。
2. **修订要基于学生自己的 rollout 分布**（on-policy，类似 DAgger）：只看老师的轨迹不够。给弱执行器写 skill 库或 prompt 时，要用弱执行器的失败来修。
3. **注意崩塌信号**：初始 playbook 和自蒸馏都会让性能低于无 playbook 的水平。RSI 必须有“不低于基线才接受”的 gate，弱模型不宜给自己当老师。
4. 指令编辑贡献最大，说明“教 agent 怎么跟 VLA 说话”的收益很高。可以把它和动作校验结合：playbook 第 3 条提到，下达 OPEN 指令后仍要检查 VLA 实际输出。

## 来源
https://arxiv.org/abs/2609.33378
