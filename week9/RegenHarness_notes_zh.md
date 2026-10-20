# RegenHarness: A Robot Agent Harness with Evidence-Gated Recursive Self-Improvement 精读笔记

## 一句话概括
提出一个“证据门控”的机器人 agent harness：模型只负责提议，只有经独立验证、绑定身份与版本的结果才能通过 commit gate 写入任务状态；并在其外层加一个跨任务的 RSI 协议，用执行记录改 harness 配置，靠固定回归集 + 发布授权 + 版本回滚把关。真机部分只展示了四足机器人仓库巡检的案例，**RSI 没有给出任何定量结果**。

## 自我改进闭环拆解

1. **谁当评判者？**
   执行层面的评判者是 C3 verifier（参考实现是确定性的谓词 verifier，也可以接模型）加上 commit gate。verifier 的上下文（C3）**刻意排除 planner 自己的“完成”声明**，以避免相关性自我确认；后端自己报的 done 只能算证据，不能单独定案。全部节点 commit 后，还要由 goal verifier 复查原始目标：controller done ⇏ node success ⇏ goal success。判决分为 satisfied / violated / ambiguous / stale / unsafe 五类，证据缺失时 fail-closed。RSI 层面的评判者是固定回归集 R_j 上的接受谓词 A（安全/证据契约不退化、指定检查无回归、预先声明的目标指标有提升）加上“release authorization”。至于授权由人还是程序给出，**未详述**。作者也承认，传感器或模型共享的误差需要独立裁决。

2. **人工介入量？**
   单次任务内，恢复预算耗尽或出现 unsafe 时会升级到 safe stop → human handoff。RSI 中“发布授权”大概率需要人参与，但**未详述**。revision generator 由谁实现（LLM 还是人）也**未详述**。论文没有报告 reset 或干预频率。因此目前更接近“有人把关的配置迭代协议”，还谈不上自主 RSI。

3. **改进的是什么？**
   改的是 harness 配置，不改权重（明确排除在线权重更新），具体包括 context 选择规则、任务模板、路由偏好（Eq.4 打分权重）和有界恢复策略。每个候选都附带 change manifest：影响的组件、触发它的 traces、预期效果、回滚版本。**没有迭代轮数，也没有收益曲线**，6.8 节只给出评测协议（冻结版本 vs 候选、held-out 任务集，指标为 false completion、恢复成本、回归失败、发布开销）。饱和或崩塌的情况无从判断。

4. **安全与约束？**
   这是全文最扎实的部分：
   - identity + state-version 绑定的 commit gate，原子提交；
   - 固定回归集，覆盖 nominal、stale evidence、重复请求、中断运行、资源冲突几类；
   - updater **无权削弱**自身的接受谓词、commit gate 和底层安全联锁；
   - 新配置只在任务边界生效，进行中的任务钉住原版本，发布失败就回滚到上一个接受版本；
   - 运行时还有幂等派发、资源 lease、超时与取消、恢复预算；
   - safety veto 单调，上层只能加限制，不能减。

5. **评测怎么做？**
   只有真机案例研究，**没有基线对比，也没有消融数字**（消融只设计了，未报告结果）。四足机器人在办公室和仓库执行语音触发的巡检：导航 → 8 帧全景 → Qwen3-VL-Plus 分析（报告“箱子挡住通道”）→ 钉钉推送 → 返回 → 语音播报。任务执行耗时 259.94 s，完整语音交互 271.22 s。导航终点误差：去程 0.11 m（最后位姿 0.09），回程 0.17 m，闭环路线 0.25 m（最后位姿 0.15），到达半径设为 0.40 m，参考路径长 17.62 m 和 18.83 m。闭环案例中，机器人出发时已在 0.23 m 内（小于阈值），说明完成判定必须依赖历史（先离开再返回），不能只看终点距离。未涉及新任务泛化，也没有仿真。

## 对 Zhaoyuan 的启示
1. **把 observed 和 committed 分开**：Code-as-Policies 生成的代码跑完、返回 True，都不等于任务完成。可以照搬“controller done ⇏ node success ⇏ goal success”三级判定，再加 goal verifier 复查原始指令。
2. **verifier 的上下文要隔离**：给 VLM 判官喂的输入不要带 planner 自称的“已抓取”之类文字，只给前后观测和后置条件，减少自我确认偏差。
3. **RSI 改 harness 时，回归集里要放对抗性用例**（陈旧证据、重复派发、中断），同时规定 updater 不能改自己的 gate。这条可以直接写进我们 RSI 的规则。
4. **注意它的局限**：RSI 只有协议没有数据，引用时不要说成“已验证有效”。若要对比，需要自己补冻结版 vs 候选版的定量实验。

## 来源
https://arxiv.org/abs/2609.27612
