<!-- managed-by: paper-reading-companion -->
# 文献精读笔记

> 最后同步：2026-10-19 22:38 Asia/Shanghai

<!-- id-state: next-paper="5" -->

<!-- index:start -->
## 文献索引

| ID | 来源 | 标题 | 阅读进度 | 最近更新 |
|---|---|---|---|---|
| `p001` | `RTC.pdf` | Real-Time Execution of Action Chunking Flow Policies | 已完成初次同步；实验细节待逐图核验 | 2026-10-19 |
| `p002` | `cam.pdf` | Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection | 已完成初次同步；补充材料待逐项核验 | 2026-10-19 |
| `p003` | `diffusion.pdf` | Tutorial on Diffusion Models for Imaging and Vision | 已完成初次同步；图中文字未逐图核验 | 2026-10-19 |
| `p004` | `rax.pdf` | Scaling Online Exploration for Robot Agents | 已完成初次同步；预印本复现实验待核验 | 2026-10-19 |
<!-- index:end -->

<!-- paper:start id="p001" -->
## 文献：Real-Time Execution of Action Chunking Flow Policies

<!-- segment-id-state: next="2" -->

### 文献信息

- **Paper ID**：`p001`
- **Source identity**：`RTC.pdf`
- **来源**：[`./RTC.pdf`](./RTC.pdf)
- **作者 / 年份 / 发表处**：Kevin Black、Manuel Y. Galliker、Sergey Levine；arXiv:2506.07339v2（2025-12-05），论文标注为 NeurIPS 2025。
- **当前进度**：正文 §§1–6，p.1–10；附录中的消融和复现清单只做了结构性记录。
- **阅读状态**：已完成初次同步；尚未逐图复核原始 PDF 的曲线数值。

### 全文主线（滚动更新）

- **研究问题**：动作分块策略输出的下一个 chunk 生成时间可能超过控制周期，导致同步推理停顿，或异步切换时出现跨 chunk 的策略跳变和高加速度。
- **核心贡献**：提出只在推理时使用的 Real-Time Chunking（RTC），把下一动作块的生成写成带已知前缀约束的 flow inpainting；不需要重新训练基础 diffusion/flow VLA。
- **方法主线**：预测延迟 -> 冻结必然已经执行的前缀 -> 对剩余动作进行带软 mask 的引导去噪 -> 后台线程完成后原子替换当前 chunk。
- **关键结论**：在 Kinetix 的 12 个动态任务和基于 π0.5 的 6 个真实双臂任务中，RTC 对推理延迟更稳健，并提升任务吞吐；作者报告其在点火柴等精细任务上仍保持较高成功率。
- **主要限制**：额外的向量-Jacobian-product 计算增加了推理成本；方法只覆盖 diffusion/flow policy；真实实验没有覆盖腿式机器人。

### 分段精读

<!-- segment:start id="p001-s001" -->
#### 定位：§§2–6，p.2–10；Eq. (1)–(5)，Algorithm 1，Fig. 5–6

- **Segment ID**：`p001-s001`
- **定位别名**：`action chunking policy`、`Real-Time Chunking`、`soft masking`、`π0.5 real-world evaluation`
- **阅读范围**：从动作分块、推理延迟定义到 RTC 算法、模拟与真实机器人结果及讨论。
- **一句话核心**：把“新 chunk 必须接上旧 chunk 的已执行部分”转成 flow matching 的 inpainting 约束，从而在执行旧 chunk 的同时生成可连续的新 chunk。
- **在全文中的作用**：定义问题、推导方法、提供模拟和真实系统证据。

##### 要点与大意

1. 策略写作 \(\pi(A_t\mid o_t)\)，其中 \(A_t=[a_t,\ldots,a_{t+H-1}]\in\mathbb{R}^{H\times M}\)，\(H\) 是预测 horizon，\(M\) 是每个动作的维度。每次只执行前 \(s\le H\) 个动作。
2. 若从观测到新 chunk 的耗时为 \(\delta\)，控制周期为 \(\Delta t\)，论文定义离散推理延迟 \(d=\lfloor\delta/\Delta t\rfloor\)。异步推理会让新 chunk 的前 \(d\) 个位置已经错过，因此这些位置应当由旧 chunk 的剩余动作约束。
3. RTC 的 `GETACTION` 在控制线程中持续返回当前 chunk 的下一个动作；后台 `INFERENCE LOOP` 记录延迟、截取旧 chunk 的剩余部分并运行 `GUIDED INFERENCE`，完成后替换当前 chunk。
4. 软 mask 让前 \(d\) 个重叠动作权重为 1、最后不重叠区域权重为 0、中间区域指数衰减。这样既保留已确定动作，又允许未来动作根据新观测调整策略。
5. 模拟实验使用 Kinetix 的 12 个动态/随机任务、1M transition 数据、\(H=8\)，每个数据点 2048 次 rollout；比较 naive async、BID、temporal ensembling（TE）和 RTC。
6. 真实实验使用双 6-DoF 手臂、平行夹爪和 π0.5：\(H=50\)、\(\Delta t=20\) ms、5 次去噪；基线模型延迟约 76 ms，RTC 约 97 ms，加 LAN 和人为注入延迟后考察约 \(d\approx6,11,16\)。6 个任务共 480 个 episode，纯机器人执行时间约 28 小时。
7. 作者报告 RTC 在所有测试延迟下的平均吞吐最佳；同步推理随延迟线性退化，两个 TE 变体在 +100/+200 ms 延迟下会产生保护性停止。RTC 在不计推理暂停的控制步数上也常比同步推理更快完成任务。

##### 引用在本段中的作用

| 引用标识 | 所附着的命题 | 作用类型 | 作者态度 | 本地判定依据 | 置信度 |
|---|---|---|---|---|---|
| ΠGDM / training-free flow inpainting | 以梯度引导把生成结果匹配到带 mask 的目标 | 方法来源 | 扩展 | §3 将其改造成动作块约束，并加入权重裁剪 | 高 |
| BID | 跨 chunk 连续性的 rejection-sampling 基线 | 比较 | 中性 | §4 模拟中报告 RTC 的性能和计算量优势 | 高 |
| Temporal Ensembling | 用多个预测的平均动作平滑切换 | 比较 | 质疑其在多模态动作上的有效性 | Fig. 2、§4 的结果说明平均动作可能无效 | 高 |
| π0.5 | 真实实验的基础 VLA | 实验基座 | 采用 | §4.2 明确写出使用 π0.5 | 高 |

##### 数学原理与推导

- **适用性**：直接涉及。
- **符号与形状**：\(A_t^\tau\in\mathbb{R}^{H\times M}\) 为 flow 时间 \(\tau\) 的动作块，\(o_t\) 是观测，\(v_\pi(A_t^\tau,o_t,\tau)\) 与动作块同形状；展平后向量长度为 \(HM\)。\(W\in[0,1]^{H\times M}\) 是按时间位置广播到动作维度的 mask。
- **假设**：flow policy 的速度场可在推理时反向传播；旧 chunk 的剩余动作是执行时已知的软/硬观测；动作块的时间索引与控制周期对齐。
- **逐步推导**：
  1. 普通 flow Euler 更新为
     \[
     A_t^{\tau+1/n}=A_t^\tau+\frac{1}{n}v_\pi(A_t^\tau,o_t,\tau).
     \]
  2. 论文用一步去噪估计最终动作块
     \[
     A_t^{c1}=A_t^\tau+(1-\tau)v_\pi(A_t^\tau,o_t,\tau).
     \]
  3. 令 \(Y\) 为右填充到长度 \(H\) 的旧 chunk 剩余动作，用 \(Y-A_t^{c1}\) 表示当前结果违反已知前缀的误差；对 \(A_t^\tau\) 求向量-Jacobian-product，得到引导修正
     \[
     v^{\Pi\mathrm{GDM}}=v+\min\!\left(\beta,\frac{1-\tau}{2r_\tau^2}\right)
     (Y-A_t^{c1})^\top\!\operatorname{diag}(W)\frac{\partial A_t^{c1}}{\partial A_t^\tau},
     \qquad
     r_\tau^2=\frac{2(1-\tau)^2}{\tau^2+(1-\tau)^2}.
     \]
  4. 将修正后的速度场代回 Euler 更新，并在 \(\tau=0\to1\) 间迭代；\(\beta\) 用于裁剪引导强度，避免少量去噪步时不稳定。
- **直觉与边界情况**：\(W=1\) 的位置近似冻结，\(W=0\) 的位置由新观测自由生成；当 \(d\) 变大，冻结前缀变长，RTC 仍可运行但可用于反应新观测的自由区变短。TE 的逐维平均在多模态动作空间中可能落到任何有效动作模式之外。
- **与实现的对应**：`Aprev = Acur[s:]` 对应截去已执行部分；`compute W` 对应软 mask；`g = e · JVP` 对应自动微分；`Acur = Anew` 对应新 chunk 的原子替换。这里是论文 Algorithm 1 的实现映射，本地未检查作者代码仓库。

##### 代码实现

- **性质**：论文实际算法的精确伪代码映射；不是本地可运行实现。
- **代码位置**：论文 Algorithm 1，pp.5–6。
- **数据流与张量形状**：`Acur/Aprev/Anew: [H, M]`；`W: [H, 1]` 后广播到 `[H, M]`；`v, e: [H, M]`；展平后的 Jacobian 向量长度为 `H*M`。
- **实现**：

```python
def rtc_step(flow_policy, obs, prev_chunk, delay, executed, horizon, denoise_steps, beta):
    weights = soft_mask(horizon, delay, executed)       # [H, 1]
    prev = right_pad(prev_chunk, horizon)               # [H, M]
    action = normal_noise_like(prev)                    # [H, M]
    for tau in grid(0.0, 1.0, denoise_steps):
        def denoise(x):
            return x + (1.0 - tau) * flow_policy(x, obs, tau)
        estimate = denoise(action)
        error = prev - estimate
        jvp = vector_jacobian_product(estimate, action, error * weights)
        velocity = flow_policy(action, obs, tau)
        velocity = velocity + min(beta, (1 - tau) / (2 * r_tau_sq(tau))) * jvp
        action = action + velocity / denoise_steps
    return action
```

- **复杂度 / 数值稳定性**：每个去噪步额外需要一次反向自动微分，计算和显存开销高于直接采样；论文通过引导权重裁剪和有限去噪步缓解不稳定。
- **验证方法**：应分别检查 `W` 的三段取值、`Aprev` 右填充后的形状、无延迟时与普通 flow 采样的连续性，以及在固定随机种子下增加 `d` 时跨 chunk 加速度是否下降。当前只完成论文文本核对，未运行复现实验。

##### 证据、限制与延伸

- **直接证据**：Eq. (1)–(5)、Algorithm 1；Kinetix Fig. 5；真实机器人 Fig. 6；§6 限制说明。
- **关键假设与限制**：控制动作块可以被旧 chunk 的剩余前缀约束；flow/diffusion 模型允许推理时梯度引导；实验只覆盖 12 个模拟任务和 6 个真实双臂任务。
- **复现线索 / 缺失信息**：模拟器、训练脚本、π0.5 checkpoint、完整延迟分布和硬件控制接口未在本地目录提供；论文报告的任务吞吐和统计显著性需按附录及代码重新核验。
- **与其他段落或方法的联系**：RTC 与 MPC 一样并行计算与执行，但不需要显式动力学模型；与层级 VLA 正交，RTC 处理的是执行时延和 chunk 连续性。
- **待核验问题**：软 mask 的衰减常数如何影响不同动作维度；远程推理下网络抖动而非固定延迟时，`max(Q)` 估计是否过于保守。

<!-- segment-manual:start id="p001-s001" -->
##### 个人批注（本片段，插件保留）

<!-- 在这里自由记录；同步此片段时不会改写。 -->
<!-- segment-manual:end -->
<!-- segment:end -->

### 术语表

| 术语 | 原文 | 简明解释 | 首次定位 |
|---|---|---|---|
| 动作分块 | action chunking | 一次预测未来 \(H\) 个动作，只执行其中前 \(s\) 个 | §2 |
| 执行 horizon | execution horizon \(s\) | 每次 chunk 实际消费的动作数 | §2 |
| 推理延迟 | inference delay \(d\) | 新 chunk 从观测到可用期间经过的控制步数 | §2 |
| 软掩码 | soft masking | 对重叠动作使用从 1 到 0 的连续约束权重 | §3.2 |
| 向量-Jacobian-product | vector-Jacobian product | 将约束误差通过去噪映射反传到当前动作变量 | §3.1 |

### 复现清单

- [x] 数据与预处理：论文给出 Kinetix 专家策略和 1M transition 的设置
- [x] 模型与目标函数：flow matching 速度场 + ΠGDM 引导
- [x] 超参数与训练细节：模拟 \(H=8\)；真实 \(H=50\)、5 次去噪；其余需查附录
- [x] 指标与基线：solve rate、任务吞吐、累计进度；naive async、BID、TE、同步推理
- [ ] 计算资源与成本：真实实验时间给出，完整 GPU/CPU 成本待核验
- [ ] 代码、随机种子与版本：本地未发现作者实现

### 开放问题

- 在硬实时系统中，自动微分引导的最坏情况延迟如何保证？
- 当旧动作本身由于观测错误而不可靠时，冻结约束是否会放大错误？
- RTC 能否与离散 FAST 动作 token 或非 flow 的 autoregressive policy 组合？

<!-- paper-manual:start id="p001" -->
### 个人批注（插件保留此区域）

<!-- 在这里自由记录；同步时不会改写。 -->
<!-- paper-manual:end -->
<!-- paper:end -->

<!-- paper:start id="p002" -->
## 文献：Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection

<!-- segment-id-state: next="2" -->

### 文献信息

- **Paper ID**：`p002`
- **Source identity**：`cam.pdf`
- **来源**：[`./cam.pdf`](./cam.pdf)
- **作者 / 年份 / 发表处**：Enshen Zhou、Qi Su、Cheng Chi、Zhizheng Zhang、Zhongyuan Wang、Tiejun Huang、Lu Sheng、He Wang；arXiv:2412.04455v3（2025-03-21）。
- **当前进度**：正文 §§1–5，p.1–9；附录中的数据构造和详细任务表未逐项复核。
- **阅读状态**：已完成初次同步。

### 全文主线（滚动更新）

- **研究问题**：机器人既要在失败发生后反应式检测，也要在失败发生前主动式预防；直接频繁调用 VLM 往往慢、缺少精确 3D 时空关系。
- **核心贡献**：提出 Code-as-Monitor（CaM），把两类失败统一成时空约束满足问题，并让 VLM 生成一次可执行 monitor code，在跟踪到的约束元素上持续计算。
- **方法主线**：Constraint Generator 生成子目标和约束 -> Constraint Painter 用 ConSeg、深度和多视角图像产生点/线/面元素 -> Constraint Monitor 生成并执行 Python/NumPy 监控代码 -> 失败原因反馈给策略重新规划。
- **关键结论**：作者报告 CaM 在三个模拟器和真实场景中提升开放集失败检测的成功率和执行效率；摘要给出的严重扰动结果是成功率提高 28.7%、执行时间减少 31.8%。
- **主要限制**：约束元素的分割、3D 跟踪和代码生成仍会出错；实验中的 ConSeg 训练数据主要来自 pick-and-place，实时监控依赖已有跟踪器和相机深度质量。

### 分段精读

<!-- segment:start id="p002-s001" -->
#### 定位：§§3–5，p.3–9；Eq. (1)–(2)，Fig. 2–6，Tables 1–7

- **Segment ID**：`p002-s001`
- **定位别名**：`Code-as-Monitor`、`Constraint Generator`、`constraint elements`、`ConSeg`
- **阅读范围**：方法三模块、约束元素构造、ConSeg、监控代码、模拟/真实实验和消融。
- **一句话核心**：用紧凑的几何元素把 VLM 的语义理解变成可跟踪、可计算、可解释的约束检查。
- **在全文中的作用**：提出统一框架并给出开放集失败检测证据。

##### 要点与大意

1. 给定全局任务指令、前一子目标、失败反馈和两视角 RGB-D 观测，Constraint Generator（论文中使用 GPT-4o）输出下一子目标、执行期间必须保持的约束 \(C_d\) 和完成时必须满足的约束 \(C_u\)。
2. Painter 不把原始图像直接交给监控器，而是用约束感知分割得到实例级/部件级 mask，将多视角深度投影到 3D 点云，再体素化、聚类和连接成点、线、面等 constraint element。
3. Monitor 只在子目标开始时调用 VLM 生成代码。代码接收元素的历史 3D 位置，返回 `(boolean, textual reason)`；返回 False 时立即停止当前策略并把原因用于下一轮重规划。
4. ConSeg 由冻结的 VLM/视觉编码器和 SAM 风格解码器组成，训练时仅优化额外 LoRA 等参数。论文从 BridgeData V2 采样 10,181 条轨迹、219,356 张图像，使用 GPT-4o、Grounded SAM 和 Semantic SAM 形成训练标注；模拟器各收集 100 条轨迹做微调，真实部署的 ConSeg 不做微调。
5. CLIPort、OmniGibson、RLBench 和真实 UR5 + Leap Hand 覆盖点、线、面约束、工具使用、关节物体和动态干扰。作者报告在 CLIPort 严重扰动下 CaM 比 DoReMi 成功率高，在 OmniGibson 还减少执行时间和 token 使用；多视角、约束感知分割和连接元素的消融均有明显影响。

##### 引用在本段中的作用

| 引用标识 | 所附着的命题 | 作用类型 | 作者态度 | 本地判定依据 | 置信度 |
|---|---|---|---|---|---|
| DoReMi | 频繁 VLM 查询进行中间失败检测的基线 | 比较 | 认为其速度和 3D 判断受限 | §4 表 1–2 及文字分析 | 高 |
| LISA / PixelLM | 多粒度视觉分割比较对象 | 比较 | 中性比较 | Table 5 的 ReasonSeg/ConstraintSeg 结果 | 高 |
| ReKep | OmniGibson 的开放环控制基座之一 | 实验基线 | 中性 | §4.1 实验设定 | 高 |
| BridgeData V2 | 轨迹数据来源 | 数据集 | 采用 | §3.2 数据构造说明 | 高 |

##### 数学原理与推导

- **适用性**：直接涉及；核心是几何约束计算而非训练损失。
- **符号与形状**：
  - \(O\)：两视角 RGB-D 观测；\(L_{global}\)：全局指令；\(l_{pre}\)：前一子目标；\(f_{pre}\)：上一轮反馈。
  - 生成器输出
    \[
    (l_{next},C_d,C_u)=F_{VLM}(O,L_{global},l_{pre},f_{pre}).
    \]
  - 监控代码的附录接口约定：`end_effector` 为 \((T,3)\)，`element_position` 为 \((T,E,K,3)\)，其中 \(T\) 是历史时间长度、\(E\) 是元素数量、\(K\) 是每个元素的点数。
  - ConSeg 的核心映射为
    \[
    (l_e,h_{<SEG>})=F_{VLM}(c,o),\qquad
    M=F_{dec}(F_{MLP}(h_{<SEG>}),F_{enc}(o)),
    \]
    其中 \(c\) 是文本约束，\(o\) 是单视角图像，\(M\) 是实例级或部件级 mask。
- **假设**：深度可将 mask 投影到可信的 3D 坐标；跟踪器能在相邻帧保持元素身份；约束可以用点、线、面和历史位置的算术关系表达。
- **逐步推导**：例如“锅面保持水平”先由 ConSeg 提取锅面点集 \(p_0,p_1,p_2\)，构造 \(n=(p_1-p_0)\times(p_2-p_0)\)，归一化后检查 \(\arccos(n\cdot[0,0,1])\le\tau_{angle}\)；这把语义约束转成每帧可复用的布尔判断。
- **直觉与边界情况**：元素减少了与约束无关的像素和几何细节，便于代码计算；但元素点不足、遮挡或深度错位会让线/面退化为错误的点集。单视角消融的平均成功率从 65% 降至 40%，说明遮挡和深度歧义是实际瓶颈。
- **与实现的对应**：约束生成是低频 VLM 调用，元素跟踪和 NumPy 计算是高频闭环；因此代码执行阶段无需每帧再次调用 VLM。论文未提供本地可直接运行的实现仓库。

##### 代码实现

- **性质**：论文要求 VLM 生成的 monitor code 形式；下面是按其输入/输出契约写的最小实现草图。
- **代码位置**：论文 Fig. 2、Fig. 24 和 Appendix D.4 的接口说明。
- **数据流与张量形状**：`element_position[T, E, K, 3]` -> 选取元素/点 -> 向量差、叉积、点积 -> 标量约束 -> `(bool, reason)`。
- **实现**：

```python
import numpy as np

def constraint_monitor(end_effector, element_position, is_finished):
    # element 0: pan surface with K >= 3 points; element 1: pan handle or lobster
    surface = element_position[:, 0, :, :]       # [T, K, 3]
    p0, p1, p2 = surface[-1, 0], surface[-1, 1], surface[-1, 2]
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return False, "Pan surface points are degenerate."
    normal = normal / norm
    angle = np.degrees(np.arccos(np.clip(normal @ np.array([0., 0., 1.]), -1., 1.)))
    if not is_finished and angle > 15.0:
        return False, f"Pan is tilted by {angle:.1f} degrees."
    return True, "No failure detected."
```

- **复杂度 / 数值稳定性**：每帧对每个元素执行线性数量的点运算；点积进入 `arccos` 前需要 `clip`，法向量范数过小时需拒绝判断。VLM 代码生成错误仍可能造成逻辑漏洞，因此论文要求返回原因并在外层停止。
- **验证方法**：检查 `element_position` 的四维形状、平面点共线和法向量方向翻转两种情况；用合成平面分别注入 0°、超过阈值和退化点集；再测跟踪丢失时是否安全返回 False。当前未运行论文代码。

##### 证据、限制与延伸

- **直接证据**：Eq. (1)–(2)、Fig. 2–3、Table 1–7；摘要报告的 28.7% 成功率提升和 31.8% 执行时间减少。
- **关键假设与限制**：约束元素可由视觉和深度稳定恢复；作者没有把“失败判断准确率”作为独立标准指标，因为代码在时间上连续执行；现实系统仍受相机、跟踪和开放环策略影响。
- **复现线索 / 缺失信息**：需要 ConSeg 权重、数据标注流水线、GPT-4o 版本、三种模拟器和真实 UR5/Leap Hand；本地目录只有 PDF，没有这些依赖。
- **与其他段落或方法的联系**：CaM 是一种 Code-as-Policy 风格的监控层，可与开放环控制策略组成闭环；它把 VLM 放在“生成验证器”位置，而不是每个控制周期直接输出动作。
- **待核验问题**：VLM 生成代码的错误率和安全审计如何量化；元素跟踪丢失时是否总能触发保守停止；约束元素能否覆盖力/触觉等非视觉信号。

<!-- segment-manual:start id="p002-s001" -->
##### 个人批注（本片段，插件保留）

<!-- 在这里自由记录；同步此片段时不会改写。 -->
<!-- segment-manual:end -->
<!-- segment:end -->

### 术语表

| 术语 | 原文 | 简明解释 | 首次定位 |
|---|---|---|---|
| 反应式失败检测 | reactive failure detection | 失败已经发生后识别并停止/重规划 | §1 |
| 主动式失败检测 | proactive failure detection | 监测即将导致失败的状态并提前纠正 | §1 |
| 约束元素 | constraint element | 代表实体或部件的点、线、面等紧凑几何对象 | §3.2 |
| 监控代码 | monitor code | VLM 生成、持续计算约束并返回原因的程序 | §3.3 |
| 约束感知分割 | constraint-aware segmentation | 根据文本约束选择实例级和部件级 mask | §3.2 |

### 复现清单

- [x] 数据与预处理：BridgeData V2 采样、GPT-4o 子目标/约束标注、Grounded/Semantic SAM
- [x] 模型与目标函数：ConSeg + 多视角深度投影 + VLM monitor code
- [x] 超参数与训练细节：模拟器各 100 条轨迹微调；真实 ConSeg 不微调
- [x] 指标与基线：成功率、执行时间、token 使用；DoReMi、Inner Monologue 等
- [ ] 计算资源与成本：本地未核验
- [ ] 代码、随机种子与版本：本地未发现作者实现

### 开放问题

- 如何对自动生成的约束代码做形式验证或安全沙箱限制？
- 对动态遮挡、透明物体和接触力约束，点/线/面抽象是否足够？
- 监控层如何与低层控制器的稳定性证明结合？

<!-- paper-manual:start id="p002" -->
### 个人批注（插件保留此区域）

<!-- 在这里自由记录；同步时不会改写。 -->
<!-- paper-manual:end -->
<!-- paper:end -->

<!-- paper:start id="p003" -->
## 文献：Tutorial on Diffusion Models for Imaging and Vision

<!-- segment-id-state: next="2" -->

### 文献信息

- **Paper ID**：`p003`
- **Source identity**：`diffusion.pdf`
- **来源**：[`./diffusion.pdf`](./diffusion.pdf)
- **作者 / 年份 / 发表处**：Stanley Chan；arXiv:2403.18103v3（2025-01-08）。
- **当前进度**：正文 §§1–6，p.1–87；目录、定义、主要推导和结论已按文本提取同步。
- **阅读状态**：已完成初次同步；PDF 图中文字和全部参考文献未逐项视觉核验。
- **同目录辅助材料**：[`./diffusion-全文中文翻译.md`](./diffusion-全文中文翻译.md) 是已有中文翻译稿，本笔记仍以 `diffusion.pdf` 作为唯一来源身份，不把译稿作为独立论文。

### 全文主线（滚动更新）

- **研究问题**：从 VAE 的潜变量和 ELBO 出发，解释 DDPM、DDIM、分数匹配、SDE、Langevin 和 Fokker–Planck 方程如何共同构成扩散生成模型。
- **核心贡献**：教程以逐步概率推导和物理直觉连接离散马尔可夫链、连续时间随机过程以及数值求解器。
- **方法主线**：VAE/ELBO -> 前向加噪与反向高斯转移 -> 噪声预测 -> DDIM 非马尔可夫采样 -> score matching/Langevin -> VP/VE SDE 与反向 score 项 -> Fokker–Planck 的概率流解释。
- **关键结论**：DDPM 可看作 VAE 结构、ELBO 和重参数化在多步潜变量链上的扩展；DDPM、SMLD、VP-SDE 和 VE-SDE 在合适参数化下可用统一的 score/SDE 视角理解。
- **主要限制**：教程主要讲基础理论和一维/低维直觉，不能替代具体图像架构、采样器、条件控制和大规模工程实现的复现细节。

### 分段精读

<!-- segment:start id="p003-s001" -->
#### 定位：§§1–6，p.2–87；Eq. (1.4)–(1.7)、(2.1)、(2.14)、(2.44)、(2.59)、(3.1)、(3.3)、(4.10)–(4.20)、(5.43)–(5.47)

- **Segment ID**：`p003-s001`
- **定位别名**：`VAE/ELBO`、`DDPM`、`DDIM`、`SMLD`、`VP/VE SDE`、`Fokker–Planck`
- **阅读范围**：教程完整正文的概念链和主要方程；不含参考文献逐条内容解读。
- **一句话核心**：扩散模型通过一条可控的前向加噪链把数据推向简单先验，再学习其反向去噪/score 动力学；VAE、ELBO、Langevin 和 SDE 是同一概率生成问题的不同表达。
- **在全文中的作用**：基础教程和推导主线。

##### 要点与大意

1. VAE 用编码器近似 \(q_\phi(z\mid x)\)，解码器定义 \(p_\theta(x\mid z)\)。核心分解是 \(\log p(x)=\mathrm{ELBO}(x)+D_{KL}(q_\phi(z\mid x)\|p(z\mid x))\)，所以最大化 ELBO 同时提高似然下界并压缩后验近似误差。
2. 变分扩散把一个潜变量改成 \(x_0,x_1,\ldots,x_T\) 链。前向转移是
   \[
   q_\phi(x_t\mid x_{t-1})=\mathcal N\!\left(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I\right),
   \]
   其中 \(0<\alpha_t<1\)。记 \(\bar\alpha_t=\prod_{i=1}^t\alpha_i\)，可直接采样
   \[
   x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
   \]
3. 反向后验 \(q(x_{t-1}\mid x_t,x_0)\) 仍为高斯，其均值是 \(x_t\) 和 \(x_0\) 的线性组合；DDPM 用网络 \(\mu_\theta(x_t,t)\) 或噪声预测网络 \(\epsilon_\theta(x_t,t)\) 参数化反向过程。
4. DDPM 的训练可理解为随机采样 \(x_0,t,\epsilon\)，构造 \(x_t\)，再用网络预测 \(\epsilon\) 或等价的 \(x_0/\mu\)。推理从 \(x_T\sim\mathcal N(0,I)\) 出发，按 \(T,T-1,\ldots,1\) 逐步采样。
5. DDIM 保持相同边缘分布 \(q(x_t\mid x_0)\)，但放弃严格的 Markov 转移，使用可调的 \(\sigma_t\) 和预测噪声构造更少步数的非马尔可夫采样；随机性为零时得到确定性路径。
6. SMLD 的离散 Langevin 更新是
   \[
   x_{t+1}=x_t+\tau s_\theta(x_t)+\sqrt{2\tau}\,z,\qquad z\sim\mathcal N(0,I),
   \]
   其中 Stein score \(s_\theta(x)=\nabla_x\log p_\theta(x)\) 对数据坐标求导，不要和对参数求导的普通 score 混淆。
7. 加噪条件分布下的 denoising score matching 让网络学习 \(s_\theta(x+\sigma z)\approx-z/\sigma\)，把不可访问的真实 score 转换成可监督的噪声预测问题。
8. 连续时间下，前向 diffusion SDE 写成
   \[
   dx=f(x,t)dt+g(t)dw,
   \]
   反向时间 SDE 包含 \(-g(t)^2\nabla_x\log p_t(x)\) 的 score 修正。DDPM 对应 variance-preserving（VP）SDE，SMLD 对应 variance-exploding（VE）SDE。
9. Fokker–Planck 方程描述随机过程的密度随时间的演化；在平衡且概率流为零时，Langevin 动力学的稳态密度可恢复目标分布（满足教程中的正则性和步长假设）。

##### 引用在本段中的作用

| 引用标识 | 所附着的命题 | 作用类型 | 作者态度 | 本地判定依据 | 置信度 |
|---|---|---|---|---|---|
| Kingma & Welling / Doersch | VAE、潜变量与 ELBO | 理论基础 | 采用并教学化展开 | §1 的定义、例子与 ELBO 推导 | 高 |
| Ho et al. | DDPM 的扩散和去噪框架 | 方法来源 | 采用 | §2 的转移、ELBO 和训练/推理 | 高 |
| Song et al. | score-based model、SMLD、SDE | 方法来源/统一视角 | 采用并连接 | §§3–4 | 高 |
| Vincent | denoising score matching | 理论基础 | 采用 | §3.3 的 DSM 与 ESM 等价性 | 高 |

##### 数学原理与推导

- **适用性**：直接涉及。
- **符号与形状**：若图像被展平为 \(x_0\in\mathbb R^D\)，则每个 \(x_t,\epsilon,s_\theta(x_t)\in\mathbb R^D\)；协方差通常是 \(D\times D\) 的标量倍数单位阵；时间 \(t\) 可为离散索引或连续变量。
- **假设**：正向噪声日程已知；反向高斯或 score 网络容量足够；采样步长满足数值稳定性；密度与边界条件使 KL、积分分部和 Fokker–Planck 推导成立。
- **逐步推导**：
  1. VAE 先由 \(\log p(x)=\mathbb E_q[\log p(x,z)/q(z\mid x)]+D_{KL}(q\|p(z\mid x))\) 得到 ELBO。
  2. 扩散链把单个潜变量替换成 \(x_{1:T}\)，对前向链应用同一变分分解，得到初始块重建项、终点先验匹配项和每个转移块的 KL 一致性项。
  3. 由闭式前向采样把网络输入写成 \(x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon\)，再把 \(\mu_\theta\) 重参数化为噪声预测，训练目标变为带时间权重的 \(\|\epsilon_\theta(x_t,t)-\epsilon\|_2^2\)。
  4. 将离散更新的极限写为 SDE，并由反向时间公式得到 score 项；对该随机过程的密度求守恒方程即得到 Fokker–Planck 形式。
- **直觉与边界情况**：\(\bar\alpha_t\to1\) 时 \(x_t\) 接近数据，\(\bar\alpha_t\to0\) 时接近纯高斯；DDIM 的少步采样以相同边缘为目标但路径不同；Langevin 中去掉噪声只剩沿 log-density 梯度上升，可能陷入单个模态。
- **与实现的对应**：训练中随机抽取 `t`、`epsilon` 并一次构造 `x_t`，避免逐步生成整条前向链；推理循环从纯噪声反向更新。教程未绑定某个具体 U-Net、Transformer 或框架实现。

##### 代码实现

- **性质**：基于论文公式的最小实现草图；不是作者仓库代码。
- **代码位置**：论文 §§2.4–2.6 和 §3.1。
- **数据流与张量形状**：`x0, xt, eps: [B, D]`（图像也可为 `[B,C,H,W]`，公式按展平向量解释）；`t: [B]`；`eps_theta: [B,D]`。
- **实现**：

```python
def q_sample(x0, alpha_bar_t, eps):
    # alpha_bar_t: [B] -> broadcast to x0; all tensors share x0 shape [B, D]
    a = alpha_bar_t.sqrt().unsqueeze(-1)
    b = (1.0 - alpha_bar_t).sqrt().unsqueeze(-1)
    return a * x0 + b * eps

def ddpm_noise_loss(model, x0, alpha_bar_t, t):
    eps = torch.randn_like(x0)
    xt = q_sample(x0, alpha_bar_t, eps)
    eps_hat = model(xt, t)
    return ((eps_hat - eps) ** 2).flatten(1).mean(dim=1).mean()
```

- **复杂度 / 数值稳定性**：训练每个样本只需一次网络前向；推理成本与反向步数成正比。应缓存 \(\sqrt{\bar\alpha_t}\)、\(\sqrt{1-\bar\alpha_t}\)，并在极端 \(\bar\alpha_t\) 下避免负数平方根和过小方差。
- **验证方法**：检查 `q_sample` 的两个系数平方和为 1；用一维高斯验证前向方差；比较 DDPM 多步采样与 DDIM 少步采样的边缘统计；用有限差分检查 score 网络的梯度方向。当前未运行实现。

##### 证据、限制与延伸

- **直接证据**：教程 §§1–6 的定义、定理、Eq. (1.4)–(1.7)、(2.1)、(2.14)、(2.44)、(2.59)、(3.1)、(3.3)、(4.10)–(4.20)、(5.43)–(5.47)。
- **关键假设与限制**：教程中的“收敛到目标分布”依赖无限/足够长采样、合适步长和正则性；实际神经网络、有限步数和高维数据会引入近似误差。
- **复现线索 / 缺失信息**：缺少现代图像实验的具体数据集、网络结构、优化器、噪声日程和硬件配置；同目录翻译稿有正文/图注中文化，但未作为逐字完整性证据使用。
- **与其他段落或方法的联系**：RTC 使用 flow matching 的速度场和 inpainting；本教程的 DDPM/SMLD/SDE 提供理解其噪声路径、score/速度场和数值积分的基础。
- **待核验问题**：教程中不同章节对 \(\alpha_t\) 与累计 \(\bar\alpha_t\) 的记号需以原 PDF 公式逐页核对；工程上采用的噪声参数化、v-prediction 和 classifier-free guidance 不在本篇核心范围内。

<!-- segment-manual:start id="p003-s001" -->
##### 个人批注（本片段，插件保留）

<!-- 在这里自由记录；同步此片段时不会改写。 -->
<!-- segment-manual:end -->
<!-- segment:end -->

### 术语表

| 术语 | 原文 | 简明解释 | 首次定位 |
|---|---|---|---|
| 证据下界 | evidence lower bound, ELBO | 对数似然的可优化下界，差距是后验 KL | §1.2 |
| 前向扩散 | forward diffusion | 逐步向数据加入高斯噪声 | §2 |
| score | Stein’s score | 对数据坐标的 \(\nabla_x\log p(x)\) | §3.2 |
| 去噪 score matching | denoising score matching | 用带噪样本和已知条件高斯 score 训练网络 | §3.3 |
| 方差保持/爆炸 | VP/VE SDE | 两种连续时间噪声调度的统一表达 | §4.3 |

### 复现清单

- [x] 数据与预处理：教程给出抽象的 \(x_0\)、噪声日程和采样流程
- [x] 模型与目标函数：ELBO、噪声预测、score matching
- [x] 超参数与训练细节：概念性说明；现代图像任务细节缺失
- [x] 指标与基线：以推导、示例分布和采样轨迹为主，不是统一 benchmark
- [ ] 计算资源与成本：未提供可复现实验配置
- [ ] 代码、随机种子与版本：未提供绑定实现

### 开放问题

- 有限步数下 DDPM、DDIM、ODE solver 的误差如何在高维图像上比较？
- score、flow velocity 和噪声预测参数化之间的数值条件如何统一分析？
- 如何把教程中的 SDE 直觉用于机器人动作空间的非高斯约束？

<!-- paper-manual:start id="p003" -->
### 个人批注（插件保留此区域）

<!-- 在这里自由记录；同步时不会改写。 -->
<!-- paper-manual:end -->
<!-- paper:end -->

<!-- paper:start id="p004" -->
## 文献：Scaling Online Exploration for Robot Agents

<!-- segment-id-state: next="2" -->

### 文献信息

- **Paper ID**：`p004`
- **Source identity**：`rax.pdf`
- **来源**：[`./rax.pdf`](./rax.pdf)
- **作者 / 年份 / 发表处**：Runhan Huang、Xiangcheng Zhang、Haoyu Chen、Yitong Li、Yen-Jen Wang、Bohan Lyu、Haotian Ye、Chuye Hong、Yixuan Wang、Chloe Su、Haoyu Xiong、Yangzhen Wu、Enze Xie、Travis Manderson、Stefano Ermon、Pieter Abbeel、Antonio Torralba、Dawn Song、Weirui Ye、Yilun Du 等；arXiv:2610.22496v1（2026-10-15）。
- **当前进度**：正文 §§1–6，p.5–30；实验表格和若干附录代码片段已查看，未检查开源仓库。
- **阅读状态**：已完成初次同步；该版本是较新的预印本，结果和项目页应视为待复核。

### 全文主线（滚动更新）

- **研究问题**：传统机器人学习依赖每个新 embodiment/任务的遥操作动作示范，策略的泛化和持续自我改进受 imitation distribution 限制。
- **核心贡献**：提出 RAX（Robot Agent eXploration），让通用 coding agent 直接通过机器人 API 在线探索，生成并调试包含感知、规划、控制和检查的可执行程序；不训练动作 policy，也不依赖手工 action primitive 或 value function guidance。
- **方法主线**：多模态感知和 3D 世界模型 -> 通过工具/IK/运动规划搜索动作 -> 把成功决策编译成程序 -> 以 precondition、verification 和日志形成版本化技能库 -> 跨任务组合和持续适应。
- **关键结论**：作者报告在线探索预算增加时仿真成功率稳定上升；在 HumanoidArena、DexJoCo、LIBERO/LIBERO-Pro 等基准和真实机器人上，RAX 用很少或没有动作示范达到较高成功率，并观察到主动感知、错误恢复和安全护栏等行为。
- **主要限制**：性能依赖 coding agent、工具接口、传感器和探索预算；真实系统结果需要昂贵硬件执行和人工安全边界；本地只验证了 PDF 文本，未验证项目页或代码仓库。

### 分段精读

<!-- segment:start id="p004-s001" -->
#### 定位：§§1–6，p.5–30；Fig. 1–12，Tables 4–10

- **Segment ID**：`p004-s001`
- **定位别名**：`scaling online exploration`、`frontend–backend separation`、`verified skills`、`continual learning`
- **阅读范围**：认知架构、系统边界、仿真/真实实验、从 agent actions 到 autonomous program 的编译和技能复用。
- **一句话核心**：RAX 把机器人学习的“参数更新”换成在线试错后写入、验证、组合的程序更新，并通过扩大探索预算提升泛化。
- **在全文中的作用**：提出系统范式并给出跨 embodiment 的实验证据。

##### 要点与大意

1. RAX 接收自然语言任务、少量目标图像/视觉示范和探索预算；语言模型参数固定，不训练 VLA 或 value function，学习结果写入 workspace 中的代码、笔记和 verified skills。
2. 前端是 coding agent 的工作空间，负责观察、决策和编程；后端拥有传感器、驱动、低层控制器和安全限制。双方只通过工具接口相连，agent 不能修改后端控制器。
3. 工具暴露 RGB-D、深度点云、相机位姿、正/逆运动学、运动规划、机器人模型预览和力信号等能力。agent 通过点云拟合/几何测量建立世界模型，在 3D 位姿和关节空间中搜索可行动作。
4. 成功探索过程被编译为可独立执行的 Python 程序，程序包含阶段化技能、前置条件、检查点、异常处理和 retry；每次无人值守成功及其日志才允许进入版本化技能仓库。
5. 技能仓库支持跨 session 的持续学习和长时任务组合。机器人可以先练习单个子任务，再将已验证的 chair、cup、basket、box 等技能组合成移动操作长任务。
6. 论文中的量化结果包括：仿真平均相对最强基线提高 37.4%；HumanoidArena 最大预算下 RAX 平均 89.8%，WB-WAM 为 81.9%；DexJoCo 32 次探索的平均成功率 96.7%；LIBERO 原始布局平均 98.5%，位置扰动下无适应 76.0%，最多 5 次适应后 84.4%。
7. 从 agent 到程序的真实例子中，air-fryer 任务从 agent 的 72.3 分钟推理时间降到程序的 18 秒（论文图表约 241×），Bandu tower 从 17.2 分钟降到 10 秒（约 103×）；机器人连续执行时间仍需保留。
8. 真实任务显示 agent 会主动移动相机/手进行测量、用接触探测校正透明板高度、在碰撞后增加 workspace clearance 检查，并在场景变化时只修改失效的模块。

##### 引用在本段中的作用

| 引用标识 | 所附着的命题 | 作用类型 | 作者态度 | 本地判定依据 | 置信度 |
|---|---|---|---|---|---|
| VLA/WAM 工作 | 说明遥操作数据和 imitation policy 的局限 | 背景/比较 | 对比 | §1 对现有范式的描述 | 高 |
| LIBERO、DexJoCo、HumanoidArena、BEHAVIOR-1K | 跨 embodiment 评测 | 数据集/证据 | 采用 | §5 的设置和表 3–6 | 高 |
| WB-WAM、π0.5、Diffusion Policy、ACT 等 | 与动作示范训练的策略比较 | 比较 | 中性比较 | 实验表的 baseline | 高 |
| Sutton, The Bitter Lesson | 支持扩大经验和搜索的动机 | 背景/理论立场 | 引用其观点 | 引言开头引语及后文论证 | 中 |

##### 数学原理与推导

- **适用性**：本段没有单一训练目标；核心是程序状态、工具调用和实验指标。
- **符号与形状**：观测由 RGB-D、相机内参/位姿、关节状态和力信号组成；具体张量维度随 embodiment 变化，论文没有给出一个统一的 \(o_t\in\mathbb R^d\) 定义。程序输入是结构化状态和图像，输出是工具调用或控制目标。
- **假设**：后端工具和安全检查可信；探索环境与测试环境有足够可迁移的结构；成功程序中的 precondition/verification 能识别失效；探索预算足以覆盖关键策略分支。
- **逐步推导**：
  1. 在线 episode 产生观测、工具调用和结果日志。
  2. agent 根据成功/失败修改感知、规划和控制代码，并在新 episode 中回归测试。
  3. 通过阶段级检查将失败定位到具体 skill；通过无人值守成功证据把程序提交到技能库。
  4. 长时任务调用已有技能，并用当前场景的感知数据参数化它们；若 precondition 失败，则抛出异常回到 agent 进行修复。
- **直觉与边界情况**：程序显式保存决策逻辑，因而比只保存动作轨迹更容易解释、调试和适应布局变化；但程序也可能过拟合探索场景，安全护栏并不能替代系统级验证。
- **与实现的对应**：论文 Fig. 4 的 frontend/backend 边界是系统关键；本地未检查论文声明的 open-source RAX 实现，因此不把论文中的接口描述当作已安装代码。

##### 代码实现

- **性质**：论文系统流程的实现草图；不是作者代码的复制。
- **代码位置**：论文 §§3–4、Fig. 4、Fig. 6；真实任务程序示例在 §5.3–5.5 和附录。
- **数据流与张量形状**：`observe()` 返回结构化状态/图像；`perceive()` 生成对象模型和点云；`plan()` 返回轨迹；`verify()` 读取执行后传感器证据；各 embodiment 的关节、末端位姿和力维度不同。
- **实现**：

```python
def explore_and_compile(task, tools, budget, skill_repo):
    program = None
    for episode in range(budget):
        obs = tools.observe()
        decision = agent_reason(task, obs, tools, program)
        result = tools.execute(decision)
        program = revise_program(program, decision, result)
        if result.success and program.passes_preconditions() and program.verify(tools):
            log = tools.collect_evidence()
            skill_repo.commit(program, preconditions=program.preconditions,
                              verification=program.verification, evidence=log)
    return program
```

- **复杂度 / 数值稳定性**：探索成本随 episode、工具调用和真实机器人时间增长；程序执行阶段可把高频感知/控制从 LLM 推理中移出，但每个几何模型、IK、碰撞和力阈值都需要数值鲁棒性检查。
- **验证方法**：在未见布局上单独运行编译后的程序；记录每个阶段的 precondition、动作、传感器证据和异常；检查安全限制拒绝越界命令；比较 agent 交互时间与程序执行时间。当前未运行 RAX。

##### 证据、限制与延伸

- **直接证据**：Fig. 1 的跨基准总览；Table 4（HumanoidArena）、Table 5（DexJoCo）、Table 6（LIBERO/LIBERO-Pro）、Tables 9–10（从 agent 到程序、无动作示范）以及 §5.4–5.5 的主动感知/持续学习案例。
- **关键假设与限制**：论文结果依赖固定的 coding agent、工具 API、模拟器/机器人和探索预算；“不需要动作示范”不等于不需要目标图像、任务文本或预置工具/安全后端。
- **复现线索 / 缺失信息**：需要项目页、代码、具体模型版本、提示词、工具接口、传感器标定、探索 episode 预算和硬件日志；这些未在本地目录核验。
- **与其他段落或方法的联系**：RAX 与 CaM 都把语言模型放到程序生成/验证层；CaM 是低成本监控器，RAX 则扩展到感知、规划、控制和技能库。与 VLA/π0.5 的差别在于 RAX 不更新动作策略参数，而是更新程序和记忆。
- **待核验问题**：探索预算与成功率的 scaling 是否跨 agent 模型稳定；技能库提交标准能否阻止偶然成功；在真实机器人上如何把 LLM 生成程序放进可审计的安全沙箱。

<!-- segment-manual:start id="p004-s001" -->
##### 个人批注（本片段，插件保留）

<!-- 在这里自由记录；同步此片段时不会改写。 -->
<!-- segment-manual:end -->
<!-- segment:end -->

### 术语表

| 术语 | 原文 | 简明解释 | 首次定位 |
|---|---|---|---|
| 在线探索 | online exploration | 机器人在环境中执行、观察结果并修改程序的试错过程 | §1 |
| 机器人代理 | robot agent | 通过工具接口观察、决策并写控制程序的通用 coding agent | §1 |
| 前端/后端分离 | frontend–backend separation | agent 只通过工具调用，后端持有驱动、低层控制和安全边界 | §4.1 |
| 验证技能 | verified skill | 已无人值守成功并带前置条件、检查点和证据的程序 | §3.4 |
| 持续学习 | continual learning | 复用和局部修改历史程序来适应新任务/场景 | §3.4、§5.5 |

### 复现清单

- [x] 数据与预处理：各 benchmark 的探索场景、目标图像/示范和未见测试场景说明
- [x] 模型与目标函数：固定 coding agent + 工具调用 + 程序编译，没有 policy loss
- [x] 超参数与训练细节：探索预算从 2–128 episode 不等，依任务和 benchmark 变化
- [x] 指标与基线：成功率、探索预算 scaling、agent/程序推理时间；VLA/WAM/经典策略基线
- [ ] 计算资源与成本：给出部分真实执行时间，完整成本待核验
- [ ] 代码、随机种子与版本：本地未检查项目页或仓库

### 开放问题

- 如何把程序级经验转移到不同机器人而不把错误的坐标/动力学假设一起转移？
- 探索 episode 的边际收益何时饱和，如何用风险敏感预算分配替代固定预算？
- verified skill 的证据如何达到可审计、可追责和可重复的标准？

<!-- paper-manual:start id="p004" -->
### 个人批注（插件保留此区域）

<!-- 在这里自由记录；同步时不会改写。 -->
<!-- paper-manual:end -->
<!-- paper:end -->
