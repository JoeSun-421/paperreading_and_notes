# LoopWAM：通过部署 rollout 实现闭环通用策略改进

Wenxuan Guo¹˒²˒³˒*, Yingping Liang²˒*, Zhaoguo Wang¹˒*, Ziyuan Li¹, Angyuan Ma¹, Zhenyu Wu³, Chengkun Wang³, Bingyao Yu¹, Ze Chen², Borui Zhang¹˒³, Xiuwei Xu³˒†, Meng Zhang²˒†, Erjin Zhou², Jiwen Lu¹˒†

¹ 清华大学　² Dexmal　³ Loopeva  
* 核心贡献者　† 通讯作者  
arXiv:2610.18873v2 [cs.RO]，2026 年 10 月 15 日；论文日期：2026 年 10 月 16 日

## 摘要

仅扩展专家示范并不能让通用机器人策略在部署后继续改进。真实世界 rollout 包含成功、次优行为和失败：模仿每个记录下来的动作可能强化不良行为，而过滤掉失败则会丢弃有关物理交互及其后果的证据。我们提出 LoopWAM，一种闭环世界－动作模型，它直接从完整的质量范围中学习，并使用来自自身及其他已部署策略的 rollout。依赖数据来源的噪声采样将行为学习与动作条件世界建模统一起来；质量引导和质量加权监督则在保留不同质量数据的完整视频监督的同时，调节动作模仿。视频 token 与动作 token 在并行去噪过程中持续交互，使模型能够直接利用学到的后果生成动作。在 RoboChallenge Table30-V2 上，三轮部署与训练将成功率从 23.67% 提高到 43.33%，提升 19.7 个百分点，最终得分为 56.80。LoopWAM 在 RoboTwin 2.0 上的平均成功率也达到 94.5%，在 RoboCasa365 上的总体成功率达到 49.5%，且没有额外的具身预训练；在三个基准中，它取得了所比较方法里的最佳总体结果。对于具有挑战性的叠衣任务，在单任务后训练后进行两轮部署驱动的改进，成功率从 13.20% 提高到 55.56%。这些结果表明，在多任务学习和单任务后训练两种设置下，都可以利用混合质量经验实现闭环改进。

[项目主页](https://loop-wam.github.io/)

## 1. 引言

扩展人工采集的机器人数据扩大了通用操作策略的能力（O’Neill et al., 2024），但仅仅扩大示范规模并不能让部署与持续改进形成闭环。部署会暴露精选示范之外的状态、交互和失败模式，为具身数据飞轮提供新经验。真实世界强化学习（RL）（Xu et al., 2026）可以在部署期间提升策略表现，但可能会缩窄模型的预测分布，而且往往难以学习多任务通用策略。尽管从多个 RL 专家策略中蒸馏出一个通用策略（Xiao et al., 2025）可以应对这一问题，但通用策略的监督仍来自经过筛选的成功 rollout，而非完整的交互历史。因此，我们希望研究如何直接利用成功、次优和失败部署 rollout 的混合数据来改进通用策略。挑战在于如何有效利用这些经验：模仿所有记录下来的动作有吸收不良行为的风险，而只保留高质量轨迹则会丢弃有信息量的交互及其结果。

近期的视觉－语言－动作模型（VLA）和世界－动作模型分别解决了这一问题中的重要部分。在 VLA 方面，π₀.₆（Physical Intelligence, 2025）使用优势条件学习来纳入自主经验和人工干预；π₀.₇（Physical Intelligence, 2026）则利用质量、速度和错误元数据吸收混合质量数据，并将专家策略的能力蒸馏到通用策略中。这些 VLA 方法在训练时使用引导来建模不同数据质量下的行为分布，并在推理时指定期望的行为模式以实现平稳执行，但它们依赖基于视觉－语言模型（VLM）的语言－视觉－动作表征，没有显式目标来学习动作－视频转移所反映的物理动力学。因此，次优和失败 rollout 提供了行为目标，却不能直接监督其物理后果的学习。

[图 1 原图（PDF 第 2 页）](loopwam.pdf#page=2)

**图 1：** LoopWAM 从部署中学习。混合质量 rollout 通过质量引导、依赖数据来源的噪声采样和质量加权监督回流至训练。中间部分展示在 RoboChallenge Table30-V2 上的闭环自我改进：三轮部署和再训练将总体成功率从 23.67% 提高到 43.33%。底部为基准对比。

图中标签：部署、收集、改进、多样化真实任务、成功／次优／失败、专家、rollout、质量感知学习、更广的状态－动作覆盖；基准栏比较 RoboTwin 2.0、RoboChallenge Table30-V2 和 RoboCasa365。

世界建模提供了更合适的解决方案：世界－动作模型（WAM）（Li et al., 2026；Pai et al., 2025；Ye et al., 2026b）将预测性视觉表征与动作生成结合起来；动作条件世界模型（ACWM）（Gao et al., 2026；Guo et al., 2025）则根据动作预测未来观测。将二者结合，可以让 ACWM 从不同质量的数据中学习动作后果，同时让 WAM 从专家示范中学习恰当的动作。τ₀-WM（Zhou et al., 2026）分别训练 WAM 和 ACWM，将失败与恢复数据送入 ACWM 训练，并使用 ACWM 作为策略改进的筛选器。FACT（Peng et al., 2026）在共享模型中实现了这两种作用，但仍按顺序生成动作及其后果。它只支持成功／失败两类监督划分，失败后果通过共享表征隐式地影响策略。这些方法揭示了混合质量部署经验学习中仍待解决的设计挑战：VLA 引导缺少自然的动作－视频依赖建模；能够感知后果的 WAM 仍通过 ACWM 接口间接利用 rollout 数据——把 ACWM 作为动作选择模拟器，或者后续的后果预测阶段，再由共享表征影响策略。能否让一个 WAM 直接从混合质量部署 rollout 中学习，让所有质量的数据都监督模型，并让学到的后果直接参与动作生成？

我们提出 LoopWAM，这是一种闭环世界－动作模型，可以利用混合质量部署 rollout 直接更新通用策略。我们的核心想法是让一个 WAM 直接吸收完整质量范围的 rollout，并从每条数据中学习。为此，我们首先使用依赖数据来源的噪声采样，在单个模型和单个生成阶段内统一 WAM 行为学习与 ACWM 后果学习。视频和动作的噪声水平独立采样，使低噪声的 rollout 动作能够引导模型预测其观测到的后果。在这个统一模型中，质量引导对不同质量的行为进行表征，并在部署时选择最高质量的条件。为了控制对各类行为的模仿强度，我们进一步使用质量加权监督，降低对次优动作的模仿，并屏蔽失败动作目标，同时保留所有质量数据的视频监督。由此，每条轨迹都能根据其行为质量和观测到的转移来监督联合模型。视频 token 和动作 token 在一次并行去噪过程中双向交互，使学到的未来表征能够参与动作形成，无需单独的后果评估阶段。我们通过反复收集 rollout、在示范和累积经验上重新训练通用策略，再重新部署，从而形成闭环。在 RoboChallenge Table30-V2（RoboChallenge Dexmal Team, 2026）上，三轮训练将总体成功率从 23.67% 提高到 28.00%、37.67% 和 43.33%，提升 19.7 个百分点，最终得分为 56.80，在所比较方法中成功率和得分均排名第一（表 1）。LoopWAM 在 RoboTwin 2.0（Chen et al., 2025）上取得 94.5% 的平均成功率，超过报告中的 FACT（87.5%）和 LingBot-VA 2.0（93.6%）；在包含 50 项任务的 RoboCasa365 评测（Nasiriany et al., 2026）上达到 49.5%，高于 ABot-M0.6 的 46.6%（表 2–3）。我们还评估了单任务后训练后的部署驱动自我改进，任务是涉及可变形物体和多个协同步骤的高难度叠衣。两轮部署与训练将成功率从 13.20% 提高到 55.56%（第 4.4 节）。这些结果表明，通过并行世界－动作去噪直接学习混合质量 rollout，可以在多任务和单任务设置下取得领先的基准表现，并在反复部署中持续改进。

## 2. 相关工作

**机器人操作策略。** 机器人操作策略越来越多地将可扩展的模仿学习与预训练表征结合起来。ACT（Zhao et al., 2023）和 Diffusion Policy（Chi et al., 2025）从示范中学习动作序列；RT-2（Zitkovich et al., 2023）和 OpenVLA（Kim et al., 2024）则将视觉－语言先验迁移到语言条件控制中。π₀（Black et al., 2024）和 π₀.₅（Physical Intelligence et al., 2025）等基于流的策略进一步扩展了连续动作生成与泛化能力。Open X-Embodiment（O’Neill et al., 2024）和 DROID（Khazatsky et al., 2024）扩大了跨本体和环境的训练覆盖范围。除离线示范之外，DAgger（Ross et al., 2011）通过在策略访问到的状态上收集专家标注来应对策略诱发的分布偏移。LoopWAM 研究如何从部署 rollout 记录的观测和动作中学习：利用轨迹质量引导策略更新，同时保留来自未成功交互的世界监督。

**世界－动作模型。** 基于视频的机器人策略使用预测性视觉表征来支持控制。mimic-video（Pai et al., 2025）根据预训练视频模型的表征来调节动作解码器。DreamZero（Ye et al., 2026b）联合生成视频和动作；LingBot-VA（Li et al., 2026）则在因果自回归框架内组织视觉预测和动作解码。LingBot-VA 2.0（Zhang et al., 2026a）进一步使用面向控制的 tokenizer 开展原生视频－动作预训练。提升效率的设计包括 Fast-WAM（Yuan et al., 2026），它保留视频联合训练，但在推理时移除未来视频生成；以及 GigaWorld-Policy（Ye et al., 2026a），它学习动作条件视觉动力学，同时允许仅使用动作进行推理。Being-H0.7（Luo et al., 2026）则使用未来观测监督潜在表征。这些研究展示了利用预测性监督支持动作学习的多种方式。LoopWAM 的重点是闭合部署－训练循环：混合质量 rollout 扩大世界模型训练范围，而联合视频－动作建模、质量引导和加权动作监督则将后果感知与高质量行为联系起来。

## 3. 方法

LoopWAM 学习一种世界－动作模型，可以利用部署期间收集的经验进行更新。该框架在共享去噪过程中联合视频和动作生成，再利用轨迹质量区分部署数据如何监督世界建模和控制。图 2 总结了模型架构与训练流程。我们先介绍联合世界－动作建模，再介绍从混合轨迹中开展质量感知学习，以及部署－学习循环。

[图 2 原图（PDF 第 4 页）](loopwam.pdf#page=4)

**图 2：** LoopWAM 概览。LoopWAM 采用 Transformer 混合（Mixture-of-Transformer，MoT）架构，双向注意力使视频分支和动作分支能够进行特征交互。训练结合了质量条件指令、依赖数据来源的噪声采样和质量加权动作监督。专家轨迹同时监督两个模态；质量较低的 rollout 保留完整的世界监督，并根据其质量参与动作模仿。部署时，训练好的模型使用专家分数 $q^\star$；记录到的经验则返回到下一轮训练。

图中标签：观测、带噪未来视频、带噪动作、混合质量数据、机器人状态 token、文本 token、交叉注意力、视频分支、动作分支、联合注意力（视频 ↔ 动作）、未来视频、动作块、质量引导、训练时使用质量条件文本、推理时采用高质量行为；下方三个部分分别表示噪声采样、质量加权监督和 rollout 标注。图中还区分专家与 rollout、世界与动作监督、逐回合与批次级标注。

### 3.1. 联合世界－动作建模

我们考虑语言条件操作任务，其输入包括视觉观测 $o_t$、机器人状态 $s_t$ 和指令 $\ell$。预测目标是一个动作块 $A_t=(a_t,\ldots,a_{t+H-1})$ 及相应的未来观测 $V_t$，二者在相同时间跨度内采样。预训练视频 VAE 将 $(o_t,V_t)$ 映射为观测潜变量 $z_o$ 和未来潜变量 $Z_t$。LoopWAM 对它们的联合分布建模：

$$
p_\theta(A_t,Z_t\mid o_t,s_t,\ell,q).
$$

<div align="right">(1)</div>

其中，$q$ 是轨迹质量的数值分数，详见第 3.2 节。这一统一表述使单个模型能够从成对的动作－观测序列中学习动作生成和视觉动力学（Zhu et al., 2025）。

我们采用双流扩散 Transformer 来实现这一表述，其视频骨干网络由 Wan2.2-TI2V-5B（Wan et al., 2025）初始化。视频流和动作流各自具有独立的 token 投影、Transformer 参数和预测头，每一层都通过联合注意力连接两个流。为了用流匹配（Lipman et al., 2023）训练，我们对每个目标 $X_m$ 独立加扰，其中 $X_v=Z_t$，$X_a=A_t$：

$$
X_m^{\sigma_m}=(1-\sigma_m)X_m+\sigma_m\epsilon_m,
\qquad \epsilon_m\sim\mathcal{N}(0,I),
\qquad m\in\{v,a\}.
$$

<div align="right">(2)</div>

联合网络预测速度场 $(v_{\theta_v},v_{\theta_a})$，其目标为 $u_m=\epsilon_m-X_m$。每个流接收自己的噪声水平 $\sigma_m$，其中 0 表示干净数据，1 表示纯噪声；观测潜变量 $z_o$ 始终保持干净。

注意力结构使两个预测在整个去噪过程中相互耦合。分别用 $O$、$V$ 和 $A$ 表示观测、未来视频和动作 token。其块掩码中，查询对应行，键对应列：

$$
M(O,V,A)=
\begin{pmatrix}
1&0&0\\
1&1&1\\
1&1&1
\end{pmatrix}.
$$

<div align="right">(3)</div>

其中 1 表示允许注意。观测 token 只在初始帧内部进行注意。两个流都对质量条件指令进行交叉注意；机器人状态 token 还额外为动作流提供条件。结合第 3.2 节的训练方案，这种架构有助于学习具身世界表征以及对动作后果的感知。对专家和混合质量 rollout 进行密集视频监督，扩大了模型学习这些表征时接触到的交互范围。通过双向注意力，世界预测损失也会训练用于动作生成的特征。这种训练将“如何完成任务”与“特定动作如何影响结果”的学习结合起来，鼓励视频生成与动作生成保持一致。

### 3.2. 从混合轨迹中进行质量感知学习

我们的训练语料由专家示范 $\mathcal{D}_E$ 和任意质量的 rollout 数据 $\mathcal{D}_R$ 组成，其中包括成功、次优和失败的执行。这些 rollout 大幅扩展了专家路径之外的状态－动作－结果分布；即使其中的动作不适合作为模仿目标，它们仍然能够提供世界模型监督。每条轨迹都分配一个数据来源指示符 $d\in\{E,R\}$；从该轨迹提取的每个训练片段都使用相同的指示符。我们结合依赖数据来源的噪声采样、质量条件化和选择性动作监督，从更广泛的经验中学习，同时优先学习高质量行为。

**依赖数据来源的噪声采样。** 在 WAM 中联合预测视频和动作，本身并不能保证预测的未来反映生成动作所带来的后果。为了提高视频生成与动作生成的一致性，使世界预测对控制更有用，我们希望在同一统一模型中加强动作条件世界建模（ACWM）。因此，我们使用依赖数据来源的噪声采样，使预测观测到的后果时能够利用已记录的 rollout 动作，同时仍在单个训练阶段中联合生成视频和动作。具体来说，我们为专家片段和 rollout 片段选择不同的噪声分布，并在每个片段内独立采样视频和动作的噪声水平：

$$
\begin{aligned}
d=E &: \quad \sigma_v\sim\mathrm{Beta}(7,1),\quad \sigma_a\sim\mathcal{P}_0,\\
d=R &: \quad \sigma_v\sim\mathcal{P}_0,\quad \sigma_a\sim\mathrm{Beta}(1,7).
\end{aligned}
$$

<div align="right">(4)</div>

这里 $\mathcal{P}_0$ 是基础噪声分布：采样 $u\sim\mathcal{U}(0,1)$，并令 $\sigma=\varphi_\kappa(u)=\kappa u/[1+(\kappa-1)u]$，其中 $\kappa=5$。Beta 分布（Johnson et al., 1995）直接作用于 $\sigma$。对于专家数据，较高的视频噪声使模型着重从强噪声输入中恢复专家未来，同时在基础噪声分布上生成动作。对于 rollout，较低的动作噪声保留了大部分已记录动作，而视频流预测其视觉后果。这些执行轨迹扩大了世界模型看到的动作－结果分布，其中包括任务尚未完成或交互失败的转移。之后，联合注意力会将这一后果学习信号连接到用于动作生成的表征。

**质量引导。** 在这一统一训练方案中，每条轨迹都分配一个数值质量分数 $q$，从该轨迹提取的所有训练片段共享该分数。我们将质量编码为指令的后缀 $\ell_q=[\ell;\text{Quality: }q]$，并使用其文本嵌入为视频流和动作流提供条件。我们支持两种轨迹标注方案：(1) 逐回合标注，由人工标注者或视觉－语言模型（VLM）逐条评估轨迹，以细粒度地估计质量；(2) 粗粒度标注，根据某个策略检查点的总体成功率和动作平滑性，由人工为该检查点的所有 rollout 统一赋予一个分数。在粗粒度标注中，成功轨迹使用检查点分数，失败轨迹使用最低分数，因此每个检查点只需要一次标注。两种方案都使用数值分数表示 rollout 质量。对于专家示范，我们赋予专家分数 $q^\star$，使其高于 rollout 质量分数的最大值。质量条件指令使同一模型能够表示同一任务指令下的不同行为。部署期间，我们令 $q=q^\star$，引导采样朝专家质量的行为移动。

**质量加权监督。** 给定这些噪声水平和质量分数，我们保留对所有轨迹质量的未来视频预测，并使用非递减权重 $\lambda(q)\in[0,1]$ 调节动作监督。专家轨迹获得完整动作监督，$\lambda(q^\star)=1$；失败轨迹使用 $\lambda(q)=0$，不接受动作监督。对于样本 $i$ 和模态 $m$，定义去噪损失：

$$
\ell_{m,i}=\omega(\sigma_{m,i})\left\|v_{\theta,i}^{m}-(\epsilon_{m,i}-X_{m,i})\right\|_{\mathrm{valid}}^2,
$$

<div align="right">(5)</div>

其中，范数表示在有效目标元素上计算的均方误差，$\omega$ 是通用的时间步加权函数。干净的观测潜变量不参与视频监督；两个模态中的填充目标都会被掩蔽。对于含 $B$ 个片段的小批次，完整目标函数为：

$$
\mathcal{L}_{\mathrm{LoopWAM}}
=\frac{1}{B}\sum_{i=1}^{B}\ell_{v,i}
+\frac{\sum_{i=1}^{B}\lambda(q_i)\ell_{a,i}}
{\max\left(1,\sum_{i=1}^{B}\lambda(q_i)\right)}.
$$

<div align="right">(6)</div>

其中第一项表示世界监督，第二项表示质量加权动作监督。

数据来源指示符决定加噪方式，而质量决定行为条件和动作权重。具体而言，$\lambda(q)=0$ 会移除对某条轨迹动作的直接模仿，同时保留由视频和动作表征联合注意力产生的世界模型梯度。因此，质量引导和加权监督既能扩大转移学习范围，也能将动作模仿集中在更理想的行为上。数据预处理细节见附录 A。

### 3.3. 闭环自我改进

部署策略使用联合模型中高质量动作的边缘分布：

$$
\pi_\theta(A_t\mid o_t,s_t,\ell;q^\star)
=\int p_\theta(A_t,Z_t\mid o_t,s_t,\ell,q^\star)\,dZ_t.
$$

<div align="right">(7)</div>

实际操作中，我们从高斯噪声出发联合去噪视频和动作来采样这个边缘分布，同时固定当前观测潜变量。从经专家训练的模型开始，我们收集并评估部署 rollout，然后使用式 (6) 在专家示范和所有累积 rollout 上更新模型。更新后的模型使用专家分数 $q^\star$ 重新部署，以继续部署－学习循环。

更一般地说，LoopWAM 的自我改进更新可以使用任意已部署策略产生的 rollout，即使这些策略表现不佳也可以。联合架构和质量感知目标能够处理不同的轨迹质量，使每条轨迹都能根据其观测到的转移和行为质量作出贡献。这使自我改进能够利用其他策略以及 LoopWAM 自身收集的部署经验。我们在 RoboTwin 2.0 和 RoboCasa365 上验证了这一更广泛的设置（第 4.3 节）。

## 4. 实验

### 4.1. 实现细节

我们使用 Wan2.2-TI2V-5B（Wan et al., 2025）初始化视频骨干网络、文本编码器和视频 VAE，并采用隐藏维度为 1024 的动作 DiT。每个训练片段包含一个 64 步动作块，以及按每 8 个控制步采样的一组 9 帧 RGB 图像，其中包括初始观测。多摄像头视角拼接成单张图像；动作和机器人状态使用专家数据统计量归一化。训练遵循第 3.2 节的质量条件联合目标，部署时使用专家分数 $q^\star$。Table30-V2 使用相对于起始状态的关节目标和绝对夹爪命令。更多训练、预处理和子任务分割细节见附录 A。

### 4.2. 基准实验设置

我们在真实机器人基准 RoboChallenge Table30-V2 和两个仿真基准 RoboTwin 2.0、RoboCasa365 上评估 LoopWAM。RoboChallenge 使用 LoopWAM 自身的真实机器人部署来评估闭环自我改进。两个仿真基准还通过在专家示范中加入由专家训练的 LoopWAM 及其他策略产生的 rollout，测试从不同已部署策略中学习的能力。在这些设置中，成功、次优和失败 rollout 都将状态－动作－结果分布扩展到专家示范之外。

**RoboChallenge Table30-V2。** RoboChallenge Table30-V2（RoboChallenge Dexmal Team, 2026）是标准化真实机器人基准，包含 30 项桌面操作任务，覆盖四种机器人本体：ALOHA、DOS-W1、ARX5 和 UR5。任务测试工具使用、精细操作和双臂协同控制等能力。我们遵循官方多任务评测协议，并报告各本体上的总体表现。从每种本体各自一个经专家训练的多任务 LoopWAM 模型开始，我们进行三轮连续的闭环自我改进。每轮部署时，我们让每个模型执行其分配到的任务，并为每项任务收集约 100 条 rollout 轨迹。这些轨迹经过质量标注后，与专家示范以及此前所有部署轮次收集的 rollout 一起用于后续训练。

**RoboTwin 2.0。** RoboTwin 2.0（Chen et al., 2025）包含 50 项双臂操作任务，在干净场景和域随机化场景中进行评测。按照 LingBot-VA（Li et al., 2026）的多任务设置，我们使用来自干净场景的 2,500 条专家示范，以及来自高度随机化场景的 25,000 条示范。我们再加入由专家训练的 LoopWAM、X-VLA（Zheng et al., 2025）和 π₀.₅（Physical Intelligence et al., 2025）生成的 rollout。对于每项任务，每种策略产生 20 条干净场景 rollout 和 130 条随机化场景 rollout；三种策略在全部 50 项任务上共生成 22,500 条轨迹。rollout 语料的总时长约等于专家语料。rollout 收集和评测使用不同随机种子，以避免数据泄漏。rollout 质量使用第 3.2 节的粗粒度数值分数标注。

**RoboCasa365。** RoboCasa365（Nasiriany et al., 2026）是一个大规模仿真框架，用于在多样化厨房环境中训练家务操作。我们在其 50 项任务基准上评测，该基准包含原子任务和复合操作任务。训练时将专家示范与专家训练的 LoopWAM 和 RLDX-1（Kim et al., 2026）生成的 rollout 结合，合并后 rollout 的总时长约等于专家数据。专家数据和 rollout 数据都遵循 Human300 训练划分：16 项 Composite-Unseen 任务不参与训练，留作零样本评测。与 RoboTwin 2.0 一样，训练数据和评测数据使用不同的随机种子。

### 4.3. 基准结果

**RoboChallenge Table30-V2。** 表 1 将专家训练后的 LoopWAM 和最终 LoopWAM 与官方排行榜（RoboChallenge, 2026）中的领先结果进行比较。经过三轮部署与训练后，LoopWAM 的总体得分为 56.80，成功率为 43.33%；它比 VLA-DM0.5 高 2.38 分，成功率高 2.67 个百分点，在所比较方法中两项指标都排名第一。附录 B 收录了排行榜快照。第 4.4 节分析了相对于只进行专家训练，连续部署更新如何带来这一提升。

**表 1：** RoboChallenge Table30-V2 上的表现。我们报告四种机器人本体各自的成功率（SR，%）和总体表现。基线按总体 SR 排序。粗体表示最佳表现；蓝色和浅蓝色分别标出 LoopWAM 和排名第二的方法。

| 方法 | ALOHA 得分 | ALOHA SR | DOS-W1 得分 | DOS-W1 SR | ARX5 得分 | ARX5 SR | UR5 得分 | UR5 SR | 总体得分 | 总体 SR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| debuggers_vla | 5.40 | 0.00 | 5.15 | 2.00 | 53.21 | 34.29 | 27.83 | 23.33 | 18.72 | 11.00 |
| W0 | 12.25 | 2.00 | 4.40 | 0.00 | 48.64 | 30.00 | 49.50 | 43.33 | 21.85 | 12.00 |
| test_01 | 6.25 | 1.00 | 8.95 | 3.00 | 49.50 | 28.57 | 61.17 | 43.33 | 22.73 | 12.33 |
| holobrain | 11.75 | 3.00 | 13.30 | 6.00 | 53.64 | 30.00 | 70.67 | 43.33 | 27.93 | 14.33 |
| Baseline Model | 21.55 | 6.00 | 17.35 | 6.00 | 60.14 | 32.86 | 44.83 | 26.67 | 31.48 | 14.33 |
| MagicBot | 4.70 | 0.00 | 6.85 | 5.00 | 47.57 | 32.86 | 58.67 | 53.33 | 20.82 | 14.67 |
| mc | 10.60 | 7.00 | 2.00 | 0.00 | 73.64 | 47.14 | 22.50 | 20.00 | 23.63 | 15.33 |
| JV0 | 12.70 | 5.00 | 17.80 | 16.00 | 49.64 | 20.00 | 58.33 | 43.33 | 27.58 | 16.00 |
| my16 | 18.95 | 13.00 | 36.60 | 27.00 | 73.29 | 54.29 | 56.50 | 46.67 | 41.27 | 30.67 |
| VLA-DM0.5 | 26.50 | 18.00 | 55.45 | 42.00 | 79.07 | 58.57 | 86.50 | 70.00 | 54.42 | 40.67 |
| LoopWAM（专家训练） | 23.05 | 8.00 | 46.35 | 38.00 | 37.00 | 24.29 | 43.33 | 26.67 | 36.10 | 23.67 |
| LoopWAM | 38.65 | 26.00 | 54.90 | 42.00 | 76.21 | 55.71 | 78.33 | 76.67 | 56.80 | 43.33 |

[表 1 原图（PDF 第 8 页）](loopwam.pdf#page=8)

**RoboTwin 2.0。** LoopWAM 在干净场景中的成功率为 94.58%，在域随机化场景中为 94.42%，平均为 94.50%（表 2）。没有额外具身预训练时，它比 AHA-WAM 高 1.70 个百分点，平均表现也超过经过预训练的 LingBot-VA 2.0 0.90 个百分点。总体而言，这些结果表明，联合利用专家示范和多样化部署 rollout 进行学习，可以在两种评测设置下得到有竞争力的多任务策略。

**表 2：** RoboTwin 2.0 上的表现。我们报告干净和随机化评测设置下的成功率（SR，%）。Embodied PT. 表示在基准训练之前进行额外的具身数据预训练。LoopWAM 使用第 4.2 节所述的专家－rollout 混合语料训练。蓝色和浅蓝色分别标出 LoopWAM 和排名第二的方法。

| 方法 | Embodied PT. | 干净场景 | 随机场景 | 平均值 |
|---|:---:|---:|---:|---:|
| π₀（Black et al., 2024） | ✓ | 65.9 | 58.4 | 62.2 |
| X-VLA（Zheng et al., 2025） | ✓ | 72.9 | 72.8 | 72.9 |
| π₀.₅（Physical Intelligence et al., 2025） | ✓ | 82.7 | 76.8 | 79.8 |
| ABot-M0（Yang et al., 2026） | ✓ | 81.2 | 80.4 | 80.8 |
| Motus（来自 Wan2.2）（Bi et al., 2025） | ✗ | 77.6 | 77.0 | 77.3 |
| Motus（Bi et al., 2025） | ✓ | 88.7 | 87.0 | 87.8 |
| LingBot-VA（Li et al., 2026） | ✓ | 92.9 | 91.5 | 92.2 |
| Fast-WAM（Yuan et al., 2026） | ✗ | 91.9 | 91.8 | 91.8 |
| AHA-WAM（Cai et al., 2026） | ✗ | 93.4 | 92.2 | 92.8 |
| LingBot-VA 2.0（Zhang et al., 2026a） | ✓ | 93.8 | 93.4 | 93.6 |
| LoopWAM | ✗ | 94.6 | 94.4 | 94.5 |

**RoboCasa365。** LoopWAM 在 Atomic-Seen、Composite-Seen 和 Composite-Unseen 上的成功率分别为 79.7%、50.3% 和 14.9%；在 50 项任务上的总体成功率为 49.5%（表 3）。没有额外具身预训练时，它的总体表现比 ABot-M0.6 高 2.9 个百分点。该提升也扩展到了留出的复合任务，表明模型能够迁移到专家和 rollout 训练数据中都没有出现过的任务组合。

**表 3：** RoboCasa365 上的表现。我们报告三个评测划分上的成功率（SR，%）。基线结果来自官方排行榜。Embodied PT. 的定义见表 2。蓝色和浅蓝色分别标出 LoopWAM 和排名第二的方法。

| 方法 | Embodied PT. | Atomic Seen | Composite Seen | Composite Unseen | 总体 |
|---|:---:|---:|---:|---:|---:|
| Diffusion Policy（Chi et al., 2025） | ✗ | 15.7 | 0.2 | 1.3 | 6.1 |
| π₀（Black et al., 2024） | ✓ | 34.6 | 6.1 | 1.1 | 14.8 |
| π₀.₅（Physical Intelligence et al., 2025） | ✓ | 39.6 | 7.1 | 1.2 | 16.9 |
| GigaWorld-Policy 0.1（Ye et al., 2026a） | ✓ | 44.4 | 11.8 | 2.9 | 20.7 |
| GR00T N1.6（NVIDIA et al., 2025） | ✓ | 51.1 | 9.4 | 1.7 | 21.9 |
| GR00T N1.5（NVIDIA et al., 2025） | ✓ | 50.7 | 14.8 | 2.7 | 23.9 |
| RLDX-1（Kim et al., 2026） | ✓ | 67.6 | 27.9 | 8.5 | 36.0 |
| PRTS（Zhang et al., 2026b） | ✓ | 66.3 | 30.3 | 18.8 | 39.6 |
| ABot-M0.5（Chen et al., 2026） | ✓ | 75.6 | 37.7 | 3.3 | 40.3 |
| ABot-M0.6（Yang et al., 2026） | ✓ | 79.4 | 48.3 | 7.9 | 46.6 |
| LoopWAM | ✗ | 79.7 | 50.3 | 14.9 | 49.5 |

[表 2、表 3 原图（PDF 第 9 页）](loopwam.pdf#page=9)

### 4.4. 通过部署实现闭环改进

我们在两种真实机器人设置下考察部署驱动的改进：多任务学习和单任务后训练（图 3）。两者都遵循第 3.3 节的更新流程：每轮迭代中，我们部署当前策略并为收集到的 rollout 赋予质量分数，然后在专家示范和迄今收集到的所有 rollout 上进行质量感知联合训练，以更新模型。

**多任务部署。** 如第 4.2 节所述，我们为 ALOHA、DOS-W1、ARX5 和 UR5 训练多任务模型，覆盖 Table30-V2 的 30 项任务。从专家训练的检查点开始，每个模型都利用所分配任务上的经验进行三轮部署和训练。总体成功率从 23.67% 上升至 28.00%、37.67% 和 43.33%，得分从 36.10 上升至 42.63、51.65 和 56.80。两项汇总指标在每轮迭代中都持续提高，相比初始部署，成功率提升了 19.7 个百分点。这表明 LoopWAM 可以利用自身的部署经验来改进在多个任务上训练的策略。

**单任务后训练。** 我们还评估了叠衣任务，这是一项具有挑战性的可变形物体操作任务，初始状态多样且杂乱。衬衫会被随意抛到机器人的工作空间，形成多种凌乱的初始状态，其中衣角可能被遮住。机器人必须找到合适的抓取点，逐步展开衬衫，直到衣角露出，再抓住这些衣角完成后续折叠。我们从在 Table30-V2 上训练的 DOS-W1 多任务模型开始，使用新收集的叠衣专家示范进行后训练。得到的任务专用专家训练检查点作为两轮部署和自我改进的起点。成功率从 13.20% 提升到 32.54% 和 55.56%，得分从 14.44 提升到 38.89 和 67.78。这些提升表明，在将多任务模型适配到具有挑战性的单项任务后，部署反馈仍然有效；无论采用哪种训练模式，都可以开展闭环改进。

[图 3 原图（PDF 第 10 页）](loopwam.pdf#page=10)

**图 3：** 来自真实机器人部署的闭环改进。上：在 RoboChallenge Table30-V2 上进行三轮部署与训练，覆盖四种机器人本体的 30 项任务。下：DOS-W1 模型经过任务专用专家后训练后，在叠衣任务上进行两轮部署与训练。每幅图都报告连续改进轮次中的成功率和得分。

### 4.5. 消融研究

我们采用第 4.2 节的 RoboTwin 2.0 设置，只改变质量引导和 rollout 动作监督；专家动作监督和视频监督保持不变。

表 4 比较无引导、额外粗粒度标注和本文的粗粒度标注方案（第 3.2 节）。额外粗粒度标注仅根据成功或失败来区分所有策略的 rollout：所有失败 rollout 都标为 Quality: low，所有成功 rollout 都标为 Quality: medium，专家示范标为 Quality: high。在本文的粗粒度方案中，同一策略产生的成功 rollout 共用一个数值分数；该分数可以因策略而异，取决于其跨任务总体成功率以及人工对动作平滑性的评估。失败 rollout 被赋予最低分，专家示范被赋予高于 rollout 分数范围的 $q^\star$。消融结果揭示了三个主要发现：(1) 质量引导是从混合质量部署 rollout 中学习的关键，使模型能够区分不同质量的轨迹。(2) 即使是额外粗粒度标签也有效，说明受益于引导并不需要细粒度标注。(3) 质量感知的 rollout 动作监督在两种标注方案下都能进一步提升总体表现。

**表 4：** RoboTwin 2.0 上的消融研究。我们使用不同的标注方案评估质量引导，并考察质量加权 rollout 动作监督（Act.）的效果。

| 质量引导 | 质量加权动作监督 | 干净场景 | 随机场景 | 平均值 |
|---|:---:|---:|---:|---:|
| 无 | ✗ | 73.5 | 71.1 | 72.3 |
| 额外粗粒度 | ✗ | 92.2 | 93.3 | 92.8 |
| 额外粗粒度 | ✓ | 93.9 | 93.3 | 93.6 |
| 粗粒度 | ✗ | 93.2 | 92.2 | 92.7 |
| 粗粒度（本文） | ✓ | 94.6 | 94.4 | 94.5 |

[表 4 原图（PDF 第 10 页）](loopwam.pdf#page=10)

## 5. 结论

我们提出 LoopWAM，直接利用部署经验改进通用世界－动作策略。关键在于从不同质量的轨迹中学习物理后果，同时控制这些轨迹的动作以何种方式监督行为。依赖数据来源的噪声采样、质量引导和质量加权监督，在联合视频－动作去噪过程中实现了这种区分，使后果表征能够直接参与动作生成。相同的表述既能利用 LoopWAM 自身的 rollout，也能利用其他已部署策略的 rollout。实验取得了最佳总体结果：在 RoboChallenge Table30-V2 上得分为 56.80、成功率为 43.33%，在 RoboTwin 2.0 上的平均成功率为 94.5%，在 RoboCasa365 上的总体成功率为 49.5%。真实机器人上的反复部署经过三轮将 Table30-V2 成功率提高了 19.7 个百分点；单任务后训练后，经过两轮叠衣部署，成功率从 13.20% 提高到 55.56%。这些结果支持将混合质量交互历史作为训练资源，以持续改进多任务策略和适配到单项任务的策略。

## 参考文献

Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A Unified Latent Action World Model. arXiv preprint arXiv:2512.13030, 2025. （原文页码标记：9）

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv preprint arXiv:2410.24164, 2024. （原文页码标记：3, 9）

Jisong Cai, Long Ling, Shiwei Chu, Zhongshan Liu, Jiayue Kang, Zhixuan Liang, Wenjie Xu, Yinan Mao, Weinan Zhang, Xiaokang Yang, Ru Ying, Ran Zheng, and Yao Mu. AHA-WAM: Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Context Routing. arXiv preprint arXiv:2606.09811, 2026. （原文页码标记：9）

Ronghan Chen, Yandan Yang, Zuojin Tang, Dongjie Huo, Tong Lin, Haoning Wu, Haoyun Liu, Yuzhi Chen, Lulu Zheng, Botai Yuan, et al. ABot-M0.5: Unified Mobility-and-Manipulation World Action Model. arXiv preprint arXiv:2607.00678, 2026. （原文页码标记：9）

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation. arXiv preprint arXiv:2506.18088, 2025. （原文页码标记：3, 7）

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. IJRR, 2025. （原文页码标记：3, 9）

Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, Wei-Cheng Tseng, Yuzhu Dong, Kaichun Mo, Chen-Hsuan Lin, et al. Dreamdojo: A generalist robot world model from large-scale human videos. arXiv preprint arXiv:2602.06949, 2026. （原文页码标记：2）

Yanjiang Guo, Lucy Xiaoyang Shi, Jianyu Chen, and Chelsea Finn. Ctrl-world: A controllable generative world model for robot manipulation. arXiv preprint arXiv:2510.10125, 2025. （原文页码标记：2）

Norman L. Johnson, Samuel Kotz, and Narayanaswamy Balakrishnan. Continuous Univariate Distributions, volume 2. John Wiley & Sons, 2nd edition, 1995. ISBN 9780471584940. （原文页码标记：5）

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset. arXiv preprint arXiv:2403.12945, 2024. （原文页码标记：3）

Dongyoung Kim, Huiwon Jang, Myungkyu Koo, Suhyeok Jang, Taeyoung Kim, Beomjun Kim, Byungjun Yoon, Changsung Jang, Daewon Choi, Dongsu Han, et al. RLDX-1 Technical Report. arXiv preprint arXiv:2605.03269, 2026. URL https://arxiv.org/abs/2605.03269. （原文页码标记：7, 9）

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. OpenVLA: An Open-Source Vision-Language-Action Model. arXiv preprint arXiv:2406.09246, 2024. （原文页码标记：3）

Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026. （原文页码标记：2, 3, 7, 9）

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow Matching for Generative Modeling. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.02747. （原文页码标记：4）

Hao Luo, Wanpeng Zhang, Yicheng Feng, Sipeng Zheng, Haiweng Xu, Chaoyi Xu, Ziheng Xi, Yuhui Fu, and Zongqing Lu. Being-H0.7: A Latent World-Action Model from Egocentric Videos. arXiv preprint arXiv:2605.00078, 2026. URL https://arxiv.org/abs/2605.00078. （原文页码标记：3）

Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, and Yuke Zhu. RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots. arXiv preprint arXiv:2603.04356, 2026. （原文页码标记：3, 7）

NVIDIA, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, et al. GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. arXiv preprint arXiv:2503.14734, 2025. （原文页码标记：9）

Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024. （原文页码标记：1, 3）

Jonas Pai, Liam Achenbach, Victoriano Montesinos, Benedek Forrai, Oier Mees, and Elvis Nava. mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs. arXiv preprint arXiv:2512.15692, 2025. （原文页码标记：2, 3）

Quanquan Peng, Yutong Liang, Rui Yan, Nicklas Hansen, and Xiaolong Wang. FACT: Failure-Aware Causal Training for World-Action Models. arXiv preprint arXiv:2608.10232, 2026. （原文页码标记：2）

Physical Intelligence. π0.6: A VLA That Learns From Experience. arXiv preprint arXiv:2511.14759, 2025. （原文页码标记：1）

Physical Intelligence. π0.7: A Steerable Generalist Robotic Foundation Model with Emergent Capabilities. arXiv preprint arXiv:2604.15483, 2026. （原文页码标记：1）

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: A Vision-Language-Action Model with Open-World Generalization. arXiv preprint arXiv:2504.16054, 2025. （原文页码标记：3, 7, 9）

RoboChallenge. RoboChallenge Table30-V2 Leaderboard, 2026. URL https://robochallenge.cn/leaderboard?selectBenchmark=table30_v2&currentPage=1&pageSize=30. Accessed September 22, 2026. （原文页码标记：7）

RoboChallenge Dexmal Team. Table30 V2: Evaluating Generalized Models by Real Robots at Scale. Technical report, Dexmal, 2026. URL https://robochallenge.ai/robochallenge_techreport_V2.pdf. （原文页码标记：3, 7）

Stephane Ross, Geoffrey Gordon, and Drew Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15, pages 627–635. PMLR, 2011. URL https://proceedings.mlr.press/v15/ross11a.html. （原文页码标记：3）

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. （原文页码标记：4, 6）

Wenli Xiao, Haotian Lin, Andy Peng, Haoru Xue, Tairan He, Yuqi Xie, Fengyuan Hu, Jimmy Wu, Zhengyi Luo, Linxi Fan, Guanya Shi, and Yuke Zhu. Self-Improving Vision-Language-Action Models with Data Generation via Residual RL. arXiv preprint arXiv:2511.00091, 2025. （原文页码标记：1）

Charles Xu, Jost Tobias Springenberg, Michael Equi, Ali Amin, Adnan Esmail, Sergey Levine, and Liyiming Ke. RL Token: Bootstrapping Online RL with Vision-Language-Action Models. arXiv preprint arXiv:2604.23073, 2026. URL https://arxiv.org/abs/2604.23073. （原文页码标记：1）

Yandan Yang, Shuang Zeng, Tong Lin, Xinyuan Chang, Dekang Qi, Junjin Xiao, Haoyun Liu, Ronghan Chen, Yuzhi Chen, Dongjie Huo, et al. ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning. arXiv preprint arXiv:2602.11236, 2026. （原文页码标记：9）

Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, et al. GigaWorld-Policy: An Efficient Action-Centered World–Action Model. arXiv preprint arXiv:2603.17240, 2026a. （原文页码标记：3, 9）

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026b. （原文页码标记：2, 3）

Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do World Action Models Need Test-time Future Imagination? arXiv preprint arXiv:2603.16666, 2026. （原文页码标记：3, 9）

Qihang Zhang, Lin Li, Luyao Zhang, Shuai Yang, Yiming Luo, Shuaiting Li, Ruilin Wang, Junke Wang, Jiahao Shao, Gangwei Xu, et al. Native Video-Action Pretraining for Generalizable Robot Control. arXiv preprint arXiv:2607.08639, 2026a. URL https://arxiv.org/abs/2607.08639. （原文页码标记：3, 9）

Yang Zhang, Jiangyuan Zhao, Chenyou Fan, Fangzheng Yan, Tian Li, Haitong Tang, Sen Fu, Xuan’er Wu, Qizhen Weng, Weinan Zhang, Xiu Li, Chi Zhang, Chenjia Bai, and Xuelong Li. PRTS: A Primitive Reasoning and Tasking System via Contrastive Representations. arXiv preprint arXiv:2604.27472, 2026b. （原文页码标记：9）

Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023. （原文页码标记：3）

Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model. arXiv preprint arXiv:2510.10274, 2025. （原文页码标记：7, 9）

Pengfei Zhou, Shengcong Chen, Di Chen, Jiaxu Wang, Rongjun Jin, Bingwen Zhu, Yike Pan, Songen Gu, Kuanning Wang, Shufeng Nan, Xingyu Qiu, Chenhao Qiu, Pu Yang, Yunuo Cai, Jianxiong Gao, Yifan Li, Yanwei Fu, Xiangyu Yue, Zhi Chen, and Jianlan Luo. τ0-WM: A Unified Video-Action World Model for Robotic Manipulation. arXiv preprint arXiv:2606.01027, 2026. （原文页码标记：2）

Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets, 2025. URL https://arxiv.org/abs/2504.02792. （原文页码标记：4）

Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.
 （原文页码标记：3）

## A. 训练与数据处理细节

### A.1. 轨迹质量标注

我们根据标注粒度使用两种数值质量尺度。逐回合标注使用 1–12 的尺度，其中 rollout 分数位于 $[1,10]$，专家示范被赋予 $q^\star=12$。粗粒度标注使用 1–5 的尺度，其中 rollout 分数取自 $\{1,2,3,4\}$，专家示范被赋予 $q^\star=5$。所有设置中，每条失败轨迹都获得 $q=1$。专家示范被视为最优参考轨迹，分数高于 rollout 范围；部署时也使用相同的专家分数。每个分数都分配给一条完整轨迹，并由该轨迹提取的所有训练片段继承。

**RoboChallenge Table30-V2。** 我们使用 RoboChallenge 官方平台提供的人工逐回合标注。平台操作人员已经根据任务完成情况，使用 1–10 的尺度为每条 rollout 评分；成功轨迹通常至少获得 8.5 分。我们直接将平台评分作为 rollout 质量分数，但所有失败轨迹都改赋 $q=1$。专家示范获得 $q^\star=12$。

**单任务叠衣后训练。** 我们同样使用人工逐回合标注。每次 rollout 后，数据采集者会在重置场景之前评估任务完成情况，并采用与 RoboChallenge 相同的、基于完成情况的评分方式，在 1–10 的尺度上评分。失败轨迹获得 $q=1$，任务专用专家示范获得 $q^\star=12$。

**RoboTwin 2.0。** 我们分别对专家训练的 LoopWAM、X-VLA 和 π₀.₅ 的 rollout 进行粗粒度标注。对于每种策略，我们计算其跨所有任务的总体 rollout 成功率，并人工检查约 0.7% 的轨迹，以评估动作平滑性和动作准确度。这两项评估共同决定该策略所有成功轨迹的统一分数；每条失败轨迹获得 $q=1$。专家训练的 LoopWAM 在 rollout 收集期间成功率约为 90%，且轨迹相对平滑，因此其成功 rollout 获得 $q=4$。X-VLA 和 π₀.₅ 的成功率均低于 50%，且产生的轨迹平滑性较差，因此其成功 rollout 获得 $q=3$。专家示范获得 $q^\star=5$。

**RoboCasa365。** 我们使用相同的粗粒度方案，根据各来源策略跨任务的总体 rollout 成功率，并人工检查约 0.7% 的轨迹以评估动作平滑性和准确度。我们使用 64 步和 32 步动作块，分别得到两个专家训练的 LoopWAM 策略，称为策略 1 和策略 2；此外还收集 RLDX-1 的 rollout。按照 RoboTwin 使用的相同评分标准，策略 1 的所有成功 rollout 共用 $q=3$，策略 2 的共用 $q=4$，RLDX-1 的共用 $q=2$。所有失败 rollout 获得 $q=1$，专家示范获得 $q^\star=5$。

### A.2. 训练细节

**时间对齐和机器人表征。** 训练配置使用 $H=64$ 的动作时间跨度，并每隔 8 个控制步取一帧视觉图像，因此总计得到 9 帧 RGB 图像，其中包含初始观测。多个相机视角拼接成一张图像；机器人状态和动作使用专家数据统计量归一化。对于 Table30-V2，关节目标表示为相对于每个动作块开始时关节状态的相对量，而夹爪命令保持绝对值。执行时，只有关节分量会转换回绝对目标；夹爪分量不做差分。为了减少冗余的静止数据，真实世界数据流水线会检查未归一化的状态序列，使用绝对容差 $10^{-3}$，并对近乎静止的窗口最多重新采样三次。专家回合最后 20 秒内开始的窗口不参与此筛选，以保留终止行为。去噪损失不计入填充部分。

**噪声调度和损失加权。** 基础噪声采样器对 $u\sim\mathcal{U}(0,1)$ 使用 $\varphi_5(u)=5u/(1+4u)$。从式 (4) 的 Beta 分布中采样得到的值会裁剪到 $[10^{-4},1-10^{-4}]$，不再进行额外的时间平移。两种模态使用相同的调度器加权函数，其定义见式 (5)：

$$
\begin{aligned}
g(\sigma)&=\exp\left[-2\left(\sigma-\frac{1}{2}\right)^2\right]-e^{-1/2},\\
\omega(\sigma)&=\frac{g(\sigma)}{C+10^{-10}},\qquad
C=\frac{1}{T}\sum_{j=0}^{T-1}g\left(\varphi_5\left(1-j/T\right)\right),\qquad T=1000.
\end{aligned}
$$

<div align="right">(8)</div>

专家样本和 rollout 样本共用这一归一化方式。对于 $K$ 步推理，噪声调度为 $\sigma_k=\varphi_5(1-k/K)$，其中 $k=0,\ldots,K$。每种模态均按 Euler 更新 $X_m^{k+1}=X_m^k+(\sigma_{k+1}-\sigma_k)v_{\theta_m}$；向量场在联合采样过程中一起计算，初始观测潜变量在整个采样期间保持固定。

### A.3. Table30-V2 子任务分割

我们使用 Qwen2.5-VL-7B 进行子任务分割，在机器人在线执行期间切换到各阶段对应的语言指令。表 5 列出了由此得到的有序提示。图中展示的四项任务是按按钮、给植物浇水、碎纸和印章定位。其余任务保留原始的完整任务指令。

**表 5：** Table30-V2 子任务指令，按执行顺序排列。

| 任务 | 机器人 | 有序子任务指令 |
|---|---|---|
| 按按钮 | ARX5 | 1. 按下粉色按钮。<br>2. 按下蓝色按钮。<br>3. 按下绿色按钮。<br>4. 按下黄色按钮，然后收回夹爪。 |
| 给花浇水 | ARX5 | 1. 拿起浇水壶，给盆栽浇水。<br>2. 浇水完成后，把浇水壶放下。 |
| 碎纸 | UR5 | 1. 把纸放进碎纸机。<br>2. 收回夹爪。 |
| 印章定位 | ALOHA | 1. 拿起印章。<br>2. 蘸一下印泥。<br>3. 在签名区域盖章，然后返回。 |

### A.4. 子任务切换示例

图 4–7 分别展示四项任务中的一条轨迹。第一张图像是执行开始时拼接后的策略输入；此后每张图像都在当前子任务提示发生切换时采集。每个阶段还配有该阶段使用的指令。

**阶段指令：**

- 图 4，按按钮（ARX5）：阶段 1，按下粉色按钮；阶段 2，按下蓝色按钮；阶段 3，按下绿色按钮；阶段 4，按下黄色按钮，然后收回夹爪。
- 图 5，给植物浇水（ARX5）：阶段 1，拿起浇水壶，给盆栽浇水；阶段 2，浇水完成后，把浇水壶放下。
- 图 6，碎纸（UR5）：阶段 1，把纸放进碎纸机；阶段 2，收回夹爪。
- 图 7，印章定位（ALOHA）：阶段 1，拿起印章；阶段 2，蘸一下印泥；阶段 3，在签名区域盖章，然后返回。

[图 4–7 原图（PDF 第 16–18 页）](loopwam.pdf#page=16)

**图 4：** 按按钮（ARX5）的子任务分割示例。按顺序展示初始观测（阶段 1）和后续各阶段的观测，每张图像都配有对应的子任务指令。

**图 5：** 给植物浇水（ARX5）的子任务分割示例。按顺序展示初始观测（阶段 1）和后续各阶段的观测，每张图像都配有对应的子任务指令。

**图 6：** 碎纸（UR5）的子任务分割示例。按顺序展示初始观测（阶段 1）和后续各阶段的观测，每张图像都配有对应的子任务指令。

**图 7：** 印章定位（ALOHA）的子任务分割示例。按顺序展示初始观测（阶段 1）和后续各阶段的观测，每张图像都配有对应的子任务指令。

## B. RoboChallenge Table30-V2 排行榜

图 8 展示评估时的 RoboChallenge Table30-V2 官方排行榜。LoopWAM 排名第一，总体得分为 56.80，成功率为 43.33%。

[图 8 原图（PDF 第 20 页）](loopwam.pdf#page=20)

**图 8：** RoboChallenge Table30-V2 排行榜。LoopWAM 为每种机器人本体使用一个多任务策略，排名第一，总体得分为 56.80，成功率为 43.33%。

## C. 失败恢复与迭代执行

真实世界机器人操作经常会遇到由感知不完善、执行不准确和复杂物体交互动力学引起的不确定性。

为了分析 LoopWAM 支持迭代执行的能力，图 9 展示了初次尝试未能完成目标任务的定性示例。观察到由此产生的环境状态后，模型会生成后续纠正行为，以改善执行结果。这些示例凸显了对未来状态和环境转移进行建模的重要性，而不应只预测孤立动作。

**图 9 中的任务与阶段标记：**

- 整理化妆台：第一次尝试；拿取成功但物品掉落；第二次尝试；成功！
- 用粘毛滚筒清理污物：第一次尝试；清理不完整；第二次尝试；清理不完整；成功！
- 清扫垃圾：第一次尝试；碎屑滚回原处；第二次尝试；第三次尝试；成功！

[图 9 原图（PDF 第 21 页）](loopwam.pdf#page=21)

**图 9：** 迭代执行与失败恢复的定性示例。第一次尝试可能因执行不确定性、交互不完整或物体移位而失败。后续尝试会生成纠正行为，并成功完成任务。

## D. 未来观测预测的定性结果

作为学习得到的机器人世界模型，LoopWAM 根据当前机器人状态和交互历史预测未来视觉观测。

为了进一步考察模型学到的视觉动力学，图 10 比较了预测的未来帧与机器人执行时的真实观测。

预测结果捕捉了场景演变的主要方面，包括机器人运动、物体移动和操作进度。尽管生成的观测在细微视觉细节上可能不同于真实图像，但它们保留了交互过程的时间结构。

**图 10 中的标记与任务指令：** GT 表示真实观测，Pred 表示预测观测；head 和 left 分别指头部相机视角和左侧相机视角。两个示例任务指令分别是“把桌上的笔放进笔筒”和“把蓝色碗放进米色碗，再把绿色碗放进蓝色碗”。

[图 10 原图（PDF 第 22 页）](loopwam.pdf#page=22)

**图 10：** LoopWAM 预测的未来观测与机器人真实执行结果的定性对比。模型捕捉了机器人与物体交互的时间演变，并预测出合理的未来视觉状态。

## E. 代表性机器人平台与任务

LoopWAM 在多种机器人平台上评测，这些平台具有不同的机器人本体、感知配置和操作空间。

为了展示学到的机器人行为的多样性，我们提供来自 ARX、ALOHA、UR5 和 W1 平台的更多定性示例。这些示例涵盖物体操作、协同控制、精细交互和长时程任务。

图 11 展示 ARX 平台上的代表性任务。图 12 给出 ALOHA 平台的示例，其中包含协同操作行为。图 13 展示 UR5 平台上的单臂操作任务。最后，图 14 展示 W1 平台上具有代表性的长时程任务。

**图 11：** ARX 机器人平台上的代表性任务。这些示例展示了在不同机器人本体上评估 LoopWAM 时使用的多种操作场景。图中任务指令包括“把 4 朵花放进花瓶”和“给盆栽浇水”。

**图 12：** ALOHA 机器人平台上的代表性任务。这些示例包括需要时间一致控制的协同操作与物体交互。图中任务指令包括“用果酱涂面包”“把书放回书架”“用桌上的布把物品捆在一起”以及“在纸张的签名区域盖章”。

[图 11–12 原图（PDF 第 23 页）](loopwam.pdf#page=23)

**图 13：** UR5 机器人平台上的代表性任务。这些示例展示结构化环境中的单臂操作。图中任务指令包括“把文具放进黄色盒子，把电子物品放进蓝色盒子”和“把纸放进碎纸机”。

**图 14：** W1 机器人平台上具有代表性的长时程任务。这些示例展示复杂环境中的连续操作行为。图中任务指令包括“把桌上的球放到小托盘上，再把小托盘移到大托盘上”“用桌上的绳子打一个结”“把桌上的垃圾扫进簸箕”以及“打开抽屉，把开瓶器放进去，然后关上抽屉”。

[图 13–14 原图（PDF 第 24 页）](loopwam.pdf#page=24)



