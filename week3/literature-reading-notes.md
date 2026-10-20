<!-- managed-by: paper-reading-companion -->
# π0：用于通用机器人控制的视觉-语言-动作流模型

> 原文：`./pi0.pdf`；以下为全文中文翻译。公式、引用编号和专有名词按原文保留。

<!-- id-state: next-paper="2" -->

<!-- index:start -->
## 文献索引

| ID | 来源 | 标题 | 阅读进度 | 最近更新 |
|---|---|---|---|---|
| `p001` | `./pi0.pdf` | π0：用于通用机器人控制的视觉-语言-动作流模型 | 全文（17 页） | 2026-10-09 |
<!-- index:end -->

<!-- paper:start id="p001" -->

## 论文信息

- **作者**：Kevin Black、Noah Brown、Danny Driess、Adnan Esmail、Michael Equi、Chelsea Finn、Niccolo Fusai、Lachy Groom、Karol Hausman、Brian Ichter、Szymon Jakubczak、Tim Jones、Liyiming Ke、Sergey Levine、Adrian Li-Bell、Mohith Mothukuri、Suraj Nair、Karl Pertsch、Lucy Xiaoyang Shi、James Tanner、Quan Vuong、Anna Walling、Haohuan Wang、Ury Zhilinsky
- **机构**：Physical Intelligence，旧金山，加利福尼亚州，美国
- **来源**：https://physicalintelligence.company/blog/pi0

## 图 1

我们的通用机器人策略使用预训练的视觉-语言模型（VLM）作为骨干，并使用包含多种灵巧操作任务的多样化跨本体数据集。我们通过加入一个独立的动作专家来使模型适应机器人控制；该专家通过流匹配产生连续动作，从而实现精确而流畅的操作技能。随后，模型可以直接根据提示执行任务，也可以在高质量数据上进行微调，以完成复杂的多阶段任务，例如折叠多件衣物或组装盒子。

## 摘要

机器人学习有望释放灵活、通用且灵巧的机器人系统的全部潜能，也有望帮助解决人工智能领域一些最深刻的问题。然而，要使机器人学习达到有效现实世界系统所需的通用程度，在数据、泛化能力和鲁棒性方面仍面临重大障碍。本文讨论通用机器人策略（即机器人基础模型）如何应对这些挑战，以及如何为复杂且高度灵巧的任务设计有效的通用机器人策略。我们提出一种新颖的流匹配架构，它建立在预训练视觉-语言模型（VLM）之上，从而继承互联网规模的语义知识。随后，我们讨论如何使用来自多个灵巧机器人平台的大规模、多样化数据集训练这一模型，其中包括单臂机器人、双臂机器人和移动操作机器人。我们从以下方面评估模型：通过直接提示执行任务；遵循人类以及高级 VLM 策略给出的语言指令；以及通过微调获得新技能。实验涵盖多种任务，例如折叠衣物、清理桌面和组装盒子。

## 图 2

π0 控制移动操作机器人折叠衣物。我们的模型在来自 7 种不同机器人配置、涵盖 68 个任务的多样化数据上进行预训练，之后既可以直接提示，也可以针对复杂下游任务进行微调。图中洗衣折叠策略会从烘干机中取出衣物，将其放入洗衣篮，把洗衣篮带到折叠桌，然后依次折叠每件衣物。

# I. 引言

人应该能够换尿布、策划入侵、屠宰猪、驾驶船只、设计建筑、写十四行诗、平衡账目、砌墙、接骨、安慰临终者、接受命令、发布命令、合作、独自行事、解方程、分析新问题、施肥、编程、烹饪美味的饭菜、高效作战、英勇赴死。

专门化是昆虫的特性。

——Robert A. Heinlein，《Time Enough for Love》

人工智能系统形态各异、规模不一：既有解决人类无法处理的复杂问题的高度专用系统，例如预测蛋白质构象 [21]；也有能够根据文本提示生成逼真、高分辨率图像或视频的系统 [40]。然而，人类智能最显著地超越机器智能的维度是多才多艺：在不同的物理环境中解决多种任务，同时对环境约束、语言命令和意外扰动做出明智反应。人工智能在这类多才多艺能力上的最切实进展，也许可以从大型语言模型和视觉-语言模型 [1, 48] 中看到：这些系统在来自互联网的大规模、多样化图像和文本语料上预训练，然后使用更加精心整理的数据集进行微调（“对齐”），以诱导所期望的行为模式和响应方式。虽然已有研究表明，这些模型具备广泛的指令遵循和问题解决能力 [53, 27]，但它们并不像人类一样真正处于物理世界中，对物理交互的理解完全建立在抽象描述之上。如果这些方法要在构建具有人类那种物理情境多才多艺能力的人工智能系统方面取得实质进展，就需要使用物理情境数据训练它们，也就是来自具身机器人智能体的数据。

能够被要求执行多种机器人行为的灵活通用模型具有巨大的实际意义，同时也可能为机器人学习目前面临的一些最困难挑战提供解决方案，例如数据可得性、泛化和鲁棒性。在自然语言 [1] 和计算机视觉 [39] 中，在多样化多任务数据上预训练的通用基础模型往往优于针对狭窄任务定制的专用方案。例如，如果目标是在照片中识别鸟类，那么先在许多不同的图像-语言关联上预训练，再针对鸟类识别任务进行微调或提示，可能比只在鸟类识别数据上训练更有效。同样，对于有效的专用机器人系统，我们也许会发现，先在高度多样化的机器人数据上预训练，再针对目标任务微调或提示，会更加有效。这可以缓解数据稀缺问题，因为通用模型可以获得更多数据来源，包括其他任务、其他机器人，甚至非机器人来源的数据；它也可能缓解鲁棒性和泛化问题，因为多样化数据覆盖了更广泛的观测和动作，提供了更丰富的场景、纠错行为和恢复行为，而这些内容可能不会出现在更狭窄的专用数据中。因此，采用大规模预训练方法进行机器人学习，有潜力应对该领域的许多挑战，使真正由学习驱动的实用机器人变成现实，同时推进我们对人工智能最深层问题的理解。

然而，开发这种通用机器人策略——即机器人基础模型——涉及若干重大挑战。第一，这类研究必须在非常大的规模上进行，因为大规模预训练的全部收益往往不会在较小规模上出现 [54]。第二，需要开发能够有效利用多种数据源的模型架构，同时能够表示与复杂物理场景交互所需的细致、微妙行为。第三，需要恰当的训练配方。这也许是最重要的因素，因为自然语言处理和计算机视觉领域近期的大模型进展，很大程度上依赖于精心设计的预训练和后训练数据整理策略 [35]。

本文提出一个称为 π0 的原型模型与学习框架，说明如何处理上述三个瓶颈。图 1 展示了我们的模型和系统。为了纳入多样化数据源，我们首先利用预训练视觉-语言模型（VLM）引入互联网规模的经验。以 VLM 为基础，可以继承语言模型和视觉-语言模型所拥有的通用知识、语义推理能力和问题解决能力。随后，我们进一步训练模型以纳入机器人动作，将其转变为视觉-语言-动作（VLA）模型 [7]。为了能够使用各种不同的机器人数据源，我们采用跨本体训练 [10]，把许多机器人类型的数据组合到同一个模型中。这些机器人具有不同的配置空间和动作表示，包括单臂系统、双臂系统以及移动操作机器人。此外，为了执行高度灵巧而复杂的物理任务，我们使用带有流匹配（扩散的一种变体）的动作分块架构 [57] 来表示复杂的连续动作分布 [28, 32]。这使模型能够以最高 50 Hz 的频率控制机器人完成洗衣折叠等灵巧任务（见图 1）。为了把流匹配与 VLM 结合起来，我们使用一个新颖的动作专家，用基于流的输出增强标准 VLM。

与语言模型一样，模型架构只是方法的一部分。为了灵活、鲁棒地执行复杂任务，还需要正确的训练配方。我们的配方仿照超大规模语言模型和图像-语言模型中常见的预训练/后训练分离方式 [1, 48]：模型先在非常大且多样化的语料上预训练，再在更狭窄、更精心整理的数据上微调，以诱导目标行为；在我们的场景中，这些行为是灵巧性、效率和鲁棒性。直观地说，只在高质量数据上训练并不能教会模型如何从错误中恢复，因为这类数据很少包含错误；只在质量较低的预训练数据上训练，又不能教会模型高效、鲁棒地行动。二者结合可以得到理想行为：模型尽可能以接近高质量数据的方式行动，同时保留在犯错时可调用的恢复和纠错策略。

我们的贡献包括：一种基于 VLM 预训练和流匹配的新型通用机器人策略架构；以及对机器人基础模型预训练/后训练配方的实证研究。我们评估模型直接接收语言命令时的表现、在下游任务上的微调表现，以及与输出中间语言命令的高级语义策略结合时执行复杂长时任务的表现。虽然模型和系统使用了近期工作中的多种思想，但这些要素的组合是新颖的；实证评估显示，其灵巧性和通用性显著超越此前展示的机器人基础模型。我们在超过 10,000 小时机器人数据上进行预训练，并针对多种灵巧任务进行微调，包括折叠衣物（见图 2）、清理桌面、把餐具放入微波炉、把鸡蛋叠放进蛋盒、组装盒子以及装袋杂货。

# II. 相关工作

我们的工作建立在近期提出的大规模机器人学习方法和多模态语言模型之上。与我们最接近的是近期提出的视觉-语言-动作（VLA）模型，这些模型使用预训练 VLM 并针对机器人控制进行微调 [7, 24, 55]。这类模型采用自回归离散化，以类似文本 token 的方式表示动作。相比之下，我们提出一种新设计，对 VLM 进行微调，使其通过流匹配 [32, 28]（扩散 [20, 46] 的一种变体）产生动作。这使模型能够处理高频动作块 [57]（最高 50 Hz）和高度灵巧的任务；我们表明，这些任务对先前的自回归 VLA 构成了重大挑战 [7]。这一方法与近期一些使用扩散模型生成动作的工作相似 [9, 60]。不同之处在于，我们的模型使用预训练 VLM 骨干 [5]。我们的贡献从根本上具有整合性，关注机器人基础模型的完整框架，不仅包括模型架构，还包括预训练配方、预训练和后训练阶段，以及一系列现实世界实验。

在机器人控制之外，已有许多工作将预训练语言模型与扩散模型结合 [40, 41, 14]，其中也包括专门混合扩散与自回归大型语言模型的模型 [19, 29, 59]。这些模型通常关注图像生成，但我们的动作生成模型建立在若干先前提出的概念之上。与 Zhou 等人 [59] 类似，我们使用施加在单个序列元素上的扩散式（流匹配）损失训练模型，以替代仅解码器 Transformer 的标准交叉熵损失。与 Liu 等人 [29] 类似，我们为对应扩散过程的 token 使用单独的一组权重。我们将这些概念引入 VLA 模型，据我们所知，提出了第一种能够为灵巧控制产生高频动作块的流匹配 VLA。

我们的工作还建立在大规模机器人学习的丰富历史之上。早期工作常使用自监督或自主数据收集 [26, 22, 8]，为抓取 [18, 37] 或推动 [56] 等简单任务提供可行的数据源，但尚不能处理更加灵巧的行为。近来，已有若干高质量机器人控制数据集被收集出来，使模型能够广泛泛化 [23, 10, 52, 33, 34, 43, 13, 6]；不过，这些数据集通常针对更简单的任务，例如物体搬运和基础家具操作（如打开抽屉）[31, 15]。更灵巧的任务通常在较小规模上研究，训练轨迹数量一般只有几十或几百条 [57]，相当于不超过 10 小时的数据。由于我们的目标之一是研究复杂而灵巧的行为，我们使用了更大的数据集，包含约 10,000 小时演示数据，并辅以开源 OXE 数据集 [10]。据我们所知，就机器人数据量而言，这是迄今规模最大的机器人学习实验。我们在这一规模上表明，更复杂的预训练/后训练配方非常有效：类似大型语言模型所采用的配方，预训练阶段赋予模型广泛的知识基础，之后再用更高质量的精选数据在后训练阶段进行细化，以获得期望行为。

我们展示的任务复杂度远高于此前工作。近期工作展示了打鞋带 [58] 或烹饪虾 [17] 等更加复杂、灵巧的行为；我们进一步表明，该框架能够学习持续时间很长的任务，有时长达几十分钟，这些任务同时包含身体灵巧性和组合复杂性。例如，洗衣折叠任务要求机器人操作各种可能以任意配置开始的衣物，并按顺序折叠多件衣物。我们的餐桌收拾任务要求机器人识别新物体属于垃圾还是餐具。我们表明，可以使用一个跨本体模型作为这些任务的基础模型。据我们所知，我们的工作展示了端到端机器人学习文献中持续时间最长的灵巧任务。

# III. 概览

图 3 给出了模型和训练流程的概览。在训练框架中，我们首先组装一个预训练混合数据集，它由我们自己的灵巧操作数据集（第 V-C 节）与完整 OXE 数据集 [10] 的加权组合构成。前者采集自 7 种不同机器人配置，涵盖 68 个不同任务；OXE 数据集包含来自 22 种机器人的数据。预训练阶段（第 V-A 节）还使用多样化的语言标签，将任务名称和片段标注结合起来；片段标注是针对子轨迹的细粒度标签，通常长度约为 2 秒。预训练阶段的目的，是训练一个具备广泛能力和泛化能力的基础模型，而不是让它专门在某一个任务上获得最高性能。该基础模型能够遵循语言命令，以初步熟练程度完成多种任务。对于复杂且灵巧的任务，我们随后采用后训练流程（第 V-A 节），使用高质量精选数据使模型适应特定下游任务。我们同时研究两种设置：使用少量到中等规模数据进行高效后训练；以及使用更大规模高质量数据集，针对洗衣折叠和移动操作等复杂任务进行后训练。

**图 3：我们的框架概览。** 我们从一个预训练混合数据集开始，其中既包括自己的灵巧操作数据集，也包括开源数据。我们使用该混合数据训练流匹配 VLA 模型。模型由较大的 VLM 骨干和较小的动作专家组成，后者处理机器人状态和动作。VLM 骨干权重从 PaliGemma [5] 初始化，从而提供大规模互联网预训练获得的表示。最终的 π0 模型可以控制具有不同动作空间的多种机器人本体，以完成多种多样的任务。

我们在第 IV 节描述的模型以 PaliGemma 视觉-语言模型 [5] 为基础，并使用混合数据进一步训练。为了把基础 PaliGemma VLM 转换为 π0，我们加入使用流匹配 [32, 28] 生成连续动作分布的动作输出。下一节将详细描述这一设计。我们使用 PaliGemma 是出于便利以及它相对较小的规模（有利于实时控制），但我们的框架兼容任何基础预训练 VLM。

# IV. π0 模型

图 3 所示的 π0 模型主要由语言模型 Transformer 骨干组成。遵循标准的后融合 VLM 配方 [3, 11, 30]，图像编码器把机器人的图像观测嵌入到与语言 token 相同的嵌入空间中。我们进一步为该骨干加入机器人专用的输入和输出，即本体感知状态和机器人动作。π0 使用条件流匹配 [28, 32] 建模动作的连续分布。流匹配为模型提供高精度和多模态建模能力，因此特别适合高频灵巧任务。

我们的架构受到 Transfusion [59] 的启发。Transfusion 使用多个目标训练单个 Transformer：对应连续输出的 token 使用流匹配损失监督，对应离散输出的 token 使用交叉熵损失监督。基于 Transfusion，我们进一步发现，为机器人专用的动作和状态 token 使用单独的一组权重可以提升性能。这种设计类似于具有两个混合元素的混合专家结构 [45, 25, 12, 16]：第一个元素处理图像和文本输入，第二个元素处理机器人专用输入与输出。我们将第二组权重称为**动作专家**。

> 注：本文中的“token”指序列维度上的输入/输出槽位，无论该槽位对应离散变量（如语言 token）还是连续变量（如图像 patch 或机器人动作）。

形式化地说，我们希望建模数据分布 $p(A_t\mid o_t)$，其中
$$
A_t=[a_t,a_{t+1},\ldots,a_{t+H-1}]
$$
表示未来动作组成的动作块（我们的任务使用 $H=50$），而 $o_t$ 表示观测。观测由多张 RGB 图像、语言命令和机器人的本体感知状态组成：
$$
o_t=[I_t^1,\ldots,I_t^n,\ell_t,q_t],
$$
其中 $I_t^i$ 是第 $i$ 张图像（每个机器人使用 2 或 3 张图像），$\ell_t$ 是语言 token 序列，$q_t$ 是关节角向量。图像 $I_t^i$ 和状态 $q_t$ 分别通过对应编码器编码，再经线性投影层映射到与语言 token 相同的嵌入空间。

动作块 $A_t$ 中的每一个动作 $a_{t'}$ 都有一个对应的动作 token，并通过动作专家处理。在训练期间，我们使用条件流匹配损失 [28, 32] 监督这些动作 token：

$$
\mathcal{L}^{\tau}(\theta)=\mathbb{E}_{p(A_t\mid o_t),q(A_t^{\tau}\mid A_t)}\left[\left\|v_{\theta}(A_t^{\tau},o_t)-u(A_t^{\tau}\mid A_t)\right\|^2\right].
$$

下标表示机器人的时间步，上标表示流匹配时间步，且 $\tau\in[0,1]$。近期关于高分辨率图像 [14] 和视频 [38] 合成的工作表明，当结合简单的线性高斯（或最优传输）概率路径 [28] 时，流匹配可以取得很强的实证表现。该概率路径为
$$
q(A_t^{\tau}\mid A_t)=\mathcal{N}(\tau A_t,(1-\tau)I).
$$
实际训练时，我们采样随机噪声 $\epsilon\sim\mathcal{N}(0,I)$，计算“带噪动作”
$$
A_t^{\tau}=\tau A_t+(1-\tau)\epsilon,
$$
然后训练网络输出 $v_\theta(A_t^{\tau},o_t)$，使其匹配去噪向量场
$$
u(A_t^{\tau}\mid A_t)=A_t-\epsilon.
$$
动作专家使用完全双向的注意力掩码，因此所有动作 token 都可以相互关注。训练时，我们从强调较低（噪声更大）时间步的 beta 分布中采样流匹配时间步；更多细节见附录 B。

推理时，我们从随机噪声 $A_t^0\sim\mathcal{N}(0,I)$ 开始，把学习到的向量场从 $\tau=0$ 积分到 $\tau=1$，从而生成动作。我们使用前向 Euler 积分规则：
$$
A_t^{\tau+\delta}=A_t^{\tau}+\delta v_\theta(A_t^{\tau},o_t),
$$
其中 $\delta$ 是积分步长。实验使用 10 个积分步骤（对应 $\delta=0.1$）。通过缓存前缀 $o_t$ 的注意力 key 和 value，并在每个积分步骤只重新计算动作 token 对应的后缀，可以高效实现推理。附录 D 进一步介绍推理流程，包括模型各部分的推理时间。

原则上，我们的模型可以从零初始化，或从任意 VLM 骨干微调而来；但实践中我们使用 PaliGemma [5] 作为基础模型。PaliGemma 是一个拥有 30 亿参数的开源 VLM，在规模和性能之间提供了便利的折中。我们为从零初始化的动作专家增加 3 亿参数，总参数量达到 33 亿。附录 B 对模型架构给出了完整描述。

除了主要 VLA 模型外，我们还训练了一个不使用 VLM 初始化的类似基线模型，用于消融实验。该模型称为 π0-small，拥有 4.7 亿参数，不使用 VLM 初始化，并有一些有利于在没有 VLM 初始化时进行数据训练的小改动；这些改动总结在附录 C 中。该模型用于比较，以评估引入 VLM 预训练的收益。

# V. 数据收集与训练配方

广泛能力的机器人基础模型不仅需要富有表现力且强大的架构，更需要合适的数据集，尤其需要合适的训练配方。正如大型语言模型的训练通常分为预训练和后训练阶段，我们也为模型采用多阶段训练流程。预训练阶段的目标是让模型接触多种任务，使其获得广泛适用的通用物理能力；后训练阶段的目标则是使模型能够熟练、流畅地执行目标下游任务。由于两阶段的目标不同，其数据集要求也不同：预训练数据集应尽可能覆盖更多任务，并且在每个任务内部覆盖多样化行为；后训练数据集则应覆盖有助于有效执行任务的行为，并体现一致、流畅的策略。直观地说，多样但质量较低的预训练数据允许模型从错误中恢复并处理高度变化的情形，而这些情形可能不会出现在高质量后训练数据中；后训练数据则教会模型把任务做好。

## A. 预训练与后训练

图 4 概览了我们的预训练混合数据。由于每个训练样本对应一个时间步，即一个二元组 $(o_t,A_t)$，本节以时间步数量衡量数据规模。训练混合数据中有 9.1% 来自开源数据集，包括 OXE [10]、Bridge v2 [52] 和 DROID [23]。这些数据集中的机器人和任务通常使用一到两台相机和低频控制，频率在 2–10 Hz 之间，但覆盖了广泛的物体和环境。

为了学习更加灵巧、复杂的任务，我们还使用自己数据集中的 9.03 亿个时间步，其中 1.06 亿步来自单臂机器人，7.97 亿步来自双臂机器人。该数据包含 68 个任务，每个任务由复杂行为组成。例如，“餐桌收拾”任务要求将各种盘子、杯子和餐具放入收拾箱，并把大量不同的垃圾物品放入垃圾桶。这里的“任务”定义与以往工作明显不同；以往工作通常把任意名词与动词组合（例如“拿起杯子”与“拿起盘子”）视为不同任务。因此，我们数据集实际覆盖的行为范围远比 68 个任务这个数字所暗示的更广。

由于数据集规模存在一定不平衡（例如更困难的洗衣折叠任务占比过高），我们按照 $n^{0.43}$ 对每个“任务-机器人”组合加权，其中 $n$ 是该组合的样本数量，从而降低过度代表组合的权重。配置向量 $q_t$ 和动作向量 $a_t$ 始终使用数据集中最大机器人的维度（本例为 18，以容纳两条 6 自由度机械臂、两个夹爪、一个移动底座和一个垂直驱动的躯干）。对于配置空间和动作空间维度更低的机器人，我们对配置向量和动作向量进行零填充。对于图像少于三张的机器人，我们还会屏蔽缺失的图像槽位。

在后训练阶段，我们使用更小的任务专用数据集对模型进行微调，使其专门适配特定下游应用。如前所述，我们对“任务”的定义较宽泛，例如“餐桌收拾”要求操作大量不同物体。不同任务需要非常不同的数据集：最简单的任务只需要 5 小时数据，最复杂的任务则使用 100 小时或更多数据。

## B. 语言与高级策略

需要语义推理和高级策略的复杂任务（如收拾餐桌）也可以从高级策略中受益。高级策略把高级任务（如“收拾餐桌”）分解为更直接的子任务（如“拿起餐巾”或“把餐巾扔进垃圾桶”）。由于我们的模型经过语言输入训练，我们可以使用高级 VLM 做出这些语义推断；这种方法类似于 SayCan [2] 等 LLM/VLM 规划方法。我们在若干实验任务中使用这种高级策略协助模型制定高层战略，第 VI 节将对此进行讨论。

## C. 机器人系统细节

我们的灵巧操作数据集包含 7 种不同的机器人配置和 68 个任务，平台见图 5。

**图 4：** 我们的数据集概览。预训练混合数据由 OXE [10] 的子集和 π 数据集组成。我们使用 OXE 的一个子集，将其称为 OXE Magic Soup [24]。左图按步数展示不同数据集的相对规模，右图展示不同数据集在预训练混合数据中的权重。

**图 5：实验中使用的机器人。** 这些机器人包括配备 6 自由度和 7 自由度机械臂的单臂、双臂操作机器人，以及全向和非全向移动操作机器人。π0 在所有这些平台上联合训练。

- **UR5e：** 配备平行夹爪、腕部相机和肩部上方相机的机械臂，共两张相机图像，配置空间和动作空间均为 7 维。
- **双臂 UR5e：** 两套 UR5e，共三张相机图像，配置空间和动作空间均为 14 维。
- **Franka：** 两张相机图像，配置空间和动作空间均为 8 维。
- **双臂 Trossen：** 两条 6 自由度 Trossen ViperX 机械臂，配置基于 ALOHA [4, 57]，配备两台腕部相机和一台底座相机，配置空间和动作空间均为 14 维。
- **双臂 ARX 与双臂 AgileX：** 两条 6 自由度机械臂，支持 ARX 或 AgileX，配备三台相机（两台腕部相机和一台底座相机），配置空间和动作空间均为 14 维。两者是不同平台，但运动学性质相近，因此归为一类。
- **移动 Trossen 与移动 ARX：** 基于 Mobile ALOHA [57]，移动底座上有两条 6 自由度机械臂，可以是 ARX 或 Trossen ViperX。非全向底座增加两个动作维度，因此配置空间为 14 维、动作空间为 16 维。系统配备两台腕部相机和一台底座相机。
- **移动 Fibocom：** 全向底座上的两条 6 自由度 ARX 机械臂。底座增加三个动作维度（两个平移维度和一个朝向维度），因此配置空间为 14 维、动作空间为 17 维。

图 4 总结了各类机器人在数据集中的比例。

# VI. 实验评估

实验评估包括开箱即用实验，用于比较预训练基础模型；直接提示的替代模型设计；以及详细的微调实验，用于评估模型在具有挑战性的下游任务上的表现，并与其他灵巧操作方法进行比较。我们研究以下问题：

1. π0 在预训练数据中出现的多种任务上预训练后表现如何？我们直接评估 π0，并与其他机器人基础模型比较。
2. π0 遵循语言命令的能力如何？我们将 π0 与未使用 VLM 初始化的较小版本 π0-small 比较，并使用人类提供的命令以及高级 VLM 策略指定的命令进行评估。
3. π0 与专门应对灵巧操作任务的方法相比如何？我们研究可以从预训练初始化微调、也可以使用任务专用数据从零训练的下游任务，并与此前提出的灵巧操作方法比较。
4. π0 能否适应复杂的多阶段任务？最后，我们把 π0 微调到折叠衣物、收拾餐桌等特别复杂的任务。这些任务需要 5–20 分钟才能完成，其中有些还需要高级策略的引导。

## A. 评估基础模型

第一组实验在完整混合数据上预训练模型，不进行后训练，以评估基础模型执行多种任务的能力。我们与其他机器人基础模型以及在同一预训练混合数据上从零训练的模型比较。图 6 展示了以下任务，每项任务都通过语言命令交给同一个基础模型：

- **折叠衬衫：** 机器人必须折叠一件已经铺平的 T 恤。
- **简单收拾餐桌：** 机器人必须清理桌面，把垃圾放进垃圾桶，把餐具放进餐具箱；分数表示放入正确容器的物体数量。
- **困难收拾餐桌：** 简单任务的更困难版本，包含更多物体和更具挑战的配置，例如故意把餐具放在垃圾上、物体互相遮挡，以及预训练数据中未出现的物体。
- **装袋杂货：** 机器人必须把所有杂货装袋，例如薯片、棉花糖和猫粮。
- **从烤面包机取出吐司：** 机器人从烤面包机中取出吐司。

**图 6：开箱即用评估任务。** 基础模型预训练后，我们在五项任务上进行测试：折叠衬衫、简单收拾餐桌、困难收拾餐桌、装袋杂货和从烤面包机取出吐司。这些任务要求结合灵巧操作、多阶段行为和语义识别。

由于能够在这一规模运行的先前模型很少，为这些实验提供公平比较很困难。我们比较 OpenVLA [24]（一个最初在 OXE 数据集 [10] 上训练的 70 亿参数 VLA 模型），并在完整混合数据上训练 OpenVLA。这对 OpenVLA 是很困难的混合数据，因为它不支持动作分块或高频控制。我们还比较 Octo [50]（一个较小的 9300 万参数模型）。虽然 Octo 不是 VLA，但它使用扩散过程生成动作，因此是流匹配 VLA 的有价值比较对象；我们在相同混合数据上训练 Octo。

由于时间限制，我们无法让 OpenVLA 和 Octo 训练与完整模型相同的轮数。因此还比较 π0 的“计算量对等”版本：它只训练 160k 步，而主模型训练 700k 步，这与基线获得的步数相同或更少（OpenVLA 为 160k，Octo 为 320k）。我们还加入只在 UR5e 数据上微调、没有进行跨本体训练的 OpenVLA 版本，希望得到更强的 UR5e 基线。最后，我们比较第 IV 节所述的 π0-small，它可以视为没有 VLM 预训练的缩小版模型。

评估指标是每种任务和方法进行 10 个回合后得到的归一化平均分数：完全成功得分 1.0，部分成功得到分数区间内的相应分数。例如，收拾餐桌任务的分数是正确放入相应容器的物体比例。评分规则见附录 E。图 7 的结果表明，在所有开箱即用任务上，π0 都取得了明显最好的结果：折叠衬衫和简单收拾餐桌接近完全成功，并且相对于所有基线都有显著提升。仅训练 160k 步的计算量对等版本仍优于所有基线，甚至 π0-small 也优于 OpenVLA 和 Octo。

OpenVLA 在这些任务上表现不佳，因为其自回归离散化架构不支持动作分块。只使用 UR5e 的 OpenVLA 版本表现更好，但仍低于 π0。Octo 支持动作分块，却具有相对有限的表示能力。这说明必须把大型、富有表现力的架构与通过流匹配或扩散建模复杂分布的能力结合起来。与 π0-small 的比较还说明纳入 VLM 预训练的重要性。不过，该比较并不完全公平：π0-small 参数更少，而没有预训练时，大模型更难使用。总体而言，这些实验表明，π0 是一个强大的预训练模型，能够在多种机器人上有效执行多种任务，性能明显优于此前模型。

## B. 遵循语言命令

下一组实验中，我们微调基础 π0 模型，使其遵循一组评估领域中的语言命令，并与第 IV 节所述的 π0-small 比较。π0-small 不使用 VLM 初始化，因此该实验旨在衡量 VLM 预训练对语言指令遵循能力的提升。π0-small 同时也是明显更小的模型；由于 VLM 初始化既使得训练更大模型成为可能，又改进了语言跟随能力，因此很难消除这一混淆因素。

每项任务的语言命令包括要拿取的物体以及放置物体的位置，语言标注片段长度约为 2 秒。每个完整任务由许多片段组成：

- **收拾餐桌：** 将餐具放入收拾箱，将垃圾放入垃圾桶。
- **摆餐桌：** 从箱子中取出餐垫、餐具、餐巾和杯子摆桌，并根据语言指令调整它们。
- **装袋杂货：** 把咖啡豆、大麦、棉花糖、海藻、杏仁、意大利面和罐头等杂货装入袋子。

图 8 展示了语言条件任务。我们评估五种条件：π0-flat（以及 π0-small-flat）只接收总体任务命令，例如“把杂货装袋”，不接收中间语言命令；π0-human（以及 π0-small-human）接收专家人类用户提供的中间步骤命令，例如要拿哪个物体、放到哪里。这些条件测试模型遵循更详细语言命令的能力。最后，π0-HL 使用第 V-B 节所述的高级 VLM 提供的高级命令，整个过程无需人类专家。

图 9 的结果对每个任务进行 10 次试验取平均，显示 π0 的语言遵循准确率显著高于 π0-small。这说明更大的预训练 VLM 初始化带来了显著提升，并且这种能力转化为在人类专家指导（π0-human）以及高级策略指导（π0-HL）下更好的任务表现。π0 的语言能力直接带来了在高级指导下自主执行复杂任务的改进。

## C. 学习新的灵巧任务

下一组实验评估与预训练数据明显不同的新任务。我们使用不同数量的数据对模型微调。虽然每个任务都是新的，但根据它们与预训练任务的差异程度分为不同等级：

- **UR5e 堆碗：** 堆叠四个大小不同的碗。由于需要像预训练中的餐桌收拾任务一样抓取并移动餐具，因此属于“容易”级别。训练数据包含多种碗，评估中混合了见过和未见过的碗。
- **折叠毛巾：** 类似预训练中的折叠衬衫，因此属于“容易”级别。
- **微波炉中的特百惠盒：** 打开微波炉、放入塑料容器、关闭微波炉。容器形状和颜色不同，微波炉在预训练中未出现。
- **更换纸巾卷：** 从纸巾架上取下旧纸巾卷并换上新的。由于预训练中没有这种物体，属于“困难”级别。
- **Franka 抽屉物品：** 打开抽屉、把物品放入抽屉、关闭抽屉。由于预训练中没有 Franka 机器人上的类似任务，也属于“困难”级别。

微调后，我们将模型与 OpenVLA [24] 和 Octo [50] 比较；二者也采用预训练和微调配方。我们还与专门从小数据集学习灵巧任务的 ACT [57] 和扩散策略 [9] 比较。OpenVLA 和 Octo 使用在 OXE [10] 上训练的公开预训练检查点，然后针对每个任务微调；ACT 和扩散策略只在规模与其原论文实验相近的任务专用数据集上训练。我们评估从预训练基础模型微调的 π0，以及从零开始训练的 π0，从而分别衡量 π0 架构和预训练流程的收益。

图 11 展示了不同微调数据量下、每项任务平均 10 次以上试验的结果。总体上，π0 通常优于其他方法。此前最强的模型有时是完全从零开始在目标任务上训练的模型，这表明在这些领域使用预训练对预训练方法构成挑战。尽管在特百惠任务上使用 5 小时数据时，π0 与基线相近，但使用 1 小时数据的 π0 明显更好。与预期一致，预训练模型在与预训练数据更相似的任务上通常获得更大收益，有时提升达到约 2 倍。

## D. 掌握复杂的多阶段任务

最后一组实验通过微调与语言相结合，处理一系列具有挑战性的多阶段任务。有些任务在预训练中出现过，但仍需微调才能掌握；另一些任务则完全没有出现在预训练中。图 12 展示了这些任务：

- **洗衣折叠：** 静态（非移动）双臂系统折叠衣物。衣物从洗衣篮中随机皱缩的状态开始，机器人需要取出衣物、折叠并放到之前折叠好的衣物堆上。随机初始状态要求策略泛化到任意配置；该任务出现在预训练中。
- **移动洗衣：** 图 5 中的 Fibocom 移动机器人折叠衣物，同时控制朝向和平移；该任务出现在预训练中。
- **烘干机卸货：** Fibocom 移动机器人从烘干机中取出衣物并放入洗衣篮；该任务出现在预训练中。
- **收拾餐桌：** 在杂乱场景中处理大量新物体，比开箱即用基准更困难。策略必须泛化到未见过的各种形状和大小的物体，并执行复杂灵巧动作，例如扭转夹爪抓取大盘子，以及小心抓取玻璃杯等薄而易损的物品。机器人必须处理密集杂乱并智能地安排行为；例如清理带有垃圾的盘子时，必须先拾取盘子，将内容物抖入垃圾桶，再把盘子放入餐具箱。该任务不在预训练中。
- **组装盒子：** 组装一个从摊平状态开始的纸板盒。盒子需要按正确方向弯折；机器人需要按住盒子的一部分，同时折叠其他部分，还要使用双臂和桌面在折叠时支撑。某些折叠可能需要重试，因此需要反应式、智能的策略。该任务不在预训练中。
- **外带盒：** 把盘子上的多种食物移入外带盒，摆放时不能让食物伸出盒外，然后用双臂合上盒子。该任务不在预训练中。
- **装鸡蛋：** 从碗中取出六个鸡蛋，装入蛋盒并合上盒盖。需要根据鸡蛋在碗中的姿态抓取，再放入蛋盒中打开的槽位；鸡蛋的形状、易滑性和放置要求带来挑战，合上盒子需要双臂。该任务不在预训练中。

这些任务要求组合数十种行为（抓取、堆叠、折叠、铺平等），泛化到大量物体配置，并处理可变形物体和柔性纸板等复杂物理属性。

图 13 给出了每项任务 10 次试验的平均分数。分数 1.0 表示完美执行，部分分数表示部分完成，例如 0.5 表示一半物体被正确收拾。任务非常困难，我们无法用其他方法解决，因此比较 π0 的消融版本：预训练后进行微调、只预训练后直接执行（“开箱即用”）、以及不预训练而直接在微调数据上训练（“从零开始”）。π0 可以解决其中许多任务，完整的预训练加微调配方总体表现最好。许多困难任务在使用预训练模型后提升很大，说明预训练对困难任务尤其有用。π0 的绝对性能因任务而异，可能取决于任务难度以及任务在预训练中的覆盖程度。作者建议观看论文网站上的任务视频，以更完整地了解任务及其复杂性；他们认为，这种具有挑战性的自主性能代表了学习策略灵巧机器人操作的新水平。

# VII. 讨论、局限性与未来工作

我们提出了一个训练机器人基础模型的框架，称为 π0：在高度多样化数据上预训练，然后进行开箱即用评估或针对复杂下游任务微调。实证评估研究了同时具备灵巧性、泛化能力和长时多阶段行为的任务。模型把互联网规模 VLM 预训练与流匹配结合起来，以表示复杂的高频动作块。预训练混合数据包括来自 7 种机器人配置、68 个任务的 10,000 小时灵巧操作数据，以及从 OXE [10]、DROID [23] 和 Bridge [52] 收集的大量已有机器人操作数据。据我们所知，这是用于机器人操作模型的最大预训练混合数据。微调实验包含超过 20 个任务，显示模型优于多种基线，包括先前的 VLA 模型 [24] 和专门用于灵巧操作的模型 [57, 9]。后训练配方还使模型能够完成非常复杂的任务，例如从任意初始配置折叠多件衣物或组装盒子。

我们的框架大体类似大型语言模型所采用的训练流程：先在从网络收集的大型数据集上预训练基础模型，再通过后训练“对齐”模型，使其能够遵循指令并执行用户命令。通常认为，这类模型的大部分“知识”在预训练阶段获得，而后训练阶段负责教会模型如何利用这些知识完成用户命令。我们的实验表明，机器人基础模型可能出现类似现象：预训练模型具备一些零样本能力，但洗衣折叠等复杂任务需要用高质量数据微调。只在高质量数据上训练会得到脆弱模型，无法可靠地从错误中恢复；而让预训练模型零样本执行，也不总能表现出后训练数据中展示的流畅策略。

我们希望这些结果能成为构建通用、广泛适用机器人基础模型的起点。实验表明，这类模型或许很快会成为现实，但仍有若干局限和大量未来工作空间。第一，我们还没有全面理解预训练数据集应如何组成：目前只是组合了所有可获得的数据，而哪些类型的数据最值得加入、应如何加权，仍是开放问题。评估中的任务并非都能可靠运行，我们仍不清楚如何预测获得近乎完美性能所需的数据量和数据类型。最后，高度多样化数据的组合，尤其是来自不同任务和不同机器人的数据，究竟能带来多少正迁移还有待研究。虽然结果表明通用预训练机器人基础模型可能成为现实，但未来仍需研究这种通用性是否能扩展到自动驾驶、导航和足式运动等更加不同的领域。

# 致谢

感谢 Laura Smith 和 Dibya Ghosh 对论文的反馈以及对图表和视频的帮助；感谢 Philip Clark、Kelly Sims 和 Saunaz Moradi 对写作的反馈；感谢 Evan Pokrandt、Joakim Keussen、Dan Philibin、Eitan Penner、Adam Lisagor 和 Greg Miller 对插图、设计和视频的帮助。感谢 Lili Yu 提供有益的技术讨论。我们非常感谢所有机器人操作员坚持不懈地收集机器人操作数据。完整贡献声明见附录 A。

## 图 7–13 图注补全

**图 7：开箱即用评估结果。** 我们评估训练完整 700k 步的 π0、训练 160k 步且与基线模型更新次数相同的版本、π0-small，以及三个基线：在全部数据上训练的 OpenVLA 和 Octo，以及只在 UR5e 任务上训练的 OpenVLA（我们发现它在 UR5e 任务上表现更好）。在所有任务和所有比较中，即使 π0 的“计算量对等”版本也优于所有基线；完整版本的结果以较大幅度达到最佳。

**图 8：语言评估中的任务。** 我们在 3 个不同的语言条件任务上评估模型，每项任务都要求遵循一系列中间语言命令。任务包括收拾餐桌（上图：把餐具放进箱子、把垃圾放进垃圾桶）、摆餐桌（中图：从箱子中取出物品）以及装袋购物品（下图）。

**图 9：语言评估。** 我们比较只接收总体任务命令（例如“把杂货装袋”）的“flat”策略，以及接收人类专家中间命令（“human”）或高级 VLM 策略中间命令（“HL”）的策略；同时在“expert”条件下比较 π0 与 π0-small 的语言遵循准确率。人类专家提供的中间语言命令带来显著提升，自主高级策略带来较小程度的提升。由于 π0-small 的语言遵循能力有限，加入高级专家总体上不会带来收益。

**图 10：微调评估任务。** 我们把模型适配到一系列与预训练任务不同的下游任务。任务从与预训练最相似的任务（堆碗、折叠毛巾），到引入新的未见元素的任务（微波炉），再到需要新动作和新物体类型的任务（Franka 抽屉物品、更换纸巾卷）。

**图 11：使用不同数据量进行微调。** π0 即使在较少数据下也能学习一些较容易的任务；预训练模型通常比从零训练的模型获得更大的提升。

**图 12：复杂任务的后训练结果。** 我们以 10 次试验的平均分数评估复杂任务。完整预训练的 π0 在所有任务上都达到最大分数的 50% 以上，并且通常优于消融版本；在最困难的任务上提升尤其显著。

**图 13：复杂任务的后训练结果。** 这组任务包括从固定（a）或移动（b）机器人旁的洗衣篮中折叠衣物、收拾真实午餐桌（c）、组装盒子（d）、把鸡蛋装进蛋盒（e）以及把食物装进外带盒（f）。这些任务要求组合数十种行为（抓取、堆叠、折叠和铺平），泛化到大量物体配置，并处理可变形物体和柔性纸板等复杂物理属性。

# 附录

## A. 贡献

作者按字母顺序在以下方面作出贡献：

- **数据与操作：** Noah Brown、Michael Equi、Chelsea Finn、Niccolo Fusai、Lachy Groom、Liyiming Ke、Suraj Nair、Lucy Shi、Anna Walling。
- **评估实验：** Kevin Black、Michael Equi、Chelsea Finn、Brian Ichter、Liyiming Ke、Adrian Li-Bell、Suraj Nair、Karl Pertsch、Lucy Shi。
- **模型设计：** Kevin Black、Brian Ichter、Sergey Levine、Karl Pertsch、Lucy Shi、Quan Vuong。
- **后训练：** Michael Equi、Chelsea Finn、Liyiming Ke、Adrian Li-Bell、Suraj Nair、Lucy Shi。
- **预训练：** Kevin Black、Danny Driess、Brian Ichter、Sergey Levine、Karl Pertsch、Lucy Shi、Quan Vuong。
- **机器人硬件：** Noah Brown、Adnan Esmail、Chelsea Finn、Tim Jones、Mohith Mothukuri。
- **机器人软件：** Karol Hausman、Szymon Jakubczak、Sergey Levine、James Tanner、Haohuan Wang。
- **训练基础设施：** Kevin Black、Michael Equi、Sergey Levine、Adrian Li-Bell、Suraj Nair、Quan Vuong、Haohuan Wang、Ury Zhilinsky。
- **写作与插图：** Kevin Black、Chelsea Finn、Lachy Groom、Karol Hausman、Brian Ichter、Sergey Levine、Quan Vuong。

## B. 模型架构细节

本节完整描述模型架构。我们遵循 PaliGemma VLM [5] 的设计，但作出以下改变：

1. 为机器人专用 token 增加输入和输出投影，包括状态向量 $q_t$ 和动作向量 $A_t=[a_t,\ldots,a_{t+H-1}]$；
2. 增加一个用于纳入流匹配时间步信息 $\tau$ 的 MLP；
3. 为动作专家使用第二组、规模更小的权重。

### 额外输入和输出

标准 PaliGemma 架构接收图像序列 $[I_t^1,\ldots,I_t^n]$，随后接收语言提示 $\ell_t$。我们加入机器人本体感知状态 $q_t$，并通过线性投影将其映射到 Transformer 的嵌入维度。最后一组输入 token 对应带噪动作块
$$
A_t^{\tau}=[a_t^{\tau},\ldots,a_{t+H-1}^{\tau}],
$$
token 数量等于动作时域长度（我们的任务中 $H=50$）。我们只使用与这 $H$ 个带噪动作对应的 Transformer 输出，并通过线性投影将其解码为 $v_\theta(A_t^{\tau},o_t)$。

### 纳入流匹配时间步

带噪动作块 $A_t^{\tau}$ 通过一个同时纳入流匹配时间步 $\tau$ 的 MLP 映射到 Transformer 嵌入维度。对于每个带噪动作 $a_{t'}^{\tau}$，输入 Transformer 的对应嵌入为

$$
W_3\,\mathrm{swish}\left(W_2\,\mathrm{concat}\left(W_1a_{t'}^{\tau},\phi(\tau)\right)\right),
$$

其中 $\phi:\mathbb{R}\to\mathbb{R}^{w}$ 是正弦位置编码函数 [51]，$W_1\in\mathbb{R}^{w\times d}$，$W_2\in\mathbb{R}^{w\times 2w}$，$W_3\in\mathbb{R}^{w\times w}$，$d$ 是动作维度，$w$ 是动作专家的嵌入维度（即宽度）。

### 注意力掩码

π0 使用包含 3 个块的分块因果注意力掩码：

$$
[I_t^1,\ldots,I_t^n,\ell_t],\quad[q_t],\quad[a_t^{\tau},\ldots,a_{t+H-1}^{\tau}].
$$

每个块内部使用完全双向注意力，但每个块中的 token 不能关注未来块中的 token。第一个块包含 PaliGemma VLM 预训练使用的输入模态；禁止它关注未来块（其中包含新增输入），以减小相对于预训练分布的偏移。机器人状态 $q_t$ 独立成块，因为它不会随每个流匹配积分步骤变化；禁止它关注最后一个块，可以在采样时缓存对应的 key 和 value。最后一个块对应带噪动作 $A_t^{\tau}$，它可以关注完整输入序列。

### 动作专家

π0 实现为一个具有两组权重（也称为专家 [45]）的单一 Transformer，每个 token 被路由到其中一个专家；两组权重只在 Transformer 的自注意力层中交互。图像和语言提示 $[I_t^1,\ldots,I_t^n,\ell_t]$ 路由到更大的 VLM 骨干，该骨干从 PaliGemma 初始化。VLM 预训练未见过的输入 $[q_t,A_t^{\tau}]$ 路由到动作专家。

PaliGemma 基于 Gemma 2B [49] 语言模型，使用多查询注意力 [44]，配置为

$$
\{\mathrm{width}=2048,\ \mathrm{depth}=18,\ \mathrm{mlp\ dim}=16{,}384,\ \mathrm{num\ heads}=18,\ \mathrm{num\ kv\ heads}=1,\ \mathrm{head\ dim}=256\}.
$$

由于两个专家只在自注意力层交互，它们的 width 和 MLP dim 不必相同。为了加快推理（动作专家需要多次前向传播），我们把动作专家缩小为

$$
\{\mathrm{width}=1024,\ \mathrm{mlp\ dim}=4096\},
$$

其参数量约为 3 亿。

### 流匹配时间步的采样

原始流匹配工作 [28, 32] 从均匀分布中采样时间步：

$$
\tau\sim U(0,1).
$$

Esser 等人 [14] 则提出从强调中间时间步的 logit-normal 分布中采样。他们认为，在高时间步（低噪声水平）时，模型只需学习恒等函数；在低时间步（高噪声水平）时，模型只需学习数据分布的均值。

然而，我们假设动作预测任务与高分辨率图像合成略有不同：在给定文本标签的条件下预测平均图像可能相对容易，但在给定机器人观测时预测平均动作（即学习 $\mathbb{E}[A_t\mid o_t]$）要困难得多。这是因为观测 $o_t$ 包含的信息非常丰富，它对可能动作分布的约束远强于文本标签对可能图像分布的约束。因此，我们设计了强调低时间步（高噪声水平）的时间步采样分布；此外，完全不采样高于阈值 $s$ 的时间步，因为只要积分步长 $\delta>1-s$，这些时间步就不是必需的。分布为

$$
p(\tau)=\mathrm{Beta}\left(\frac{s-\tau}{s};1.5,1\right),
$$

如图 14 所示。实验使用 $s=0.999$，因此允许 $\delta>1/1000$，即最多 1,000 个积分步骤。

**图 14：流匹配时间步采样分布。** 我们从一个经过平移的 beta 分布采样 $\tau$，该分布强调较低时间步（对应噪声更大的动作），并且完全不采样高于截断值 $s$ 的时间步。实验使用 $s=0.999$。

## C. 非 VLM 基线架构

基线架构 π0-small 不以 VLM 为骨干，因此我们用它评估 VLM 预训练的收益。我们将它设计得足够有表达力，以拟合大规模数据集，同时在从零训练时保持良好性能。该模型约有 4.7 亿参数，与主模型存在以下差异：

1. 由于模型没有语言模型骨干，我们使用 DistilBERT [42] 编码语言命令 $\ell_t$ 的语言 token；
2. 动作专家与观测编码器输出进行交叉注意力，类似传统编码器-解码器 Transformer [51]，而主模型更像仅解码器式混合专家 [45]；
3. 图像使用更小的预训练 ViT 编码器编码，具体是 Steiner 等人 [47] 的 R26-S-32 ResNet-ViT 混合模型；
4. ViT 图像编码器不共享权重；
5. 用于编码观测的 Transformer 骨干（位于 ViT 图像编码器之后）没有在互联网数据上预训练；
6. 动作专家使用 DiT 架构 [36] 而不是 Gemma 架构，因此通过 AdaLN-Zero 层纳入流匹配时间步 $\tau$。

除此之外，两个模型大致相似：都使用预训练 ViT 图像编码器，都为观测编码器和动作专家使用独立权重，都接收相同的观测格式，并且都执行 10 步流匹配来预测动作块。

## D. 推理

模型接收观测

$$
o_t=[I_t^1,\ldots,I_t^n,\ell_t,q_t]
$$

和带噪动作 $A_t^{\tau}$，输出需要积分才能得到下一流匹配步骤的向量场 $v_t^{\tau}$。每次预测新的动作块 $A_t$ 时，都必须编码图像 $I_t^1,\ldots,I_t^n$，对对应 $o_t$ 的 token 执行一次前向传播，然后执行 10 步流匹配；每一步都需要对 $A_t^{\tau}$ 对应的 token 执行前向传播，而 $o_t$ 对应的 key 和 value 会被缓存。

表 I 汇总了使用三张相机图像时这一操作的计算时间。操作在 NVIDIA GeForce RTX 4090 消费级 GPU 上计时。移动机器人通过 Wi-Fi 进行离机推理，因此增加少量网络延迟。进一步的优化、量化和其他改进可能继续减少推理时间。

由于模型一次生成完整的 (H) 步动作块，因此在再次推理前最多可以执行 (H) 个动作。不过，也可以更频繁地推理，并使用各种聚合策略组合不同推理调用产生的动作。我们早期尝试了时间集成 [57]，发现它会损害策略性能，因此选择不聚合动作，而是开环执行动作块。对于 20 Hz 的 UR5e 和 Franka 机器人，我们每 0.8 秒推理一次（执行 16 个动作后）；对于所有以 50 Hz 运行的其他机器人，我们每 0.5 秒推理一次（执行 25 个动作后）。

**表 I：NVIDIA GeForce RTX 4090 上模型的推理时间**

| 模型部分 | 推理时间 |
|---|---:|
| 图像编码器 | 14 ms |
| 观测前向传播 | 32 ms |
| 10 次动作前向传播（流匹配） | 27 ms |
| 网络延迟（离机时） | 13 ms |
| 板载推理总时间 | 73 ms |
| 离机推理总时间 | 86 ms |

## E. 评估细节

对于每个任务，我们设计一个衡量任务进度的评分规则，并用它计算定量结果。具体规则见下文。

### A. 评估基础模型

- **折叠衬衫：** 记录成功或失败。每次评估开始时把衬衫平铺在桌上；成功定义为折好袖子，并沿衬衫长度完成一次对折。评估包含 4 件小 T 恤和 1 件中号 T 恤。每件物品评估 2 次，每次最多 15,000 步，约 5 分钟。
- **简单收拾餐桌：** 满分 7 分，桌上有 7 件不同物体，每个正确分类的物体得 1 分。
- **困难收拾餐桌：** 满分 12 分，桌上有 12 件不同物体，每个正确分类的物体得 1 分。任务包含特别困难的设置，例如一根筷子放在垃圾上方。
- **装袋杂货：** 满分 7 分，每件杂货正确放进袋子得 1 分。
- **从烤面包机取出吐司：** 满分 4 分，每片吐司从烤面包机中取出得 1 分，放到盘子上再得 1 分。

### B. 语言指令遵循

策略按照是否成功重新放置每件物体以及是否遵循指令进行评分。

- **收拾餐桌：** 机器人必须遵循命令拿起正确物体，并把每件物体放入正确容器。一回合共有 12 件物体和约 30 条指令。
- **摆餐桌：** 机器人安排所有餐具和餐巾，并根据语言说明进行调整。一回合共有 7 件物体和约 20 条指令。
- **装袋杂货：** 机器人从咖啡豆袋、大麦袋、棉花糖袋、猫粮、意大利面、海藻袋和杏仁袋中拿起正确物品，再将它们装入纸袋。一回合共有 7 件物体和约 14 条指令。

### C. 学习新的灵巧任务

- **堆碗：** 满分 3 分。两个碗分别正确堆入更大的碗中各得 1 分，最终产品整齐得 1 分。
- **折叠毛巾：** 满分 3 分。第一次对折、第二次对折和最终整齐程度各得 1 分。
- **微波炉中的特百惠盒：** 满分 4 分。打开微波炉、拿起特百惠盒、将其放入微波炉、关闭微波炉各得 1 分。
- **更换纸巾卷：** 满分 4 分。抓住旧纸巾卷得 1 分，取下得 1 分；抓住新纸巾卷得 1 分，放到分配器上得 1 分。
- **抽屉中的物品：** 满分 5 分。打开抽屉得 1 分，从抽屉中取出并放入的 3 件物品各得 1 分，关上抽屉得 1 分。

### D. 掌握复杂多阶段任务

- **洗衣折叠：** 满分 4 分。评估包含 5 件物品：M、L、XL 三件衬衫，以及 28 码和 36 码两条短裤。每件物品试验两次；尚未评估的物品从洗衣篮中的随机皱缩状态开始，已评估物品从折叠衣物堆开始。把物品从篮中取出并放到桌上得 1 分；把衬衫或短裤铺平得 1 分；折叠得 1 分；如果是第一个评估物品，将其放到桌角得 1 分，否则叠到已有的折叠衣物堆上得 1 分。每次评估最多 15,000 步，约 5 分钟。
- **移动洗衣：** 采用与洗衣折叠相同的评估流程；三件衬衫为 M、M、XL，两条短裤为 32W 和 31W。
- **收拾餐桌：** 满分 12 分，桌上有 12 件不同物体，每个正确分类的物体得 1 分；同样包含筷子放在垃圾上等困难设置。
- **组装盒子：** 满分 5 分。成功拿起盒子、把盒子对折以便关闭折 flap、关闭右 flap、关闭左 flap、将最终产品整齐居中，各得 1 分。
- **装鸡蛋：** 满分 7 分。每个鸡蛋放进蛋盒正确槽位得 1 分，合上盒盖得 1 分。
- **装袋食物：** 满分 5 分。拿起食物盘得 1 分，把 3 件食物放入外带盒各得 1 分，合上外带盒得 1 分。
- **烘干机卸货：** 机器人带着洗衣篮接近烘干机并把衣物卸入篮中。满分 5 分：正确接近烘干机、把洗衣篮放到凳子上、打开烘干机、把所有衣物放进篮子、关闭烘干机各得 1 分。评估使用 3 件衬衫和 2 条短裤，初始状态随机放在烘干机内。


# 参考文献

以下书目信息按原文保留，以确保作者、题名、出版物、页码和 arXiv 编号完整；题名含义已在正文首次出现处翻译。

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.
- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35: 23716–23736, 2022.
- [4] Jorge Aldaco, Travis Armstrong, Robert Baruch, Jeff Bingham, Sanky Chan, Kenneth Draper, Debidatta Dwibedi, Chelsea Finn, Pete Florence, Spencer Goodrich, et al. Aloha 2: An enhanced low-cost hardware for bimanual teleoperation. arXiv preprint arXiv:2405.02292, 2024.
- [5] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.
- [6] Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, and Vikash Kumar. RoboAgent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 4788–4795. IEEE, 2024.
- [7] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.
- [8] Serkan Cabi, Sergio Gómez Colmenarejo, Alexander Novikov, Ksenia Konyushkova, Scott Reed, Rae Jeong, Konrad Zolna, Yusuf Aytar, David Budden, Mel Vecerik, et al. Scaling data-driven robotics with reward sketching and batch reinforcement learning. arXiv preprint arXiv:1909.12200, 2019.
- [9] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023.
- [10] OX-Embodiment Collaboration, A Padalkar, A Pooley, A Jain, A Bewley, A Herzog, A Irpan, A Khazatsky, A Rai, A Singh, et al. Open X-Embodiment: Robotic learning datasets and RT-X models. arXiv preprint arXiv:2310.08864, 1(2), 2023.
- [11] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palme: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.
- [12] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of language models with mixture-of-experts. In International Conference on Machine Learning, pages 5547–5569. PMLR, 2022.
- [13] Frederik Ebert, Yanlai Yang, Karl Schmeckpeper, Bernadette Bucher, Georgios Georgakis, Kostas Daniilidis, Chelsea Finn, and Sergey Levine. Bridge data: Boosting generalization of robotic skills with crossdomain datasets. arXiv preprint arXiv:2109.13396, 2021.
- [14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.
- [15] Haritheja Etukuru, Norihito Naka, Zijin Hu, Seungjae Lee, Julian Mehu, Aaron Edsinger, Chris Paxton, Soumith Chintala, Lerrel Pinto, and Nur Muhammad Mahi Shafiullah. Robot utility models: General policies for zero-shot deployment in new environments. arXiv preprint arXiv:2409.05865, 2024.
- [16] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.
- [17] Zipeng Fu, Tony Z. Zhao, and Chelsea Finn. Mobile aloha: Learning bimanual mobile manipulation with lowcost whole-body teleoperation. In Conference on Robot Learning (CoRL), 2024.
- [18] Abhinav Gupta, Adithyavairavan Murali, Dhiraj Prakashchand Gandhi, and Lerrel Pinto. Robot learning in homes: Improving generalization and reducing dataset bias. Advances in neural information processing systems, 31, 2018.
- [19] Wanggui He, Siming Fu, Mushui Liu, Xierui Wang, Wenyi Xiao, Fangxun Shu, Yi Wang, Lei Zhang, Zhelun Yu, Haoyuan Li, et al. Mars: Mixture of auto-regressive models for fine-grained text-to-image synthesis. arXiv preprint arXiv:2407.07614, 2024.
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [21] John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žı́dek, Anna Potapenko, et al. Highly accurate protein structure prediction with alphafold. Nature, 596(7873):583–589, 2021.
- [22] Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, et al. Scalable deep reinforcement learning for visionbased robotic manipulation. In Conference on robot learning, pages 651–673. PMLR, 2018.
- [23] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. DROID: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.
- [24] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.
- [25] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020.
- [26] Sergey Levine, Peter Pastor, Alex Krizhevsky, Julian Ibarz, and Deirdre Quillen. Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection. The International journal of robotics research, 37(4-5):421–436, 2018.
- [27] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, 2022.
- [28] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [29] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024.
- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.
- [31] Peiqi Liu, Yaswanth Orru, Jay Vakil, Chris Paxton, Nur Muhammad Mahi Shafiullah, and Lerrel Pinto. Okrobot: What really matters in integrating open-knowledge models for robotics. arXiv preprint arXiv:2401.12202, 2024.
- [32] Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022.
- [33] Ajay Mandlekar, Yuke Zhu, Animesh Garg, Jonathan Booher, Max Spero, Albert Tung, Julian Gao, John Emmons, Anchit Gupta, Emre Orbay, et al. RoboTurk: A crowdsourcing platform for robotic skill learning through imitation. In Conference on Robot Learning, pages 879– 893. PMLR, 2018.
- [34] Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox. MimicGen: A data generation system for scalable robot learning using human demonstrations. arXiv preprint arXiv:2310.17596, 2023.
- [35] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [36] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [37] Lerrel Pinto and Abhinav Gupta. Supersizing selfsupervision: Learning to grasp from 50k tries and 700 robot hours. In 2016 IEEE international conference on robotics and automation (ICRA), pages 3406–3413. IEEE, 2016.
- [38] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [41] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.
- [42] V Sanh. Distilbert, a distilled version of bert: Smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.
- [43] Nur Muhammad Mahi Shafiullah, Anant Rai, Haritheja Etukuru, Yiqian Liu, Ishan Misra, Soumith Chintala, and Lerrel Pinto. On bringing robots home. arXiv preprint arXiv:2311.16098, 2023.
- [44] Noam Shazeer. Fast transformer decoding: One writehead is all you need. arXiv preprint arXiv:1911.02150, 2019.
- [45] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.
- [46] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.
- [47] Andreas Steiner, Alexander Kolesnikov, Xiaohua Zhai, Ross Wightman, Jakob Uszkoreit, and Lucas Beyer. How to train your vit? data, augmentation, and regularization in vision transformers. arXiv preprint arXiv:2106.10270, 2021.
- [48] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean- Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [49] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.
- [50] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.
- [51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.
- [52] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. BridgeData v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723– 1736. PMLR, 2023.
- [53] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.
- [54] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022.
- [55] Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, and Jian Tang. Tinyvla: Towards fast, data-efficient vision-languageaction models for robotic manipulation. arXiv preprint arXiv:2409.12514, 2024.
- [56] Kuan-Ting Yu, Maria Bauza, Nima Fazeli, and Alberto Rodriguez. More than a million ways to be pushed. a high-fidelity experimental dataset of planar pushing. In 2016 IEEE/RSJ international conference on intelligent robots and systems (IROS), pages 30–37. IEEE, 2016.
- [57] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.
- [58] Tony Z Zhao, Jonathan Tompson, Danny Driess, Pete Florence, Kamyar Ghasemipour, Chelsea Finn, and Ayzaan Wahid. Aloha unleashed: A simple recipe for robot dexterity. arXiv preprint arXiv:2410.13126, 2024.
- [59] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.
- [60] Minjie Zhu, Yichen Zhu, Jinming Li, Junjie Wen, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, et al. Scaling diffusion policy in transformer to 1 billion parameters for robotic manipulation. arXiv preprint arXiv:2409.14411, 2024.

> **PDF 图形文字说明：** 原 PDF 的图 1 和图 3 内部流程框图文字使用了当前 PDF 文本层无法可靠解码的字体编码；正文和图题已完整翻译，框图中无法读取的局部标签保留为“图中文字无法从 PDF 文本层提取”，避免臆造标签。

<!-- paper:end -->
