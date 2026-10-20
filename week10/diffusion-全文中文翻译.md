# 面向成像与视觉的扩散模型教程

Stanley Chan<sup>1</sup>

2025 年 1 月 9 日

> 说明：本文按 PDF 全文翻译正文、定义、定理、例子、证明、图注与算法。公式按原编号写成 LaTeX。参考文献表保留英文原文。图本身不重画，只翻译图注。

## 摘要

近年来生成工具的惊人增长，推动了文本到图像生成和文本到视频生成中许多令人兴奋的应用。这些生成工具背后的基本原理是扩散这一概念，它是一种特定的采样机制，克服了先前方法中一些长期存在的缺陷。本教程的目标是讨论这些扩散模型背后的核心思想。本教程的目标读者包括有兴趣研究扩散模型、或把这些工具用于解决其他问题的本科生和研究生。

arXiv:2403.18103v3 [cs.LG]，2025 年 1 月 8 日

## 目录

1. 变分自编码器（Variational Auto-Encoder，VAE）
   - 1.1 VAE 的基本构件
   - 1.2 证据下界（Evidence Lower Bound）
   - 1.3 VAE 中的优化
   - 1.4 结语
2. 去噪扩散概率模型（Denoising Diffusion Probabilistic Model，DDPM）
   - 2.1 基本构件
   - 2.2 证据下界
   - 2.3 反向过程的分布
   - 2.4 训练与推理
   - 2.5 预测噪声
   - 2.6 去噪扩散隐式模型（Denoising Diffusion Implicit Model，DDIM）
   - 2.7 结语
3. 分数匹配朗之万动力学（Score-Matching Langevin Dynamics，SMLD）
   - 3.1 从分布中采样
   - 3.2 （Stein 的）分数函数
   - 3.3 分数匹配技术
   - 3.4 结语
4. 随机微分方程（Stochastic Differential Equation，SDE）
   - 4.1 从迭代算法到常微分方程
   - 4.2 什么是 SDE？
   - 4.3 用于 DDPM 与 SMLD 的随机微分方程
   - 4.4 ODE 与 SDE 的数值求解器
   - 4.5 结语
5. 朗之万方程与福克–普朗克方程
   - 5.1 布朗运动
   - 5.2 主方程
   - 5.3 Kramers–Moyal 展开
   - 5.4 福克–普朗克方程
   - 5.5 结语
6. 结论

<sup>1</sup> 普渡大学电气与计算机工程学院，West Lafayette, IN 47907。Email: stanchan@purdue.edu。

© 2024 Stanley Chan. All Rights Reserved.

---

## 1 变分自编码器（VAE）

很久以前，在一个遥远的星系里，我们想造一个生成器——一个能根据我们交给计算机的某些输入，生成文本、语音或图像的生成器。这起初听起来像魔法，但这个问题实际上已经被研究了很长时间。为了开启本教程的讨论，我们首先考虑变分自编码器（variational autoencoder，VAE）。VAE 由 Kingma 与 Welling 于 2014 年提出 [23]。根据他们 2019 年的教程 [24]，VAE 受到 Helmholtz 机 [10] 的启发，是图模型与深度学习的结合。接下来，我们将讨论 VAE 的问题设定、它的基本构件，以及与训练相关的优化工具。

### 1.1 VAE 的基本构件

我们从讨论 VAE 的示意图开始。如下图所示，VAE 由一对模型组成（通常由深度神经网络实现）。靠近输入的那个称为编码器（encoder），靠近输出的那个称为解码器（decoder）。我们把输入（通常是一幅图像）记为向量 \(x\)，把输出（通常是另一幅图像）记为向量 \(\hat{x}\)。位于编码器与解码器之间的向量称为潜变量（latent variable），记为 \(z\)。编码器的工作是为 \(x\) 提取一个有意义的表示，解码器的工作是从潜变量 \(z\) 生成一幅新图像。

**图 1.1：** 变分自编码器由一个把输入 \(x\) 转换成潜变量 \(z\) 的编码器，以及一个从该潜变量合成输出 \(\hat{x}\) 的解码器组成。

潜变量 \(z\) 在这一设定中有两个特殊作用。对输入而言，潜变量封装了可以用来描述 \(x\) 的信息。编码过程可以是有损的，但我们的目标是尽可能保留 \(x\) 的重要内容。对输出而言，潜变量充当可以生成图像 \(\hat{x}\) 的“种子”。两个不同的 \(z\) 在理论上应当给出两幅不同的生成图像。

下面给出潜变量稍正式一些的定义。

**定义 1.1. 潜变量 [24]。** 在概率模型中，潜变量 \(z\) 是我们观察不到、因而不是训练数据集一部分的变量，尽管它们是模型的一部分。

**例 1.1.** 得到一幅图像的潜表示并不是一件陌生的事。在 JPEG 压缩的时代（可以说它已经是恐龙了），我们用离散余弦变换（discrete cosine transform，DCT）基函数 \(\varphi_n\) 来编码图像或其图像块。系数向量 \(z = [z_1,\ldots,z_N]^\top\) 通过把图像 \(x\) 投影到基所张成的空间得到，即 \(z_n = \langle \varphi_n, x \rangle\)。因此，给定图像 \(x\)，我们可以产生系数向量 \(z\)。从 \(z\) 出发，我们可以用逆变换恢复（即解码）图像。

**图 1.2：** 在离散余弦变换（DCT）中，可以把编码器看成取一幅图像 \(x\)，并通过把 \(x\) 投影到基函数上生成潜变量 \(z\)。

在这个例子中，系数向量 \(z\) 就是潜变量。编码器是 DCT 变换，解码器是逆 DCT 变换。

VAE 中“变分”一词与变分法（calculus of variations）这一学科有关，该学科研究在函数上的优化。在 VAE 中，我们感兴趣的是寻找描述 \(x\) 和 \(z\) 的最优概率分布。有鉴于此，我们需要考虑几个分布：

- \(p(x)\)：\(x\) 的真实分布。它永远未知。扩散模型的整个世界，就是要找到从 \(p(x)\) 中抽取样本的方法。如果我们知道 \(p(x)\)（比如说，我们有一个描述 \(p(x)\) 的公式），就可以直接抽取一个使 \(\log p(x)\) 最大化的样本 \(x\)。
- \(p(z)\)：潜变量的分布。通常，我们把它取成零均值、单位方差的高斯 \(\mathcal{N}(0, I)\)。一个原因是高斯的线性变换仍是高斯，这使数据处理更容易。Doersch [12] 也有一个极好的解释。文中提到，任何分布都可以通过把一个高斯映射过一个足够复杂的函数来生成。例如，在单变量情形，逆累积分布函数（cumulative distribution function，CDF）技术 [7, 第 4 章] 可用于任何具有可逆 CDF 的连续分布。一般而言，只要我们有一个足够强的函数（例如神经网络），就可以学会它，并把 i.i.d. 高斯映射成我们的问题所需要的任何潜变量。
- \(p(z \mid x)\)：与编码器相关联的条件分布，它告诉我们给定 \(x\) 时 \(z\) 的似然。我们无法得到它。\(p(z \mid x)\) 本身不是编码器，但编码器必须做某些事，使它的行为与 \(p(z \mid x)\) 一致。
- \(p(x \mid z)\)：与解码器相关联的条件分布，它告诉我们给定 \(z\) 时得到 \(x\) 的后验概率。同样，我们无法得到它。

当我们从经典参数模型转向深度神经网络时，潜变量的概念变成了深度潜变量。Kingma 与 Welling [24] 给出了如下好的定义。

**定义 1.2. 深度潜变量 [24]。** 深度潜变量是其分布 \(p(z)\)、\(p(x \mid z)\) 或 \(p(z \mid x)\) 由神经网络参数化的潜变量。

深度潜变量的优点是，即使先验分布和条件分布的结构相对简单（例如高斯），它们也能建模非常复杂的数据分布 \(p(x)\)。一种理解方式是：神经网络可以用来估计高斯的均值。高斯本身虽然简单，但均值是输入数据的函数，数据经过神经网络后产生依赖于数据的均值。于是高斯的表达能力被显著提高。

让我们回到上面四个分布。下面是一个有些平凡、但有教育意义的例子，可以说明这一想法：

**例 1.2.** 考虑随机变量 \(X\) 服从高斯混合模型，潜变量 \(z \in \{1,\ldots,K\}\) 表示簇身份，使得 \(p_Z(k) = \mathbb{P}[Z = k] = \pi_k\)，\(k = 1,\ldots,K\)。我们假设 \(\sum_{k=1}^{K} \pi_k = 1\)。于是，如果我们被告知只需要看第 \(k\) 个簇，给定 \(Z\) 时 \(X\) 的条件分布是

$$
p_{X \mid Z}(x \mid k) = \mathcal{N}(x \mid \mu_k, \sigma_k^2 I).
$$

\(x\) 的边缘分布可以用全概率公式得到，从而有

$$
p_X(x) = \sum_{k=1}^{K} p_{X \mid Z}(x \mid k)\, p_Z(k) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \sigma_k^2 I). \tag{1.1}
$$

因此，如果我们从 \(p_X(x)\) 出发，编码器的设计问题就是构造一个神奇的编码器，使得对每个样本 \(x \sim p_X(x)\)，潜码是 \(z \in \{1,\ldots,K\}\)，且分布为 \(z \sim p_Z(k)\)。

为了说明编码器和解码器如何工作，我们假设均值和方差已知并且固定。否则我们将需要通过期望最大化（expectation-maximization，EM）算法估计均值和方差。这是可以做的，但繁琐的方程会破坏这个说明的教育目的。

**编码器：** 我们如何从 \(x\) 得到 \(z\)？这很容易，因为在编码器处我们知道 \(p_X(x)\) 和 \(p_Z(k)\)。想象你只有两类 \(z \in \{1, 2\}\)。实际上你只是在做一个二元决策：样本 \(x\) 应当属于哪里。做二元决策有很多方法。如果你喜欢最大后验决策规则，可以检查

$$
p_{Z \mid X}(1 \mid x) \underset{\text{类 } 2}{\overset{\text{类 } 1}{\gtrless}} p_{Z \mid X}(2 \mid x),
$$

这会返回一个简单的决策：你给我们 \(x\)，我们告诉你 \(z \in \{1, 2\}\)。

**解码器：** 在解码器一侧，如果给定潜码 \(z \in \{1,\ldots,K\}\)，神奇的解码器只需要返回一个从 \(p_{X \mid Z}(x \mid k) = \mathcal{N}(x \mid \mu_k, \sigma_k^2 I)\) 中抽取的样本 \(x\)。不同的 \(z\) 会给我们 \(K\) 个混合分量之一。如果我们有足够多的样本，总体分布将服从该高斯混合。

这个例子当然过于简化，因为真实世界的问题可能比均值和方差已知的高斯混合模型难得多。但我们意识到的一件事是：如果我们想找到神奇的编码器和解码器，就必须有办法找到两个条件分布 \(p(z \mid x)\) 和 \(p(x \mid z)\)。然而它们都是高维的。

为了能说出更有意义的话，我们需要施加额外结构，以便把这一概念推广到更难的问题。为此，我们考虑下面两个代理分布：

- \(q_\phi(z \mid x)\)：\(p(z \mid x)\) 的代理，也是与编码器相关联的分布。\(q_\phi(z \mid x)\) 可以是任何有向图模型，并且可以用深度神经网络参数化 [24, 第 2.1 节]。例如，我们可以定义

$$
(\mu, \sigma^2) = \mathrm{EncoderNetwork}_\phi(x), \qquad q_\phi(z \mid x) = \mathcal{N}\bigl(z \mid \mu, \mathrm{diag}(\sigma^2)\bigr). \tag{1.2}
$$

  这个模型因其可处理性和计算效率而被广泛使用。

- \(p_\theta(x \mid z)\)：\(p(x \mid z)\) 的代理，也是与解码器相关联的分布。和解码器一样，解码器可以由深度神经网络参数化。例如，我们可以定义

$$
f_\theta(z) = \mathrm{DecoderNetwork}_\theta(z), \qquad p_\theta(x \mid z) = \mathcal{N}\bigl(x \mid f_\theta(z), \sigma_{\mathrm{dec}}^2 I\bigr), \tag{1.3}
$$

  其中 \(\sigma_{\mathrm{dec}}\) 是可以预先确定、也可以学习的超参数。

输入 \(x\) 与潜变量 \(z\) 之间的关系，以及条件分布，总结在图 1.3 中。有两个节点 \(x\) 和 \(z\)。“正向”关系由 \(p(z \mid x)\) 指定（并由 \(q_\phi(z \mid x)\) 近似），“反向”关系由 \(p(x \mid z)\) 指定（并由 \(p_\theta(x \mid z)\) 近似）。

**图 1.3：** 在变分自编码器中，变量 \(x\) 和 \(z\) 由条件分布 \(p(x \mid z)\) 和 \(p(z \mid x)\) 连接。为了使事情可行，我们引入代理分布 \(p_\theta(x \mid z)\) 和 \(q_\phi(z \mid x)\)。

**例 1.3.** 设随机变量 \(x \in \mathbb{R}^d\)，潜变量 \(z \in \mathbb{R}^d\)，使得

$$
x \sim p(x) = \mathcal{N}(x \mid \mu, \sigma^2 I), \qquad z \sim p(z) = \mathcal{N}(z \mid 0, I).
$$

我们想构造一个 VAE。这意味着我们想建立两个映射 \(\mathrm{Encoder}(\cdot)\) 和 \(\mathrm{Decoder}(\cdot)\)。编码器取样本 \(x\) 并把它映射到潜变量 \(z\)，解码器取潜变量 \(z\) 并把它映射到生成变量 \(\hat{x}\)。如果我们知道 \(p(x)\) 是什么，那么存在一个平凡解：\(z = (x - \mu)/\sigma\)，\(\hat{x} = \mu + \sigma z\)。在这种情况下，真实分布可以被确定，并且可以用狄拉克函数表示：

$$
p(x \mid z) = \delta\bigl(x - (\sigma z + \mu)\bigr), \qquad p(z \mid x) = \delta\bigl(z - (x - \mu)/\sigma\bigr).
$$

现在假设我们不知道 \(p(x)\)，因此需要建立一个编码器和解码器来估计 \(z\) 和 \(\hat{x}\)。先定义编码器。本例中的编码器取输入 \(x\)，并生成一对参数 \(\hat{\mu}(x)\) 和 \(\hat{\sigma}(x)^2\)，表示一个高斯的参数。然后我们把 \(q_\phi(z \mid x)\) 定义为高斯：

$$
\bigl(\hat{\mu}(x), \hat{\sigma}(x)^2\bigr) = \mathrm{Encoder}_\phi(x), \qquad q_\phi(z \mid x) = \mathcal{N}\bigl(z \mid \hat{\mu}(x), \hat{\sigma}(x)^2 I\bigr).
$$

为讨论方便，我们假设 \(\hat{\mu}\) 是 \(x\) 的仿射函数，即 \(\hat{\mu}(x) = ax + b\)，其中 \(a\)、\(b\) 为某些参数。类似地，假设 \(\hat{\sigma}(x)^2 = t^2\)，其中 \(t\) 为某个标量。这将给出

$$
q_\phi(z \mid x) = \mathcal{N}(z \mid ax + b, t^2 I).
$$

对解码器，我们采用类似结构：

$$
\bigl(\tilde{\mu}(z), \tilde{\sigma}(z)^2\bigr) = \mathrm{Decoder}_\theta(z), \qquad p_\theta(x \mid z) = \mathcal{N}\bigl(x \mid \tilde{\mu}(z), \tilde{\sigma}(z)^2 I\bigr).
$$

同样为讨论方便，我们假设 \(\tilde{\mu}\) 是仿射的，即 \(\tilde{\mu}(z) = cz + v\)，其中 \(c\)、\(v\) 为某些参数，并且 \(\tilde{\sigma}(z)^2 = s^2\)，其中 \(s\) 为某个标量。因此 \(p_\theta(x \mid z)\) 取如下形式：

$$
p_\theta(x \mid z) = \mathcal{N}(x \mid cz + v, s^2 I).
$$

参数如何确定，我们稍后讨论。

### 1.2 证据下界

我们如何用这两个代理分布来达到确定编码器和解码器的目标？如果把 \(\phi\) 和 \(\theta\) 当作优化变量，那么我们需要一个目标函数（或损失函数），以便通过训练样本优化 \(\phi\) 和 \(\theta\)。这里使用的损失函数称为证据下界（Evidence Lower BOund，ELBO）[24]：

**定义 1.3.（证据下界）** 证据下界定义为

$$
\mathrm{ELBO}(x) \stackrel{\mathrm{def}}{=} \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p(x, z)}{q_\phi(z \mid x)}\right]. \tag{1.4}
$$

你当然会困惑：世上的人究竟是怎么想出这个损失函数的？让我们看看 ELBO 是什么意思，以及它是如何推导出来的。

简而言之，ELBO 是先验分布 \(\log p(x)\) 的一个下界，因为我们可以证明

$$
\begin{aligned}
\log p(x)
&= \text{若干即将推导的神奇步骤} \\
&= \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p(x, z)}{q_\phi(z \mid x)}\right] + D_{\mathrm{KL}}\bigl(q_\phi(z \mid x) \,\|\, p(z \mid x)\bigr) \\
&\ge \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p(x, z)}{q_\phi(z \mid x)}\right] \\
&\stackrel{\mathrm{def}}{=} \mathrm{ELBO}(x),
\end{aligned}
\tag{1.5}
$$

其中不等式来自 KL 散度永远非负这一事实。因此 ELBO 是 \(\log p(x)\) 的一个有效下界。既然我们永远无法得到 \(\log p(x)\)，如果我们能够得到 ELBO，并且 ELBO 是一个好的下界，那么我们就可以通过最大化 ELBO 来有效地达到最大化 \(\log p(x)\) 这一黄金标准。现在的问题是这个下界有多好。从方程以及图 1.4 可以看到，当代理 \(q_\phi(z \mid x)\) 能够精确匹配真实分布 \(p(z \mid x)\) 时，不等式变成等式。因此，这场游戏的一部分是确保 \(q_\phi(z \mid x)\) 接近 \(p(z \mid x)\)。

**图 1.4：** \(\log p(x)\) 与 ELBO 的可视化。二者之间的间隙由 KL 散度 \(D_{\mathrm{KL}}(q_\phi(z \mid x) \,\|\, p(z \mid x))\) 决定。

式 (1.5) 的推导如下。

**定理 1.1. 对数似然的分解。** 对数似然 \(\log p(x)\) 可以分解为

$$
\log p(x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p(x, z)}{q_\phi(z \mid x)}\right]}_{\stackrel{\mathrm{def}}{=} \mathrm{ELBO}(x)} + D_{\mathrm{KL}}\bigl(q_\phi(z \mid x) \,\|\, p(z \mid x)\bigr). \tag{1.6}
$$

**证明.** 技巧是用我们神奇的代理 \(q_\phi(z \mid x)\) 去拨弄 \(p(x)\)，并推导出这个界。

$$
\begin{aligned}
\log p(x)
&= \log p(x) \times \underbrace{\int q_\phi(z \mid x)\, dz}_{=1}
&& \text{（乘以 1）} \\
&= \int \underbrace{\log p(x)}_{\text{关于 } z \text{ 的某个常数}} \times \underbrace{q_\phi(z \mid x)}_{z\text{ 上的分布}}\, dz
&& \text{（把 }\log p(x)\text{ 移进积分）} \\
&= \mathbb{E}_{q_\phi(z \mid x)}[\log p(x)],
\end{aligned}
\tag{1.7}
$$

其中最后一个等式来自：对任何随机变量 \(Z\) 和标量 \(a\)，有 \(\int a \cdot p_Z(z)\, dz = \mathbb{E}[a] = a\)。

看，我们已经得到了 \(\mathbb{E}_{q_\phi(z \mid x)}[\cdot]\)。只需再几步。使用贝叶斯定理，它指出 \(p(x, z) = p(z \mid x)\, p(x)\)：

$$
\begin{aligned}
\mathbb{E}_{q_\phi(z \mid x)}[\log p(x)]
&= \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p(x, z)}{p(z \mid x)}\right]
&& \text{（贝叶斯定理）} \\
&= \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p(x, z)}{p(z \mid x)} \times \frac{q_\phi(z \mid x)}{q_\phi(z \mid x)}\right]
&& \text{（乘以 }q_\phi(z \mid x)/q_\phi(z \mid x)\text{）} \\
&= \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p(x, z)}{q_\phi(z \mid x)}\right]}_{\mathrm{ELBO}}
+ \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{q_\phi(z \mid x)}{p(z \mid x)}\right]}_{D_{\mathrm{KL}}(q_\phi(z \mid x)\,\|\, p(z \mid x))},
\end{aligned}
\tag{1.8}
$$

其中我们认出第一项正好是 ELBO，第二项正好是 KL 散度。把式 (1.8) 与式 (1.5) 比较，证明完成。

**例 1.4.** 沿用前面的例子，如果我们知道 \(p(z \mid x)\)，就可以最小化 \(\log p(x)\) 与 \(\mathrm{ELBO}(x)\) 之间的间隙。为此注意到

$$
\log p(x) = \mathrm{ELBO}(x) + D_{\mathrm{KL}}\bigl(q_\phi(z \mid x)\,\|\, p(z \mid x)\bigr) \ge \mathrm{ELBO}(x).
$$

等式成立当且仅当 KL 散度项为零。要使 KL 散度为零，必须有 \(q_\phi(z \mid x) = p(z \mid x)\)。然而，由于 \(p(z \mid x)\) 是狄拉克函数，唯一的可能是

$$
q_\phi(z \mid x) = \mathcal{N}\!\left(z \,\middle|\, \frac{x - \mu}{\sigma},\, 0\right) = \delta\!\left(z - \frac{x - \mu}{\sigma}\right), \tag{1.9}
$$

也就是把标准差设为 \(t = 0\)。要确定 \(p_\theta(x \mid z)\)，我们还需要一些额外步骤来简化 ELBO。

我们现在有了 ELBO。但这个 ELBO 仍然不太有用，因为它涉及 \(p(x, z)\)，而我们无法得到它。所以还需要再做一点工作。

**定理 1.2. ELBO 的解释。** ELBO 可以分解为

$$
\mathrm{ELBO}(x)
= \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\bigl[\log p_\theta(x \mid z)\bigr]}_{\text{你的解码器有多好}}
- \underbrace{D_{\mathrm{KL}}\bigl(q_\phi(z \mid x)\,\|\, p(z)\bigr)}_{\text{你的编码器有多好}}. \tag{1.10}
$$

其中三项所涉及的分布都可以是高斯。

**证明.** 让我们更仔细地看 ELBO：

$$
\begin{aligned}
\mathrm{ELBO}(x)
&\stackrel{\mathrm{def}}{=} \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p(x, z)}{q_\phi(z \mid x)}\right]
&& \text{（定义）} \\
&= \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p(x \mid z)\, p(z)}{q_\phi(z \mid x)}\right]
&& \text{（}p(x, z) = p(x \mid z)\, p(z)\text{）} \\
&= \mathbb{E}_{q_\phi(z \mid x)}[\log p(x \mid z)] + \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p(z)}{q_\phi(z \mid x)}\right]
&& \text{（拆开期望）} \\
&= \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{\mathrm{KL}}\bigl(q_\phi(z \mid x)\,\|\, p(z)\bigr),
&& \text{（KL 的定义）}
\end{aligned}
$$

其中我们用其代理 \(p_\theta(x \mid z)\) 替换了无法得到的 \(p(x \mid z)\)。

这是一个漂亮的结果。我们刚刚证明了非常容易理解的东西。看式 (1.10) 中的两项：

- **重建（Reconstruction）。** 第一项关于解码器。我们希望解码器在喂入潜变量 \(z\) 时产生一幅好图像 \(x\)（当然如此）。所以我们想最大化 \(\log p_\theta(x \mid z)\)。这类似于最大似然，我们想找到模型参数以最大化观察到该图像的似然。这里的期望是对样本 \(z\)（以 \(x\) 为条件）取的。这不应令人惊讶，因为样本 \(z\) 被用来评估解码器的质量。它不能是任意噪声向量，而必须是有意义的潜向量。因此 \(z\) 需要从 \(q_\phi(z \mid x)\) 中采样。
- **先验匹配（Prior Matching）。** 第二项是编码器的 KL 散度。我们希望编码器把 \(x\) 变成潜向量 \(z\)，使该潜向量服从我们选择的分布，例如 \(z \sim \mathcal{N}(0, I)\)。稍一般地，我们把 \(p(z)\) 写成目标分布。因为 KL 散度是一种距离（两个分布越不相似，它越大），我们需要在前面放一个负号，使两个分布越相似时该项越大。

**例 1.5.** 承接上例，我们继续假设已知 \(p(z \mid x)\)。于是 ELBO 中的重建项给出

$$
\begin{aligned}
\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
&= \mathbb{E}_{q_\phi(z \mid x)}\bigl[\log \mathcal{N}(x \mid cz + v, s^2 I)\bigr] \\
&= \mathbb{E}_{q_\phi(z \mid x)}\left[-\frac{1}{2}\log 2\pi - \log s - \frac{\lVert x - (cz + v)\rVert^2}{2s^2}\right] \\
&= -\frac{1}{2}\log 2\pi - \log s - \frac{c^2}{2s^2}\, \mathbb{E}_{q_\phi(z \mid x)}\left\lVert z - \frac{x - v}{c}\right\rVert^2 \\
&= -\frac{1}{2}\log 2\pi - \log s - \frac{c^2}{2s^2}\, \mathbb{E}_{\delta\!\left(z - \frac{x-\mu}{\sigma}\right)}\left\lVert z - \frac{x - v}{c}\right\rVert^2 \\
&= -\frac{1}{2}\log 2\pi - \log s - \frac{c^2}{2s^2}\left\lVert \frac{x - \mu}{\sigma} - \frac{x - v}{c}\right\rVert^2 \\
&\le -\frac{1}{2}\log 2\pi - \log s,
\end{aligned}
$$

其中上界紧当且仅当范数平方项为零，而这在 \(v = \mu\) 且 \(c = \sigma\) 时成立。对其余项，显然 \(-\log s\) 是 \(s\) 的单调递减函数，并且当 \(s \to 0\) 时 \(-\log s \to \infty\)。因此，当 \(v = \mu\) 且 \(c = \sigma\) 时，\(\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]\) 在 \(s = 0\) 时被最大化。这意味着

$$
p_\theta(x \mid z) = \mathcal{N}(x \mid \sigma z + \mu,\, 0) = \delta\bigl(x - (\sigma z + \mu)\bigr). \tag{1.11}
$$

**ELBO 的局限。** ELBO 在实践中有用，但它并不等于真实似然 \(\log p(x)\)。如前所述，ELBO 恰好等于 \(\log p(x)\) 当且仅当 \(D_{\mathrm{KL}}(q_\phi(z \mid x)\,\|\, p(z \mid x)) = 0\)，而这发生在 \(q_\phi(z \mid x) = p(z \mid x)\) 时。下面的例子将展示一种情形：由最大化 ELBO 得到的 \(q_\phi(z \mid x)\) 并不等于 \(p(z \mid x)\)。

**例 1.6.（ELBO 的局限）。** 在上例中，如果我们对 \(p(z \mid x)\) 一无所知，就需要通过最大化 ELBO 来训练 VAE。然而，由于 ELBO 只是真实分布 \(\log p(x)\) 的下界，最大化 ELBO 不会像我们希望的那样返回狄拉克函数。相反，我们会得到某种相当有意义、但并不恰好是狄拉克函数的东西。

为简单起见，考虑那些会返回均值的无偏估计、但方差未知的分布：

$$
q_\phi(z \mid x) = \mathcal{N}\!\left(z \,\middle|\, \frac{x - \mu}{\sigma},\, t^2 I\right), \qquad p_\theta(x \mid z) = \mathcal{N}(x \mid \sigma z + \mu,\, s^2 I).
$$

这在一定程度上是“作弊”，因为理论上我们不应对均值的估计作任何假设。但从直觉角度看，既然 \(q_\phi(z \mid x)\) 和 \(p_\theta(x \mid z)\) 是 \(p(z \mid x)\) 和 \(p(x \mid z)\) 的代理，它们必须类似于狄拉克函数的某些性质。最接近的选择是把 \(q_\phi(z \mid x)\) 和 \(p_\theta(x \mid z)\) 定义为均值与两个狄拉克函数一致的高斯。方差未知，它们是本例感兴趣的对象。

我们这里的重点是最大化由先验匹配项和重建项组成的 ELBO。对先验匹配误差，我们想最小化 KL 散度：

$$
D_{\mathrm{KL}}\bigl(q_\phi(z \mid x)\,\|\, p(z)\bigr)
= D_{\mathrm{KL}}\!\left(\mathcal{N}\!\left(z \,\middle|\, \tfrac{x-\mu}{\sigma},\, t^2 I\right) \,\middle\|\, \mathcal{N}(z \mid 0, I)\right).
$$

两个多元高斯 \(\mathcal{N}(z \mid \mu_0, \Sigma_0)\) 与 \(\mathcal{N}(z \mid \mu_1, \Sigma_1)\) 的 KL 散度有闭式，可见于 Wikipedia：

$$
\begin{aligned}
&D_{\mathrm{KL}}\bigl(\mathcal{N}(\mu_0, \Sigma_0)\,\|\, \mathcal{N}(\mu_1, \Sigma_1)\bigr) \\
&= \frac{1}{2}\left(\mathrm{Tr}(\Sigma_1^{-1}\Sigma_0) - d + (\mu_1 - \mu_0)^\top \Sigma_1^{-1}(\mu_1 - \mu_0) + \log \frac{\det \Sigma_1}{\det \Sigma_0}\right).
\end{aligned}
$$

利用这一结果（并做一些代数），我们可以证明

$$
D_{\mathrm{KL}}\!\left(\mathcal{N}\!\left(z \,\middle|\, \tfrac{x-\mu}{\sigma},\, t^2 I\right) \,\middle\|\, \mathcal{N}(z \mid 0, I)\right)
= \frac{1}{2}\left[t^2 d - d + \left\lVert \frac{x - \mu}{\sigma}\right\rVert^2 - 2d \log t\right],
$$

其中 \(d\) 是 \(x\) 和 \(z\) 的维数。为最小化 KL 散度，我们对 \(t\) 求导并证明

$$
\frac{\partial}{\partial t}\frac{1}{2}\left[t^2 d - d + \left\lVert \frac{x-\mu}{\sigma}\right\rVert^2 - 2d\log t\right] = t \cdot d - \frac{d}{t}.
$$

把它设为零给出 \(t = 1\)。因此我们可以证明

$$
q_\phi(z \mid x) = \mathcal{N}\!\left(z \,\middle|\, \frac{x - \mu}{\sigma},\, I\right).
$$

对重建项，我们可以证明

$$
\begin{aligned}
\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
&= \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{1}{(\sqrt{2\pi s^2})^d}\exp\left(-\frac{\lVert x - (\sigma z + \mu)\rVert^2}{2s^2}\right)\right] \\
&= \mathbb{E}_{q_\phi(z \mid x)}\left[-\frac{d}{2}\log 2\pi - d\log s - \frac{\lVert x - (\sigma z + \mu)\rVert^2}{2s^2}\right] \\
&= -\frac{d}{2}\log 2\pi - d\log s - \frac{\sigma^2}{2s^2}\, \mathbb{E}_{q_\phi(z \mid x)}\left\lVert z - \frac{x - \mu}{\sigma}\right\rVert^2 \\
&= -\frac{d}{2}\log 2\pi - d\log s - \frac{\sigma^2}{2s^2}\, \mathrm{Trace}\,\mathbb{E}_{q_\phi(z \mid x)}\left[\left(z - \frac{x-\mu}{\sigma}\right)\left(z - \frac{x-\mu}{\sigma}\right)^\top\right] \\
&= -\frac{d}{2}\log 2\pi - d\log s - \frac{\sigma^2}{2s^2}\cdot d,
\end{aligned}
$$

因为 \(z \sim q_\phi(z \mid x)\) 的协方差是 \(I\)，所以迹给出 \(d\)。对 \(s\) 求导给出

$$
\frac{d}{ds}\left(-\frac{d}{2}\log 2\pi - d\log s - \frac{d\sigma^2}{2s^2}\right) = -\frac{d}{s} + \frac{d\sigma^2}{s^3} = 0.
$$

令其为零给出 \(s = \sigma\)。因此

$$
p_\theta(x \mid z) = \mathcal{N}(x \mid \sigma z + \mu,\, \sigma^2 I).
$$

从这个例子和上一个例子可以看到，尽管理想分布是狄拉克函数，我们得到的代理分布具有有限方差。这个有限方差给 VAE 生成的样本增加了额外随机性。这个 VAE 没有错——我们通过最大化 ELBO 正确地做了这件事。只是最大化 ELBO 并不等同于最大化 \(\log p(x)\)。

### 1.3 VAE 中的优化

前两小节我们介绍了 VAE 的基本构件和 ELBO。本小节的目标是讨论如何训练 VAE 以及如何做推理。

VAE 是一个旨在近似真实分布 \(p(x)\)、从而使我们能够抽取样本的模型。VAE 由 \((\phi, \theta)\) 参数化。因此，训练 VAE 等价于求解一个优化问题，它抓住 \(p(x)\) 的本质，同时又是可处理的。然而，由于 \(p(x)\) 不可得，自然的替代是优化作为 \(\log p(x)\) 下界的 ELBO。这意味着 VAE 的学习目标是求解下面的问题。

**定义 1.4.** VAE 的优化目标是最大化 ELBO：

$$
(\phi, \theta) = \operatorname*{argmax}_{\phi, \theta} \sum_{x \in \mathcal{X}} \mathrm{ELBO}(x), \tag{1.12}
$$

其中 \(\mathcal{X} = \{x^{(\ell)} \mid \ell = 1,\ldots,L\}\) 是训练数据集。

**ELBO 梯度的不可处理性。** 上述优化的挑战在于，ELBO 关于 \((\phi, \theta)\) 的梯度是不可处理的。由于当今大多数神经网络优化器使用一阶方法，并通过反向传播梯度来更新网络权重，不可处理的梯度会给训练 VAE 带来困难。

让我们更详细地说明梯度的不可处理性。我们首先把定义 1.3 代入上述目标函数。ELBO 的梯度是：<sup>2</sup>

$$
\begin{aligned}
\nabla_{\theta, \phi}\,\mathrm{ELBO}(x)
&= \nabla_{\theta, \phi}\, \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right] \\
&= \nabla_{\theta, \phi}\, \mathbb{E}_{q_\phi(z \mid x)}\bigl[\log p_\theta(x, z) - \log q_\phi(z \mid x)\bigr].
\end{aligned}
\tag{1.13}
$$

梯度包含两个参数。先看 \(\theta\)。我们可以证明

$$
\begin{aligned}
\nabla_\theta \,\mathrm{ELBO}(x)
&= \nabla_\theta\, \mathbb{E}_{q_\phi(z \mid x)}\bigl[\log p_\theta(x, z) - \log q_\phi(z \mid x)\bigr] \\
&= \nabla_\theta \int \bigl[\log p_\theta(x, z) - \log q_\phi(z \mid x)\bigr] \cdot q_\phi(z \mid x)\, dz \\
&= \int \nabla_\theta\bigl\{\log p_\theta(x, z) - \log q_\phi(z \mid x)\bigr\} \cdot q_\phi(z \mid x)\, dz \\
&= \mathbb{E}_{q_\phi(z \mid x)}\bigl[\nabla_\theta \log p_\theta(x, z)\bigr] \\
&\approx \frac{1}{L}\sum_{\ell=1}^{L} \nabla_\theta \log p_\theta\bigl(x, z^{(\ell)}\bigr),
\qquad \text{其中 } z^{(\ell)} \sim q_\phi(z \mid x),
\end{aligned}
\tag{1.14}
$$

其中最后一个等式是期望的蒙特卡洛近似。

在上述方程中，如果 \(p_\theta(x, z)\) 由一个可计算模型（例如神经网络）实现，那么其梯度 \(\nabla_\theta\{\log p_\theta(x, z)\}\) 可以通过自动微分计算。因此，最大化可以通过反向传播梯度来实现。

<sup>2</sup> ELBO 的原始定义使用真实联合分布 \(p(x, z)\)。实践中，由于 \(p(x, z)\) 不可得，我们用其代理 \(p_\theta(x, z)\) 替换它，后者是一个可计算的分布。

关于 \(\phi\) 的梯度更困难。我们可以证明

$$
\begin{aligned}
\nabla_\phi \,\mathrm{ELBO}(x)
&= \nabla_\phi\, \mathbb{E}_{q_\phi(z \mid x)}\bigl[\log p_\theta(x, z) - \log q_\phi(z \mid x)\bigr] \\
&= \nabla_\phi \int \bigl[\log p_\theta(x, z) - \log q_\phi(z \mid x)\bigr] \cdot q_\phi(z \mid x)\, dz \\
&= \int \nabla_\phi\bigl[\log p_\theta(x, z) - \log q_\phi(z \mid x)\bigr] \cdot q_\phi(z \mid x)\, dz \\
&\neq \int \nabla_\phi\bigl\{\log p_\theta(x, z) - \log q_\phi(z \mid x)\bigr\} \cdot q_\phi(z \mid x)\, dz \\
&= \mathbb{E}_{q_\phi(z \mid x)}\bigl[\nabla_\phi\bigl(-\log q_\phi(z \mid x)\bigr)\bigr] \\
&\approx \frac{1}{L}\sum_{\ell=1}^{L} \nabla_\phi\bigl(-\log q_\phi(z^{(\ell)} \mid x)\bigr),
\qquad \text{其中 } z^{(\ell)} \sim q_\phi(z \mid x).
\end{aligned}
\tag{1.15}
$$

可以看到，尽管我们希望保持与 \(\theta\) 类似的结构，上述推导中的期望算子与梯度算子不能交换。这使我们无法通过反向传播梯度来最大化 ELBO。

**重参数化技巧。** ELBO 梯度的不可处理性来自这样一个事实：我们需要从分布 \(q_\phi(z \mid x)\) 中抽取样本 \(z\)，而该分布本身是 \(\phi\) 的函数。如 Kingma 与 Welling [23] 所指出的，对连续潜变量，可以计算 \(\nabla_{\theta,\phi}\,\mathrm{ELBO}(x)\) 的无偏估计，从而近似计算梯度并最大化 ELBO。想法是采用称为重参数化技巧（reparameterization trick）的技术 [23]。

回忆潜变量 \(z\) 是从分布 \(q_\phi(z \mid x)\) 中抽取的样本。重参数化技巧的想法是把 \(z\) 表示为另一个随机变量 \(\epsilon\) 的某个可微且可逆的变换，而 \(\epsilon\) 的分布独立于 \(x\) 和 \(\phi\)。也就是说，我们定义一个可微且可逆的函数 \(g\)，使得

$$
z = g(\epsilon, \phi, x), \tag{1.16}
$$

其中 \(\epsilon \sim p(\epsilon)\)。为使讨论更容易，我们再加一个要求

$$
q_\phi(z \mid x) \cdot \det\left(\frac{\partial z}{\partial \epsilon}\right) = p(\epsilon), \tag{1.17}
$$

其中 \(\partial z / \partial \epsilon\) 是雅可比，\(\det(\cdot)\) 是矩阵行列式。这一要求与多元微积分中的变量替换有关。下面的例子会把它说清楚。

**例 1.7.** 设 \(z \sim q_\phi(z \mid x) \stackrel{\mathrm{def}}{=} \mathcal{N}(z \mid \mu, \mathrm{diag}(\sigma^2))\)。我们可以定义

$$
z = g(\epsilon, \phi, x) \stackrel{\mathrm{def}}{=} \epsilon \odot \sigma + \mu, \tag{1.18}
$$

其中 \(\epsilon \sim \mathcal{N}(0, I)\)，“\(\odot\)”表示逐元素乘法。参数 \(\phi\) 是 \(\phi = (\mu, \sigma^2)\)。对这一分布选择，令 \(\epsilon = (z - \mu)/\sigma\)，我们可以证明

$$
q_\phi(z \mid x)\cdot \det\left(\frac{\partial z}{\partial \epsilon}\right)
= \prod_{i=1}^{d} \frac{1}{\sqrt{2\pi \sigma_i^2}}\exp\left(-\frac{(z_i - \mu_i)^2}{2\sigma_i^2}\right)\cdot \prod_{i=1}^{d}\sigma_i
= \frac{1}{(\sqrt{2\pi})^d}\exp\left(-\frac{\lVert \epsilon \rVert^2}{2}\right)
= \mathcal{N}(0, I) = p(\epsilon).
$$

用 \(\epsilon\) 对 \(z\) 做了这一重参数化之后，我们可以看某个一般函数 \(f(z)\) 的 \(\nabla_\phi \mathbb{E}_{q_\phi(z \mid x)}[f(z)]\)。（稍后我们将考虑 \(f(z) = -\log q_\phi(z \mid x)\)。）为记号简单，我们把 \(g(\epsilon, \phi, x)\) 写成 \(g(\epsilon)\)，尽管我们知道 \(g\) 有三个输入。通过变量替换，我们可以证明

$$
\begin{aligned}
\mathbb{E}_{q_\phi(z \mid x)}[f(z)]
&= \int f(z)\cdot q_\phi(z \mid x)\, dz \\
&= \int f(g(\epsilon))\cdot q_\phi(g(\epsilon) \mid x)\, dg(\epsilon)
&& (z = g(\epsilon)) \\
&= \int f(g(\epsilon))\cdot q_\phi(g(\epsilon) \mid x)\cdot \det\left(\frac{\partial g(\epsilon)}{\partial \epsilon}\right)\, d\epsilon
&& \text{（变量替换的雅可比）} \\
&= \int f(z)\cdot p(\epsilon)\, d\epsilon
&& \text{（用式 (1.17)）} \\
&= \mathbb{E}_{p(\epsilon)}[f(z)].
\end{aligned}
\tag{1.19}
$$

因此，如果想对 \(\phi\) 取梯度，我们可以证明

$$
\begin{aligned}
\nabla_\phi\, \mathbb{E}_{q_\phi(z \mid x)}[f(z)]
&= \nabla_\phi\, \mathbb{E}_{p(\epsilon)}[f(z)]
= \nabla_\phi \int f(z)\cdot p(\epsilon)\, d\epsilon
= \int \{\nabla_\phi f(z)\}\cdot p(\epsilon)\, d\epsilon
= \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(z)],
\end{aligned}
\tag{1.20}
$$

这可以用蒙特卡洛近似。代入 \(f(z) = -\log q_\phi(z \mid x)\)，我们可以证明

$$
\begin{aligned}
\nabla_\phi\, \mathbb{E}_{q_\phi(z \mid x)}[-\log q_\phi(z \mid x)]
&= \mathbb{E}_{p(\epsilon)}[-\nabla_\phi \log q_\phi(z \mid x)] \\
&\approx -\frac{1}{L}\sum_{\ell=1}^{L}\nabla_\phi \log q_\phi(z^{(\ell)} \mid x),
\qquad z^{(\ell)} = g(\epsilon^{(\ell)}, \phi, x) \\
&= -\frac{1}{L}\sum_{\ell=1}^{L}\nabla_\phi\left[\log p(\epsilon^{(\ell)}) - \log \det \frac{\partial z^{(\ell)}}{\partial \epsilon}\right] \\
&= \frac{1}{L}\sum_{\ell=1}^{L}\nabla_\phi \log \det \frac{\partial z^{(\ell)}}{\partial \epsilon}.
\end{aligned}
$$

因此，只要行列式关于 \(\phi\) 可微，蒙特卡洛近似就可以数值计算。

**例 1.8.** 设参数和分布 \(q_\phi\) 定义如下：

$$
(\mu, \sigma^2) = \mathrm{EncoderNetwork}_\phi(x), \qquad q_\phi(z \mid x) = \mathcal{N}\bigl(z \mid \mu, \mathrm{diag}(\sigma^2)\bigr).
$$

我们可以定义 \(z = \mu + \sigma \odot \epsilon\)，其中 \(\epsilon \sim \mathcal{N}(0, I)\)。于是可以证明

$$
\log \det \frac{\partial z}{\partial \epsilon}
= \log \det \frac{\partial(\mu + \sigma \odot \epsilon)}{\partial \epsilon}
= \log \det \mathrm{diag}\{\sigma\}
= \log \prod_{i=1}^{d}\sigma_i
= \sum_{i=1}^{d}\log \sigma_i.
$$

因此我们可以证明

$$
\begin{aligned}
\nabla_\phi\, \mathbb{E}_{q_\phi(z \mid x)}[-\log q_\phi(z \mid x)]
&\approx \frac{1}{L}\sum_{\ell=1}^{L}\nabla_\phi \log \det \frac{\partial z^{(\ell)}}{\partial \epsilon} \\
&= \frac{1}{L}\sum_{\ell=1}^{L}\nabla_\phi \sum_{i=1}^{d}\log \sigma_i \\
&= \nabla_\phi \sum_{i=1}^{d}\log \sigma_i
= \frac{1}{\sigma}\odot \nabla_\phi \sigma_\phi(x),
\end{aligned}
$$

其中我们强调 \(\sigma_\phi(x)\) 是编码器的输出，而编码器是一个神经网络。

从上例可以看到，对某些特定的分布选择（例如高斯），ELBO 的梯度可以显著更容易推导。

**VAE 编码器。** 讨论完重参数化技巧之后，我们现在可以讨论 VAE 中编码器的具体结构。为使讨论集中，我们假设一个相对常见的编码器选择：

$$
(\mu, \sigma^2) = \mathrm{EncoderNetwork}_\phi(x), \qquad q_\phi(z \mid x) = \mathcal{N}(z \mid \mu, \sigma^2 I).
$$

参数 \(\mu\) 和 \(\sigma\) 在技术上是神经网络，因为它们是 \(\mathrm{EncoderNetwork}_\phi(\cdot)\) 的输出。因此，把它们记为

$$
\mu = \underbrace{\mu_\phi}_{\text{神经网络}}(x), \qquad \sigma^2 = \underbrace{\sigma_\phi^2}_{\text{神经网络}}(x)
$$

会有帮助。我们的记号稍复杂一些，因为我们想强调 \(\mu\) 是 \(x\) 的函数：你给我们一幅图像 \(x\)，我们的工作是返回高斯的参数（即均值和方差）。如果你给我们不同的 \(x\)，高斯的参数也应不同。参数 \(\phi\) 表明 \(\mu\) 由 \(\phi\) 控制（或参数化）。

假设给定第 \(\ell\) 个训练样本 \(x^{(\ell)}\)。从该 \(x^{(\ell)}\) 我们想生成潜变量 \(z^{(\ell)}\)，它是 \(q_\phi(z \mid x)\) 的一个样本。由于高斯结构，这等价于说

$$
z^{(\ell)} \sim \mathcal{N}\!\left(z \,\middle|\, \mu_\phi(x^{(\ell)}),\, \sigma_\phi^2(x^{(\ell)}) I\right). \tag{1.21}
$$

这个方程有趣的地方在于，我们用神经网络 \(\mathrm{EncoderNetwork}_\phi(\cdot)\) 估计高斯的均值和方差。然后从这个高斯中抽取样本 \(z^{(\ell)}\)，如图 1.5 所示。

**图 1.5：** VAE 编码器的实现。我们用一个神经网络取图像 \(x\)，并估计高斯分布的均值 \(\mu_\phi\) 和方差 \(\sigma_\phi^2\)。

表达式 (1.21) 的一种更方便的方式是认识到：采样操作 \(z \sim \mathcal{N}(\mu, \sigma^2 I)\) 可以用重参数化技巧完成。

**高维高斯的重参数化技巧：**

$$
z \sim \mathcal{N}(\mu, \sigma^2 I) \quad \Longleftrightarrow \quad z = \mu + \sigma \epsilon,\quad \epsilon \sim \mathcal{N}(0, I). \tag{1.22}
$$

利用重参数化技巧，式 (1.21) 可以写成

$$
z^{(\ell)} = \mu_\phi(x^{(\ell)}) + \sigma_\phi(x^{(\ell)})\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).
$$

**证明.** 我们将证明任意协方差矩阵 \(\Sigma\)（而不只是对角矩阵 \(\sigma^2 I\)）的一般情形。

对任何高维高斯 \(z \sim \mathcal{N}(z \mid \mu, \Sigma)\)，采样过程可以通过白噪声的变换完成：

$$
z = \mu + \Sigma^{1/2}\epsilon, \tag{1.23}
$$

其中 \(\epsilon \sim \mathcal{N}(0, I)\)。半矩阵 \(\Sigma^{1/2}\) 可以通过特征分解或 Cholesky 分解得到。若 \(\Sigma\) 有特征分解 \(\Sigma = U S U^\top\)，则 \(\Sigma^{1/2} = U S^{1/2} U^\top\)。特征值矩阵 \(S\) 的平方根是良定义的，因为 \(\Sigma\) 是半正定矩阵。

我们可以计算 \(z\) 的期望和协方差：

$$
\begin{aligned}
\mathbb{E}[z] &= \mathbb{E}[\mu + \Sigma^{1/2}\epsilon] = \mu + \Sigma^{1/2}\underbrace{\mathbb{E}[\epsilon]}_{=0} = \mu, \\
\mathrm{Cov}(z) &= \mathbb{E}[(z - \mu)(z - \mu)^\top]
= \mathbb{E}\bigl[\Sigma^{1/2}\epsilon\epsilon^\top (\Sigma^{1/2})^\top\bigr]
= \Sigma^{1/2}\underbrace{\mathbb{E}[\epsilon\epsilon^\top]}_{=I}(\Sigma^{1/2})^\top = \Sigma.
\end{aligned}
$$

因此，对对角矩阵 \(\Sigma = \sigma^2 I\)，上式化为

$$
z = \mu + \sigma \epsilon, \qquad \text{其中 } \epsilon \sim \mathcal{N}(0, I). \tag{1.24}
$$

给定 VAE 编码器结构和 \(q_\phi(z \mid x)\)，我们可以回到 ELBO。回忆 ELBO 由先验匹配项和重建项组成。先验匹配项用 KL 散度 \(D_{\mathrm{KL}}(q_\phi(z \mid x)\,\|\, p(z))\) 度量。让我们计算这个 KL 散度。

为计算 KL 散度，我们（重新）使用下面总结的结果：

**定理 1.3. 两个高斯的 KL 散度。** 两个 \(d\) 维高斯分布 \(\mathcal{N}(\mu_0, \Sigma_0)\) 与 \(\mathcal{N}(\mu_1, \Sigma_1)\) 的 KL 散度为

$$
\begin{aligned}
&D_{\mathrm{KL}}\bigl(\mathcal{N}(\mu_0, \Sigma_0)\,\|\, \mathcal{N}(\mu_1, \Sigma_1)\bigr) \\
&= \frac{1}{2}\left(\mathrm{Tr}(\Sigma_1^{-1}\Sigma_0) - d + (\mu_1 - \mu_0)^\top \Sigma_1^{-1}(\mu_1 - \mu_0) + \log\frac{\det\Sigma_1}{\det\Sigma_0}\right).
\end{aligned}
\tag{1.25}
$$

代入我们的分布，考虑

$$
\mu_0 = \mu_\phi(x),\quad \Sigma_0 = \sigma_\phi^2(x) I, \qquad \mu_1 = 0,\quad \Sigma_1 = I,
$$

我们可以证明 KL 散度有解析表达式

$$
D_{\mathrm{KL}}\bigl(q_\phi(z \mid x)\,\|\, p(z)\bigr)
= \frac{1}{2}\bigl(\sigma_\phi^2(x)\, d - d + \lVert \mu_\phi(x)\rVert^2 - 2d\log \sigma_\phi(x)\bigr), \tag{1.26}
$$

其中 \(d\) 是向量 \(z\) 的维数。KL 散度关于 \(\phi\) 的梯度没有闭式，但可以数值计算：

$$
\nabla_\phi D_{\mathrm{KL}}\bigl(q_\phi(z \mid x)\,\|\, p(z)\bigr)
= \frac{1}{2}\nabla_\phi\bigl(\sigma_\phi^2(x)\, d - d + \lVert \mu_\phi(x)\rVert^2 - 2d\log \sigma_\phi(x)\bigr). \tag{1.27}
$$

关于 \(\theta\) 的梯度为零，因为没有任何东西依赖于 \(\theta\)。

**VAE 解码器。** 解码器通过神经网络实现。为记号简单，把它定义为 \(\mathrm{DecoderNetwork}_\theta(\cdot)\)，其中 \(\theta\) 表示网络参数。解码器网络的工作是取潜变量 \(z\) 并生成图像 \(f_\theta(z)\)：

$$
f_\theta(z) = \mathrm{DecoderNetwork}_\theta(z). \tag{1.28}
$$

分布 \(p_\theta(x \mid z)\) 可以定义为

$$
p_\theta(x \mid z) = \mathcal{N}\bigl(x \mid f_\theta(z), \sigma_{\mathrm{dec}}^2 I\bigr), \qquad \text{对某个超参数 } \sigma_{\mathrm{dec}}. \tag{1.29}
$$

\(p_\theta(x \mid z)\) 的解释是：我们通过网络估计 \(f_\theta(z)\)，并把它作为高斯的均值。如果我们从 \(p_\theta(x \mid z)\) 抽取样本 \(x\)，那么由重参数化技巧可以把生成图像 \(\hat{x}\) 写成

$$
\hat{x} = f_\theta(z) + \sigma_{\mathrm{dec}}\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).
$$

此外，如果取似然的对数，我们可以证明

$$
\begin{aligned}
\log p_\theta(x \mid z)
&= \log \mathcal{N}\bigl(x \mid f_\theta(z), \sigma_{\mathrm{dec}}^2 I\bigr) \\
&= \log \frac{1}{(2\pi\sigma_{\mathrm{dec}}^2)^{d/2}}\exp\left(-\frac{\lVert x - f_\theta(z)\rVert^2}{2\sigma_{\mathrm{dec}}^2}\right) \\
&= -\frac{\lVert x - f_\theta(z)\rVert^2}{2\sigma_{\mathrm{dec}}^2} - \underbrace{\tfrac{1}{2}\log(2\pi\sigma_{\mathrm{dec}}^2)^d}_{\text{与 }\theta\text{ 无关，可以丢掉}}.
\end{aligned}
\tag{1.30}
$$

回到 ELBO，我们想计算 \(\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]\)。如果直接计算期望，需要计算积分

$$
\begin{aligned}
\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
&= \int \log \mathcal{N}\bigl(x \mid f_\theta(z), \sigma_{\mathrm{dec}}^2 I\bigr)\cdot \mathcal{N}\bigl(z \mid \mu_\phi(x), \sigma_\phi^2(x)\bigr)\, dz \\
&= -\int \frac{\lVert x - f_\theta(z)\rVert^2}{2\sigma_{\mathrm{dec}}^2}\cdot \mathcal{N}\bigl(z \mid \mu_\phi(x), \sigma_\phi^2(x)\bigr)\, dz + C,
\end{aligned}
$$

其中来自高斯对数的常数 \(C\) 可以丢掉。利用重参数化技巧，我们写 \(z = \mu_\phi(x) + \sigma_\phi(x)\epsilon\) 并代入上式。这将给出<sup>3</sup>

$$
\begin{aligned}
\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
&\approx -\frac{1}{M}\sum_{m=1}^{M}\frac{\lVert x - f_\theta(z^{(m)})\rVert^2}{2\sigma_{\mathrm{dec}}^2} \\
&= -\frac{1}{M}\sum_{m=1}^{M}\frac{\bigl\lVert x - f_\theta\bigl(\mu_\phi(x) + \sigma_\phi(x)\epsilon^{(m)}\bigr)\bigr\rVert^2}{2\sigma_{\mathrm{dec}}^2}.
\end{aligned}
\tag{1.31}
$$

上述近似来自蒙特卡洛，随机性基于对 \(\epsilon \sim \mathcal{N}(\epsilon \mid 0, I)\) 的采样。下标 \(M\) 指定我们想用来近似期望的蒙特卡洛样本数。注意输入图像 \(x\) 是固定的，因为 \(\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]\) 是 \(x\) 的函数。

\(\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]\) 关于 \(\theta\) 的梯度相对容易计算。因为只有 \(f_\theta\) 依赖于 \(\theta\)，我们可以做自动微分。关于 \(\phi\) 的梯度稍难，但仍然可计算，因为我们用链式法则进入 \(\mu_\phi(x)\) 和 \(\sigma_\phi(x)\)。

检查式 (1.31)，我们注意到一件有趣的事：损失函数就是重建图像 \(f_\theta(z)\) 与真值图像 \(x\) 之间的 \(\ell_2\) 范数。这意味着如果我们有生成图像 \(f_\theta(z)\)，就可以通过通常的 \(\ell_2\) 损失与真值 \(x\) 直接比较，如图 1.6 所示。

<sup>3</sup> 这里的负号不是笔误。我们想最大化 \(\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]\)，这等价于最小化 \(\ell_2\) 范数的负值。

**图 1.6：** VAE 解码器的实现。我们用一个神经网络取潜向量 \(z\) 并生成图像 \(f_\theta(z)\)。如果我们假设高斯分布，对数似然将给出一个二次式。

**训练 VAE。** 给定由干净图像组成的训练数据集 \(\mathcal{X} = \{(x^{(\ell)})\}_{\ell=1}^{L}\)，VAE 的训练目标是最大化 ELBO

$$
\operatorname*{argmax}_{\theta, \phi}\sum_{x \in \mathcal{X}} \mathrm{ELBO}_{\phi, \theta}(x),
$$

其中求和针对整个训练数据集。单个 ELBO 基于我们上面推导的各项之和：

$$
\mathrm{ELBO}_{\phi, \theta}(x) = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{\mathrm{KL}}\bigl(q_\phi(z \mid x)\,\|\, p(z)\bigr). \tag{1.32}
$$

这里，重建项是

$$
\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
\approx -\frac{1}{M}\sum_{m=1}^{M}\frac{\bigl\lVert x - f_\theta\bigl(\mu_\phi(x) + \sigma_\phi(x)\epsilon^{(m)}\bigr)\bigr\rVert^2}{2\sigma_{\mathrm{dec}}^2}, \tag{1.33}
$$

而先验匹配项是

$$
D_{\mathrm{KL}}\bigl(q_\phi(z \mid x)\,\|\, p(z)\bigr)
= \frac{1}{2}\bigl(\sigma_\phi^2(x)\, d - d + \lVert \mu_\phi(x)\rVert^2 - 2d\log \sigma_\phi(x)\bigr). \tag{1.34}
$$

为优化 \(\theta\) 和 \(\phi\)，我们可以运行随机梯度下降。梯度可以基于神经网络的计算图来取。在计算机上，这由自动微分自动完成。

让我们总结这些。

**定理 1.4.（VAE 训练）。** 要训练一个 VAE，我们需要求解优化问题

$$
\operatorname*{argmax}_{\theta, \phi}\sum_{x \in \mathcal{X}} \mathrm{ELBO}_{\phi, \theta}(x),
$$

其中

$$
\begin{aligned}
\mathrm{ELBO}_{\phi, \theta}(x)
&= -\frac{1}{M}\sum_{m=1}^{M}\frac{\bigl\lVert x - f_\theta\bigl(\mu_\phi(x) + \sigma_\phi(x)\epsilon^{(m)}\bigr)\bigr\rVert^2}{2\sigma_{\mathrm{dec}}^2} \\
&\quad - \frac{1}{2}\bigl(\sigma_\phi^2(x)\, d - d + \lVert \mu_\phi(x)\rVert^2 - 2d\log \sigma_\phi(x)\bigr).
\end{aligned}
\tag{1.35}
$$

（按式 (1.32)，先验匹配项应从重建项中减去。原文排版在第二项前印成加号；此处按该式的推导写成减号，使它与式 (1.32)–(1.34) 一致。）

**VAE 推理。** VAE 的推理相对简单。一旦 VAE 训练完成，我们可以丢掉编码器，只保留解码器，如图 1.7 所示。要从模型生成一幅新图像，我们选取一个随机潜向量 \(z \in \mathbb{R}^d\)。把这个 \(z\) 送过解码器 \(f_\theta\)，就能生成一幅新图像 \(\hat{x} = f_\theta(z)\)。

**图 1.7：** 用 VAE 生成图像，简单到只需把一个潜噪声码 \(z\) 送过解码器。

### 1.4 结语

对寻找更多参考文献的读者，我们强烈推荐 Kingma 与 Welling 的教程 [24]，它基于他们原来的 VAE 论文 [23]。Doersch 等人较短的教程 [12] 也会有帮助。[24] 包含一长串好论文，其中包括 Rezende 与 Mohamed [32] 关于归一化流的论文，该文与 Kingma 和 Welling 的 VAE 论文大约同时发表。

VAE 与经典变分推断和图模型有许多联系 [45]。VAE 也与 Goodfellow 等人的生成对抗网络（generative adversarial network，GAN）[15] 相关。Kingma 与 Welling 在 [24] 中评论说，VAE 与 GAN 具有互补性质：GAN 产生感知质量更好的图像，但与数据似然的联系较弱。VAE 能更好地满足数据似然准则，但样本有时在感知上不如 GAN 好。
---

## 2 去噪扩散概率模型（DDPM）

本节我们讨论扩散模型。扩散模型可以如何推导，有许多不同视角，例如分数匹配、微分方程等。我们将遵循 Ho 等人关于去噪扩散概率模型的原始论文所概述的方法 [16]。

在讨论数学细节之前，让我们从 VAE 的扩展这一视角总结 DDPM：

> 扩散模型是增量更新，把整体组装起来就给出编码器–解码器结构。

为什么要增量？这就像转动一艘巨轮的方向。你需要把船慢慢转向你想要的方向，否则就会失去控制。同样的原则也适用于你公司的人力资源部门和你大学的行政部门。

> 一次只弯一英寸。

好了。哲学说够了。让我们回到正事。

DDPM 与 Sohl-Dickstein 等人 2015 年的一项更早工作有很多联系 [38]。Sohl-Dickstein 等人提出的问题是：如何从一个分布转换到另一个分布。VAE 提供了一种方法：参照上一节，我们可以把源分布想成潜变量 \(z \sim p(z)\)，把目标分布想成输入变量 \(x \sim p(x)\)。然后通过建立代理分布 \(p_\theta(x \mid z)\) 和 \(q_\phi(z \mid x)\)，我们可以训练编码器和解码器，使解码器服务于生成图像的目标。但 VAE 大体上是一步生成——如果你给我们一个潜码 \(z\)，我们就要求神经网络 \(f_\theta(\cdot)\) 立刻返回生成信号 \(x \sim \mathcal{N}(x \mid f_\theta(z), \sigma_{\mathrm{dec}}^2 I)\)。在某种意义上，这对神经网络要求太多。我们要求它用几层神经元立刻从一个分布 \(p(z)\) 转换到另一个分布 \(p(x)\)。这太多了。

Sohl-Dickstein 等人提出的想法是构造一连串转换，而不是一步过程。为此他们定义了两个过程，类似于 VAE 中的编码器和解码器。他们把编码器称为前向过程（forward process），把解码器称为反向过程（reverse process）。在两个过程中，他们考虑一列变量 \(x_0,\ldots,x_T\)，其联合分布分别记为前向的 \(q_\phi(x_{0:T})\) 和反向的 \(p_\theta(x_{0:T})\)。为使两个过程都可处理（并且灵活），他们施加马尔可夫链结构（即无记忆），其中

$$
\begin{aligned}
\text{从 } x_0 \text{ 到 } x_T \text{ 的前向：}\quad
& q_\phi(x_{0:T}) = q(x_0)\prod_{t=1}^{T} q_\phi(x_t \mid x_{t-1}), \\
\text{从 } x_T \text{ 到 } x_0 \text{ 的反向：}\quad
& p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t).
\end{aligned}
$$

在两个方程中，转移分布只依赖于紧邻的前一阶段。因此，如果每个转移通过某种形式的神经网络实现，整体生成过程就被分解成许多更小的任务。这并不意味着我们需要 \(T\) 倍多的神经网络。我们只是把同一个网络重复使用 \(T\) 次。

把整体过程拆成更小的步骤，使我们能在每一步使用简单分布。如下面各小节将讨论的，我们可以用高斯分布作为转移。由于高斯的性质，如果似然和先验都是高斯，后验仍是高斯。因此，如果上面每个转移分布都是高斯，联合分布也是高斯。由于高斯完全由前两阶矩（均值和方差）刻画，计算高度可处理。在 Sohl-Dickstein 等人的原始论文中，还有二项扩散过程的案例研究。

在给出概念的高层概览之后，让我们谈一些细节。扩散模型的起点是考虑 VAE 结构，并把它变成一连串增量更新，如图 2.1 所示。这一特定结构称为变分扩散模型（variational diffusion model），这个名字由 Kingma 等人于 2021 年给出 [22]。变分扩散模型有一列状态 \(x_0, x_1, \ldots, x_T\)，解释如下：

- \(x_0\)：原始图像，与 VAE 中的 \(x\) 相同。
- \(x_T\)：潜变量，与 VAE 中的 \(z\) 相同。如上所述，为简单、可处理性和计算效率，我们选择 \(x_T \sim \mathcal{N}(0, I)\)。
- \(x_1,\ldots,x_{T-1}\)：中间状态。它们也是潜变量，但不是白高斯。

变分扩散模型的结构由两条路径组成。前向路径和反向路径类似于单步变分自编码器的路径。区别在于编码器和解码器具有相同的输入–输出维数。所有前向构件组装起来给出编码器，所有反向构件组装起来给出解码器。

**图 2.1：** Kingma 等人的变分扩散模型 [22]。在该模型中，输入图像是 \(x_0\)，白噪声是 \(x_T\)。中间变量（或状态）\(x_1,\ldots,x_{T-1}\) 是潜变量。从 \(x_{t-1}\) 到 \(x_t\) 的转移类似于 VAE 中的前向步骤（编码器），从 \(x_t\) 到 \(x_{t-1}\) 的转移类似于 VAE 中的反向步骤（解码器）。在变分扩散模型中，编码器/解码器的输入维数和输出维数相同。

### 2.1 基本构件

让我们谈谈变分扩散模型的基本构件。有三类构件：转移块、初始块和最终块。

**转移块。** 第 \(t\) 个转移块由三个状态 \(x_{t-1}\)、\(x_t\) 和 \(x_{t+1}\) 组成。到达状态 \(x_t\) 有两条可能路径，如图 2.2 所示。

**图 2.2：** 变分扩散模型的转移块由三个节点组成。转移分布 \(p(x_t \mid x_{t+1})\) 和 \(p(x_t \mid x_{t-1})\) 不可得，但我们可以用高斯近似它们。

- 第一条路径是从 \(x_{t-1}\) 到 \(x_t\) 的前向转移。相关联的转移分布是 \(p(x_t \mid x_{t-1})\)。用白话说，如果你告诉我们 \(x_{t-1}\)，我们可以按照 \(p(x_t \mid x_{t-1})\) 抽取样本 \(x_t\)。然而，就像 VAE 一样，转移分布 \(p(x_t \mid x_{t-1})\) 不可得。我们可以用某些简单分布（例如高斯）来近似它。近似分布记为 \(q_\phi(x_t \mid x_{t-1})\)。我们稍后讨论 \(q_\phi\) 的确切形式。
- 第二条路径是从 \(x_{t+1}\) 到 \(x_t\) 的反向转移。同样，我们不知道 \(p(x_t \mid x_{t+1})\)，于是有另一个代理分布（例如高斯）来近似真实分布。这个代理分布记为 \(p_\theta(x_t \mid x_{t+1})\)。

**初始块。** 变分扩散模型的初始块聚焦于状态 \(x_0\)。由于我们从 \(x_0\) 开始，只需要从 \(x_1\) 到 \(x_0\) 的反向转移。从 \(x_{-1}\) 到 \(x_0\) 的前向转移可以丢掉。因此我们只需要考虑 \(p(x_0 \mid x_1)\)。但既然 \(p(x_0 \mid x_1)\) 永远不可得，我们用高斯 \(p_\theta(x_0 \mid x_1)\) 近似它，其均值通过神经网络计算。见图 2.3。

**图 2.3：** 变分扩散模型的初始块聚焦于节点 \(x_0\)。由于时刻 \(t = 0\) 之前没有状态，我们只有从 \(x_1\) 到 \(x_0\) 的反向转移。

**最终块。** 最终块聚焦于状态 \(x_T\)。记住 \(x_T\) 应当是我们最终的潜变量，即一个白高斯噪声向量。因为它是最终块，我们只需要从 \(x_{T-1}\) 到 \(x_T\) 的前向转移，而不需要诸如从 \(x_{T+1}\) 到 \(x_T\) 之类的东西。前向转移由高斯 \(q_\phi(x_T \mid x_{T-1})\) 近似。见图 2.4。

**图 2.4：** 变分扩散模型的最终块聚焦于节点 \(x_T\)。由于时刻 \(t = T\) 之后没有状态，我们只有从 \(x_{T-1}\) 到 \(x_T\) 的前向转移。

**理解转移分布。** 在继续之前，我们需要解释转移分布 \(q_\phi(x_t \mid x_{t-1})\)。我们知道它是高斯。但这个高斯的均值和方差是什么？

**定义 2.1. 转移分布 \(q_\phi(x_t \mid x_{t-1})\)。** 在变分扩散模型中（以及我们稍后讨论的 DDPM 中），转移分布 \(q_\phi(x_t \mid x_{t-1})\) 定义为

$$
q_\phi(x_t \mid x_{t-1}) \stackrel{\mathrm{def}}{=} \mathcal{N}\bigl(x_t \mid \sqrt{\alpha_t}\, x_{t-1},\, (1 - \alpha_t)I\bigr). \tag{2.1}
$$

换言之，\(q_\phi(x_t \mid x_{t-1})\) 是高斯。均值是 \(\sqrt{\alpha_t}\, x_{t-1}\)，方差是 \(1 - \alpha_t\)。选择缩放因子 \(\sqrt{\alpha_t}\) 是为了确保方差幅度被保持，从而经过许多次迭代后既不会爆炸也不会消失。

**例 2.1.** 考虑高斯混合模型

$$
x_0 \sim p_0(x) = \pi_1 \mathcal{N}(x \mid \mu_1, \sigma_1^2) + \pi_2 \mathcal{N}(x \mid \mu_2, \sigma_2^2).
$$

给定转移概率，我们知道若 \(x_t \sim q_\phi(x_t \mid x_{t-1})\)，则

$$
x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1 - \alpha_t}\,\epsilon, \qquad \text{其中 } \epsilon \sim \mathcal{N}(0, I).
$$

我们的目标是看这一迭代过程（使用上述转移概率）在平衡态（即 \(t \to \infty\)）是否会给我们白高斯。

对混合模型，不难证明 \(x_t\) 的概率分布可以通过对 \(t = 1, 2, \ldots, T\) 的算法递归计算（证明稍后给出）：

$$
\begin{aligned}
x_t \sim p_t(x)
&= \pi_1 \mathcal{N}\bigl(x \mid \sqrt{\alpha_t}\,\mu_{1,t-1},\, \alpha_t \sigma_{1,t-1}^2 + (1 - \alpha_t)\bigr) \\
&\quad + \pi_2 \mathcal{N}\bigl(x \mid \sqrt{\alpha_t}\,\mu_{2,t-1},\, \alpha_t \sigma_{2,t-1}^2 + (1 - \alpha_t)\bigr),
\end{aligned}
\tag{2.2}
$$

其中 \(\mu_{1,t-1}\) 是时刻 \(t-1\) 第 1 类的均值，且 \(\mu_{1,0} = \mu_1\) 为初始均值。类似地，\(\sigma_{1,t-1}^2\) 是时刻 \(t-1\) 第 1 类的方差，且 \(\sigma_{1,0}^2 = \sigma_1^2\) 为初始方差。

在下图中，我们展示一个数值例子，其中 \(\pi_1 = 0.3\)，\(\pi_2 = 0.7\)，\(\mu_1 = -2\)，\(\mu_2 = 2\)，\(\sigma_1 = 0.2\)，\(\sigma_2 = 1\)。速率对所有 \(t\) 定义为 \(\alpha_t = 0.97\)。我们画出不同 \(t\) 的概率分布函数。

**图 2.5：** 分布 \(p_t(x)\) 的演化。随着时间 \(t\) 推进，双峰分布逐渐变成高斯。

**式 (2.2) 的证明。** 对想理解我们如何推导式 (2.2) 中混合模型概率密度的读者，我们可以给出一个简单推导。考虑混合模型

$$
p(x) = \sum_{k=1}^{K} \pi_k \underbrace{\mathcal{N}(x \mid \mu_k, \sigma_k^2 I)}_{p(x \mid k)}.
$$

若考虑新变量 \(y = \sqrt{\alpha}\, x + \sqrt{1 - \alpha}\,\epsilon\)，其中 \(\epsilon \sim \mathcal{N}(0, I)\)，则 \(y\) 的分布可以用全概率公式推导：

$$
p(y) = \sum_{k=1}^{K} p(y \mid k)\, p(k) = \sum_{k=1}^{K} \pi_k\, p(y \mid k).
$$

由于 \(y \mid k = \sqrt{\alpha}\, x\mid k + \sqrt{1-\alpha}\,\epsilon\) 是（条件）高斯随机变量 \(x\mid k\) 与另一个高斯随机变量 \(\epsilon\) 的线性组合，和 \(y\mid k\) 仍是高斯。均值和方差是

$$
\begin{aligned}
\mathbb{E}[y \mid k] &= \sqrt{\alpha}\,\mathbb{E}[x \mid k] + \sqrt{1-\alpha}\,\mathbb{E}[\epsilon] = \sqrt{\alpha}\,\mu_k, \\
\mathrm{Var}[y \mid k] &= \alpha\,\mathrm{Var}[x \mid k] + (1-\alpha)\,\mathrm{Var}[\epsilon] = \alpha\sigma_k^2 + (1-\alpha),
\end{aligned}
$$

其中用了 \(\mathbb{E}[\epsilon] = 0\) 和 \(\mathrm{Var}[\epsilon] = 1\)。既然我们刚论证 \(y\mid k\) 是高斯，\(y\mid k\) 的分布一旦知道均值和方差就完全确定。代入上述结果，我们知道 \(p(y \mid k) = \mathcal{N}(y \mid \sqrt{\alpha}\,\mu_k,\, \alpha\sigma_k^2 + (1-\alpha))\)。推导完成。

**神奇的标量 \(\sqrt{\alpha_t}\) 和 \(\sqrt{1-\alpha_t}\)。** 你可能会想，天才们（去噪扩散论文的作者）是如何为上述转移概率想出神奇标量 \(\sqrt{\alpha_t}\) 和 \(\sqrt{1-\alpha_t}\) 的。为揭开神秘，考虑两个无关的标量 \(a \in \mathbb{R}\) 和 \(b \in \mathbb{R}\)，并把转移分布定义为

$$
q_\phi(x_t \mid x_{t-1}) = \mathcal{N}(x_t \mid a x_{t-1},\, b^2 I). \tag{2.3}
$$

下面是发现：

**定理 2.1.（为什么是 \(\sqrt{\alpha}\) 和 \(\sqrt{1-\alpha}\)？）** 设对某些常数 \(a\) 和 \(b\) 有 \(q_\phi(x_t \mid x_{t-1}) = \mathcal{N}(x_t \mid a x_{t-1}, b^2 I)\)。如果我们想选择 \(a\) 和 \(b\)，使得 \(x_t\) 的分布变成 \(\mathcal{N}(0, I)\)，则必须有

$$
a = \sqrt{\alpha}, \qquad b = \sqrt{1-\alpha}.
$$

因此转移分布是

$$
q_\phi(x_t \mid x_{t-1}) \stackrel{\mathrm{def}}{=} \mathcal{N}\bigl(x_t \mid \sqrt{\alpha}\, x_{t-1},\, (1-\alpha)I\bigr). \tag{2.4}
$$

注：如果你喜欢噪声日程，可以把 \(\alpha\) 换成 \(\alpha_t\)。

**证明.** 我们想证明 \(a = \sqrt{\alpha}\) 且 \(b = \sqrt{1-\alpha}\)。对式 (2.3) 所示分布，等价的采样步骤是

$$
x_t = a x_{t-1} + b\epsilon_{t-1}, \qquad \text{其中 } \epsilon_{t-1} \sim \mathcal{N}(0, I). \tag{2.5}
$$

我们可以继续递推，证明

$$
\begin{aligned}
x_t &= a x_{t-1} + b\epsilon_{t-1} \\
&= a(a x_{t-2} + b\epsilon_{t-2}) + b\epsilon_{t-1}
&& \text{（代入 }x_{t-1} = a x_{t-2} + b\epsilon_{t-2}\text{）} \\
&= a^2 x_{t-2} + ab\epsilon_{t-2} + b\epsilon_{t-1}
&& \text{（重组项）} \\
&= \cdots \\
&= a^t x_0 + b\underbrace{\bigl[\epsilon_{t-1} + a\epsilon_{t-2} + a^2\epsilon_{t-3} + \cdots + a^{t-1}\epsilon_0\bigr]}_{\stackrel{\mathrm{def}}{=} w_t}.
\end{aligned}
\tag{2.6}
$$

上面的有限和是独立高斯随机变量之和。均值向量 \(\mathbb{E}[w_t]\) 保持为零，因为每一项均值都为零。协方差矩阵（对零均值向量）是

$$
\begin{aligned}
\mathrm{Cov}[w_t] &\stackrel{\mathrm{def}}{=} \mathbb{E}[w_t w_t^\top] \\
&= b^2\bigl(\mathrm{Cov}(\epsilon_{t-1}) + a^2\mathrm{Cov}(\epsilon_{t-2}) + \cdots + (a^{t-1})^2\mathrm{Cov}(\epsilon_0)\bigr) \\
&= b^2(1 + a^2 + a^4 + \cdots + a^{2(t-1)})I \\
&= b^2\cdot \frac{1 - a^{2t}}{1 - a^2}\, I.
\end{aligned}
$$

当 \(t \to \infty\) 时，对任何 \(0 < a < 1\) 有 \(a^t \to 0\)。因此在极限 \(t = \infty\) 处，

$$
\lim_{t\to\infty}\mathrm{Cov}[w_t] = \frac{b^2}{1-a^2}\, I.
$$

所以，如果我们想要 \(\lim_{t\to\infty}\mathrm{Cov}[w_t] = I\)（从而使 \(x_t\) 的分布趋近 \(\mathcal{N}(0, I)\)），则需要

$$
1 = \frac{b^2}{1-a^2},
$$

或等价地 \(b = \sqrt{1-a^2}\)。现在若令 \(a = \sqrt{\alpha}\)，则 \(b = \sqrt{1-\alpha}\)。这将给出

$$
x_t = \sqrt{\alpha}\, x_{t-1} + \sqrt{1-\alpha}\,\epsilon_{t-1}. \tag{2.7}
$$

**分布 \(q_\phi(x_t \mid x_0)\)。** 理解了神奇标量之后，我们可以谈论分布 \(q_\phi(x_t \mid x_0)\)。也就是说，我们想知道给定 \(x_0\) 时 \(x_t\) 如何分布。

**定理 2.2.（条件分布 \(q_\phi(x_t \mid x_0)\)）。** 条件分布 \(q_\phi(x_t \mid x_0)\) 由下式给出

$$
q_\phi(x_t \mid x_0) = \mathcal{N}\bigl(x_t \mid \sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t)I\bigr), \tag{2.8}
$$

其中 \(\bar\alpha_t = \prod_{i=1}^{t}\alpha_i\)。

**证明.** 要看式 (2.8) 如何推导，我们可以重做递推，但这次用 \(\sqrt{\alpha_t}\, x_{t-1}\) 和 \((1-\alpha_t)I\) 分别作为均值和协方差。这将给出

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_{t-1} \\
&= \sqrt{\alpha_t}\bigl(\sqrt{\alpha_{t-1}}\, x_{t-2} + \sqrt{1-\alpha_{t-1}}\,\epsilon_{t-2}\bigr) + \sqrt{1-\alpha_t}\,\epsilon_{t-1} \\
&= \sqrt{\alpha_t\alpha_{t-1}}\, x_{t-2} + \underbrace{\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}}\,\epsilon_{t-2} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}}_{w_1}.
\end{aligned}
\tag{2.9}
$$

因此我们有两个高斯之和。但两个高斯之和仍是高斯，我们只需计算新协方差（因为均值保持为零）。新协方差是

$$
\mathbb{E}[w_1 w_1^\top]
= \bigl[(\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}})^2 + (\sqrt{1-\alpha_t})^2\bigr]I
= [\alpha_t(1-\alpha_{t-1}) + 1 - \alpha_t]I
= [1 - \alpha_t\alpha_{t-1}]I.
$$

回到式 (2.9)，我们可以证明递推变成 \(x_{t-2}\) 与噪声向量 \(\epsilon_{t-2}\) 的线性组合：

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t\alpha_{t-1}}\, x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\,\epsilon_{t-2} \\
&= \sqrt{\alpha_t\alpha_{t-1}\alpha_{t-2}}\, x_{t-3} + \sqrt{1-\alpha_t\alpha_{t-1}\alpha_{t-2}}\,\epsilon_{t-3} \\
&= \cdots \\
&= \left(\sqrt{\prod_{i=1}^{t}\alpha_i}\right) x_0 + \left(\sqrt{1 - \prod_{i=1}^{t}\alpha_i}\right)\epsilon_0.
\end{aligned}
\tag{2.10}
$$

因此，若定义 \(\bar\alpha_t = \prod_{i=1}^{t}\alpha_i\)，我们可以证明

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon_0. \tag{2.11}
$$

换言之，分布 \(q_\phi(x_t \mid x_0)\) 是

$$
x_t \sim q_\phi(x_t \mid x_0) = \mathcal{N}\bigl(x_t \mid \sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t)I\bigr). \tag{2.12}
$$

新分布 \(q_\phi(x_t \mid x_0)\) 的用处在于，与链条 \(x_0 \to x_1 \to \cdots \to x_{T-1} \to x_T\) 相比，它是一步前向扩散。在前向扩散模型的每一步，既然我们已经知道 \(x_0\)，并假设所有后续转移都是高斯，我们对任何 \(t\) 都知道 \(x_t\)。情形可从图 2.6 理解。

**图 2.6：** \(q_\phi(x_t \mid x_{t-1})\) 与 \(q_\phi(x_t \mid x_0)\) 的区别。

**例 2.2.** 对高斯混合模型 \(x_0 \sim p_0(x) = \sum_{k=1}^{K}\pi_k \mathcal{N}(x \mid \mu_k, \sigma_k^2 I)\)，我们可以证明时刻 \(t\) 的分布是

$$
\begin{aligned}
x_t \sim p_t(x)
&= \sum_{k=1}^{K}\pi_k \mathcal{N}\bigl(x \mid \sqrt{\bar\alpha_t}\,\mu_k,\, (1-\bar\alpha_t)I + \bar\alpha_t\sigma_k^2 I\bigr) \\
&= \sum_{k=1}^{K}\pi_k \mathcal{N}\bigl(x \mid \sqrt{\alpha^t}\,\mu_k,\, (1-\alpha^t)I + \alpha^t\sigma_k^2 I\bigr),
\end{aligned}
\tag{2.13}
$$

若 \(\alpha_t = \alpha\)，从而 \(\bar\alpha_t = \prod_{i=1}^{t}\alpha = \alpha^t\)。

如果你好奇概率分布 \(p_t\) 如何随时间 \(t\) 演化，我们可以可视化例 2.1 中讨论的高斯混合分布的轨迹。我们用式 (2.13) 画热图。可以看到当 \(t = 0\) 时，初始分布是两个高斯的混合。当我们按照式 (2.13) 定义的转移推进时，可以看到分布逐渐变成单个高斯 \(\mathcal{N}(0, I)\)。

**图 2.7：** 由 \(x_t\) 形成的随机轨迹的实现。背景色图表示概率分布 \(p_t(x)\)。

在同一张图中，我们叠加并展示随机样本 \(x_t\) 作为时间 \(t\) 的函数的若干瞬时轨迹。我们用来生成样本的方程是

$$
x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).
$$

可以看到，\(x_t\) 的轨迹大体上跟随分布 \(p_t(x)\)。

许多读者感到困惑的一点是：如果目标是从图像 \(p(x_0)\) 转换到白噪声 \(p(x_T)\)，推导 \(q_\phi(x_t \mid x_{t-1})\) 和 \(q_\phi(x_t \mid x_0)\) 有什么意义？答案是，到目前为止我们一直只在谈论前向过程。在扩散模型中，前向过程被选择成可以写成闭式。更有趣的部分是反向过程。如将讨论的，反向过程通过一连串去噪操作实现。每一步去噪应当与前向过程中对应的步骤耦合。\(q_\phi(x_t \mid x_0)\) 只是为我们实现前向过程提供了一种稍更方便的方式。

### 2.2 证据下界

既然我们理解了变分扩散模型的结构，就可以写下 ELBO，从而训练模型。

**定理 2.3.（变分扩散模型的 ELBO）。** 变分扩散模型的 ELBO 是

$$
\begin{aligned}
\mathrm{ELBO}_{\phi,\theta}(x)
&= \underbrace{\mathbb{E}_{q_\phi(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)]}_{\text{初始块有多好}} \\
&\quad - \underbrace{\mathbb{E}_{q_\phi(x_{T-1} \mid x_0)}\bigl[D_{\mathrm{KL}}\bigl(q_\phi(x_T \mid x_{T-1})\,\|\, p(x_T)\bigr)\bigr]}_{\text{最终块有多好}} \\
&\quad - \sum_{t=1}^{T-1}\underbrace{\mathbb{E}_{q_\phi(x_{t-1}, x_{t+1} \mid x_0)}\bigl[D_{\mathrm{KL}}\bigl(q_\phi(x_t \mid x_{t-1})\,\|\, p_\theta(x_t \mid x_{t+1})\bigr)\bigr]}_{\text{转移块有多好}},
\end{aligned}
\tag{2.14}
$$

其中 \(x_0 = x\)，且 \(x_T \sim \mathcal{N}(0, I)\)。

如果你是本教程的随意读者，我们希望这个方程不会把你吓跑。它看起来是个怪物，但确实有结构。我们在试图理解它时只需要有耐心。

**重建（初始块）。** 先看这一项

$$
\mathbb{E}_{q_\phi(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)].
$$

这一项基于初始块，类似于式 (1.10)。期望内部的对象是对数似然 \(\log p_\theta(x_0 \mid x_1)\)。这个对数似然度量与 \(p_\theta\) 相关联的神经网络能多好地从潜变量 \(x_1\) 恢复 \(x_0\)。

期望是对从 \(q_\phi(x_1 \mid x_0)\) 抽取的样本取的。回忆 \(q_\phi(x_1 \mid x_0)\) 是生成 \(x_1\) 的分布。我们要求 \(x_1\) 从这个分布中抽取，因为 \(x_1\) 不是从天上掉下来的，而是由前向转移 \(q_\phi(x_1 \mid x_0)\) 创造的。这里需要对 \(x_0\) 条件化，因为我们需要知道原始图像是什么。

这里使用期望的原因是 \(p_\theta(x_0 \mid x_1)\) 是 \(x_1\) 的函数（因此若 \(x_1\) 随机，则 \(p_\theta(x_0 \mid x_1)\) 也随机）。对不同的中间状态 \(x_1\)，概率 \(p_\theta(x_0 \mid x_1)\) 会不同。期望消除了对 \(x_1\) 的依赖。

**先验匹配（最终块）。** 先验匹配项是

$$
-\mathbb{E}_{q_\phi(x_{T-1} \mid x_0)}\bigl[D_{\mathrm{KL}}\bigl(q_\phi(x_T \mid x_{T-1})\,\|\, p(x_T)\bigr)\bigr], \tag{2.15}
$$

它基于最终块。我们用 KL 散度度量 \(q_\phi(x_T \mid x_{T-1})\) 与 \(p(x_T)\) 之间的差异。分布 \(q_\phi(x_T \mid x_{T-1})\) 是从 \(x_{T-1}\) 到 \(x_T\) 的前向转移。它描述 \(x_T\) 如何生成。第二个分布是 \(p(x_T)\)。因为我们懒，假设 \(p(x_T) = \mathcal{N}(0, I)\)。我们希望 \(q_\phi(x_T \mid x_{T-1})\) 尽可能接近 \(\mathcal{N}(0, I)\)。

计算 KL 散度时，变量 \(x_T\) 是哑变量。然而，由于 \(q_\phi\) 以 \(x_{T-1}\) 为条件，这里算出的 KL 散度是条件变量 \(x_{T-1}\) 的函数。\(x_{T-1}\) 从哪里来？它由 \(q_\phi(x_{T-1} \mid x_0)\) 生成。我们使用条件分布 \(q_\phi(x_{T-1} \mid x_0)\)，因为 \(x_{T-1}\) 依赖于我们一开始使用的 \(x_0\)。对 \(q_\phi(x_{T-1} \mid x_0)\) 取期望是说：对由 \(q_\phi(x_{T-1} \mid x_0)\) 生成的每一个 \(x_{T-1}\)，我们都会有一个 KL 散度的值。我们对所有可能生成的 \(x_{T-1}\) 取期望，以消除这一依赖。

**一致性（转移块）。** 一致性项是

$$
-\sum_{t=1}^{T-1}\mathbb{E}_{q_\phi(x_{t-1}, x_{t+1} \mid x_0)}\bigl[D_{\mathrm{KL}}\bigl(q_\phi(x_t \mid x_{t-1})\,\|\, p_\theta(x_t \mid x_{t+1})\bigr)\bigr], \tag{2.16}
$$

它基于转移块。如果你回忆图 2.2，有两个方向。前向转移由分布 \(q_\phi(x_t \mid x_{t-1})\) 决定，反向转移由另一个分布 \(p_\theta(x_t \mid x_{t+1})\) 决定。一致性项用 KL 散度度量偏差。

期望是对从 \(q_\phi(x_{t-1}, x_{t+1} \mid x_0)\) 抽取的样本对 \((x_{t-1}, x_{t+1})\) 取的。原因是上面的 KL 散度是 \(x_{t-1}\) 和 \(x_{t+1}\) 的函数。（你可以忽略 \(x_t\)，因为它是哑变量，在我们计算期望的积分过程中会被消去。）由于依赖于 \(x_{t-1}\) 和 \(x_{t+1}\)，我们需要取期望。

**定理 2.3 的证明.** 定义记号：\(x_{0:T} = \{x_0,\ldots,x_T\}\) 表示从 \(t = 0\) 到 \(t = T\) 的全部状态变量的集合。我们也回忆先验分布 \(p(x)\) 是图像 \(x_0\) 的分布，因此它等价于 \(p(x_0)\)。有了这些，我们可以证明

$$
\begin{aligned}
\log p(x) &= \log p(x_0) \\
&= \log \int p(x_{0:T})\, dx_{1:T}
&& \text{（对 }x_{1:T}\text{ 积分做边缘化）} \\
&= \log \int p(x_{0:T})\frac{q_\phi(x_{1:T} \mid x_0)}{q_\phi(x_{1:T} \mid x_0)}\, dx_{1:T}
&& \text{（乘除 }q_\phi(x_{1:T} \mid x_0)\text{）} \\
&= \log \int q_\phi(x_{1:T} \mid x_0)\left[\frac{p(x_{0:T})}{q_\phi(x_{1:T} \mid x_0)}\right] dx_{1:T}
&& \text{（重排项）} \\
&= \log \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\frac{p(x_{0:T})}{q_\phi(x_{1:T} \mid x_0)}\right]
&& \text{（期望的定义）}.
\end{aligned}
$$

现在我们需要用 Jensen 不等式，它指出对任何随机变量 \(X\) 和任何凹函数 \(f\)，有 \(f(\mathbb{E}[X]) \ge \mathbb{E}[f(X)]\)。认出 \(f(\cdot) = \log(\cdot)\)，我们可以证明

$$
\log p(x)
= \log \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\frac{p(x_{0:T})}{q_\phi(x_{1:T} \mid x_0)}\right]
\ge \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log \frac{p(x_{0:T})}{q_\phi(x_{1:T} \mid x_0)}\right]. \tag{2.17}
$$

让我们更仔细地看 \(p(x_{0:T})\)。检查图 2.2，我们注意到如果想分解 \(p(x_{0:T})\)，应当对 \(x_{t-1}\mid x_t\) 做条件化。这导致

$$
p(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p(x_{t-1} \mid x_t) = p(x_T)\, p(x_0 \mid x_1)\prod_{t=2}^{T} p(x_{t-1} \mid x_t). \tag{2.18}
$$

至于 \(q_\phi(x_{1:T} \mid x_0)\)，图 2.2 建议我们需要对 \(x_t \mid x_{t-1}\) 做条件化。然而，由于顺序关系，我们可以写

$$
q_\phi(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q_\phi(x_t \mid x_{t-1}) = q_\phi(x_T \mid x_{T-1})\prod_{t=1}^{T-1} q_\phi(x_t \mid x_{t-1}). \tag{2.19}
$$

把式 (2.18) 和式 (2.19) 代回式 (2.17)，我们可以证明

$$
\begin{aligned}
\log p(x)
&\ge \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_{0:T})}{q_\phi(x_{1:T} \mid x_0)}\right] \\
&= \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_T)p(x_0 \mid x_1)\prod_{t=2}^{T}p(x_{t-1}\mid x_t)}{q_\phi(x_T \mid x_{T-1})\prod_{t=1}^{T-1}q_\phi(x_t \mid x_{t-1})}\right] \\
&= \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_T)p(x_0 \mid x_1)\prod_{t=1}^{T-1}p(x_t \mid x_{t+1})}{q_\phi(x_T \mid x_{T-1})\prod_{t=1}^{T-1}q_\phi(x_t \mid x_{t-1})}\right]
&& \text{（把 }t\text{ 移到 }t+1\text{）} \\
&= \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_T)p(x_0 \mid x_1)}{q_\phi(x_T \mid x_{T-1})}\right]
+ \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\prod_{t=1}^{T-1}\frac{p(x_t \mid x_{t+1})}{q_\phi(x_t \mid x_{t-1})}\right]
&& \text{（拆开期望）}.
\end{aligned}
$$

上面第一项可以进一步分解成两个期望：

$$
\mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_T)p(x_0 \mid x_1)}{q_\phi(x_T \mid x_{T-1})}\right]
= \underbrace{\mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}[\log p(x_0 \mid x_1)]}_{\text{重建}}
+ \underbrace{\mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_T)}{q_\phi(x_T \mid x_{T-1})}\right]}_{\text{先验匹配}}.
$$

重建项可以简化为

$$
\mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}[\log p(x_0 \mid x_1)] = \mathbb{E}_{q_\phi(x_1 \mid x_0)}[\log p(x_0 \mid x_1)],
$$

其中我们用了如下事实：当感兴趣的对象（即 \(\log p(x_0 \mid x_1)\)）只涉及 \(x_0\) 和 \(x_1\) 时，条件 \(x_{1:T}\mid x_0\) 等价于 \(x_1 \mid x_0\)。

先验匹配项是

$$
\mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_T)}{q_\phi(x_T \mid x_{T-1})}\right]
= \mathbb{E}_{q_\phi(x_T, x_{T-1} \mid x_0)}\left[\log\frac{p(x_T)}{q_\phi(x_T \mid x_{T-1})}\right],
$$

其中我们注意到条件期望可以简化为只对样本 \(x_T\) 和 \(x_{T-1}\)，因为 \(\log p(x_T)/q_\phi(x_T \mid x_{T-1})\) 只依赖于 \(x_T\) 和 \(x_{T-1}\)。对期望项，概率的链式法则告诉我们 \(q_\phi(x_T, x_{T-1} \mid x_0) = q_\phi(x_T \mid x_{T-1}, x_0)\, q_\phi(x_{T-1} \mid x_0)\)。由于 \(q_\phi\) 是马尔可夫的，我们可以进一步写 \(q_\phi(x_T \mid x_{T-1}, x_0) = q_\phi(x_T \mid x_{T-1})\)。因此联合期望 \(\mathbb{E}_{q_\phi(x_T, x_{T-1} \mid x_0)}\) 可以写成两个期望的乘积 \(\mathbb{E}_{q_\phi(x_{T-1} \mid x_0)}\mathbb{E}_{q_\phi(x_T \mid x_{T-1})}\)。这将给出

$$
\begin{aligned}
\mathbb{E}_{q_\phi(x_T, x_{T-1} \mid x_0)}\left[\log\frac{p(x_T)}{q_\phi(x_T \mid x_{T-1})}\right]
&= \mathbb{E}_{q_\phi(x_{T-1} \mid x_0)}\left[\mathbb{E}_{q_\phi(x_T \mid x_{T-1})}\left[\log\frac{p(x_T)}{q_\phi(x_T \mid x_{T-1})}\right]\right] \\
&= -\mathbb{E}_{q_\phi(x_{T-1} \mid x_0)}\bigl[D_{\mathrm{KL}}\bigl(q_\phi(x_T \mid x_{T-1})\,\|\, p(x_T)\bigr)\bigr].
\end{aligned}
$$

最后看乘积项。我们可以证明

$$
\begin{aligned}
\mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\prod_{t=1}^{T-1}\frac{p(x_t \mid x_{t+1})}{q_\phi(x_t \mid x_{t-1})}\right]
&= \sum_{t=1}^{T-1}\mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_t \mid x_{t+1})}{q_\phi(x_t \mid x_{t-1})}\right] \\
&= \sum_{t=1}^{T-1}\mathbb{E}_{q_\phi(x_{t-1}, x_t, x_{t+1} \mid x_0)}\left[\log\frac{p(x_t \mid x_{t+1})}{q_\phi(x_t \mid x_{t-1})}\right],
\end{aligned}
$$

其中我们再次用了期望只需要 \(x_{t-1}\)、\(x_t\) 和 \(x_{t+1}\) 这一事实。然后，用同样的条件独立论证，我们可以证明

$$
\begin{aligned}
&\sum_{t=1}^{T-1}\mathbb{E}_{q_\phi(x_{t-1}, x_t, x_{t+1} \mid x_0)}\left[\log\frac{p(x_t \mid x_{t+1})}{q_\phi(x_t \mid x_{t-1})}\right] \\
&= \sum_{t=1}^{T-1}\mathbb{E}_{q_\phi(x_{t-1}, x_{t+1} \mid x_0)}\left[\mathbb{E}_{q_\phi(x_t \mid x_{t-1})}\left[\log\frac{p(x_t \mid x_{t+1})}{q_\phi(x_t \mid x_{t-1})}\right]\right] \\
&= -\sum_{t=1}^{T-1}\mathbb{E}_{q_\phi(x_{t-1}, x_{t+1} \mid x_0)}\bigl[D_{\mathrm{KL}}\bigl(q_\phi(x_t \mid x_{t-1})\,\|\, p(x_t \mid x_{t+1})\bigr)\bigr].
\end{aligned}
$$

把 \(p(x_0 \mid x_1)\) 换成 \(p_\theta(x_0 \mid x_1)\)，把 \(p(x_t \mid x_{t+1})\) 换成 \(p_\theta(x_t \mid x_{t+1})\)，我们就完成了。

**改写一致性项。** 上述变分扩散模型的噩梦是，我们需要从联合分布 \(q_\phi(x_{t-1}, x_{t+1} \mid x_0)\) 抽取样本 \((x_{t-1}, x_{t+1})\)。我们不知道 \(q_\phi(x_{t-1}, x_{t+1} \mid x_0)\) 是什么！按我们的选择它是高斯，但我们仍然需要用未来样本 \(x_{t+1}\) 来抽取当前样本 \(x_t\)。这很奇怪。

检查一致性项，我们注意到 \(q_\phi(x_t \mid x_{t-1})\) 和 \(p_\theta(x_t \mid x_{t+1})\) 沿两个相反方向运动。因此，我们不可避免地需要使用 \(x_{t-1}\) 和 \(x_{t+1}\)。我们需要问的问题是：能否想出某种东西，使我们不必处理两个相反方向，同时仍能检查一致性？

于是，这里有一个称为贝叶斯定理的简单技巧，它将给我们

$$
q(x_t \mid x_{t-1}) = \frac{q(x_{t-1} \mid x_t)\, q(x_t)}{q(x_{t-1})}
\quad\Longrightarrow\quad
q(x_t \mid x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0)\, q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}. \tag{2.20}
$$

通过这一条件顺序的改变，我们可以把 \(q(x_t \mid x_{t-1}, x_0)\) 换成 \(q(x_{t-1} \mid x_t, x_0)\)，方法是再加一个条件变量 \(x_0\)。（如果不对 \(x_0\) 条件化，我们就无法例如从 \(q(x_{t-1})\) 抽取样本，因为 \(x_{t-1}\) 的特定状态依赖于初始图像 \(x_0\)。）方向 \(q(x_{t-1} \mid x_t, x_0)\) 现在与 \(p_\theta(x_{t-1} \mid x_t)\) 平行，如图 2.8 所示。因此，如果我们想改写一致性项，一个自然选择是计算 \(q_\phi(x_{t-1} \mid x_t, x_0)\) 与 \(p_\theta(x_{t-1} \mid x_t)\) 之间的 KL 散度。

**图 2.8：** 如果我们考虑式 (2.20) 中的贝叶斯定理，就可以定义一个方向与 \(p_\theta(x_{t-1} \mid x_t)\) 平行的分布 \(q_\phi(x_{t-1} \mid x_t, x_0)\)。

如果我们设法走过几步（无聊的）代数推导，可以证明 ELBO 现在是：

**定理 2.4.（变分扩散模型的 ELBO）。** 设 \(x = x_0\)，且 \(x_T \sim \mathcal{N}(0, I)\)。定理 2.3 中变分扩散模型的 ELBO 可以等价地写成

$$
\begin{aligned}
\mathrm{ELBO}_{\phi,\theta}(x)
&= \underbrace{\mathbb{E}_{q_\phi(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)]}_{\text{与以前相同}}
- \underbrace{D_{\mathrm{KL}}\bigl(q_\phi(x_T \mid x_0)\,\|\, p(x_T)\bigr)}_{\text{新的先验匹配}} \\
&\quad - \sum_{t=2}^{T}\underbrace{\mathbb{E}_{q_\phi(x_t \mid x_0)}\bigl[D_{\mathrm{KL}}\bigl(q_\phi(x_{t-1} \mid x_t, x_0)\,\|\, p_\theta(x_{t-1} \mid x_t)\bigr)\bigr]}_{\text{新的一致性}}.
\end{aligned}
\tag{2.21}
$$

让我们快速做三种解释：

- **重建。** 新的重建项与以前相同。我们仍在最大化对数似然。
- **先验匹配。** 新的先验匹配简化为 \(q_\phi(x_T \mid x_0)\) 与 \(p(x_T)\) 之间的 KL 散度。变化是因为我们现在以 \(x_0\) 为条件。因此不需要从 \(q_\phi(x_{T-1} \mid x_0)\) 抽取样本并取期望。
- **一致性。** 新的一致性项在两方面不同于以前。首先，运行下标 \(t\) 从 \(t = 2\) 开始、到 \(t = T\) 结束。以前是从 \(t = 1\) 到 \(t = T-1\)。与此相伴的是分布匹配，现在是在 \(q_\phi(x_{t-1} \mid x_t, x_0)\) 与 \(p_\theta(x_{t-1} \mid x_t)\) 之间。因此，我们不是要求一个前向转移去匹配一个反向转移，而是用 \(q_\phi\) 构造一个反向转移，并用它去匹配 \(p_\theta\)。

**定理 2.4 的证明.** 我们从式 (2.17) 开始，证明

$$
\begin{aligned}
\log p(x)
&\ge \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_{0:T})}{q_\phi(x_{1:T} \mid x_0)}\right]
&& \text{（由式 (2.17)）} \\
&= \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_T)p(x_0 \mid x_1)\prod_{t=2}^{T}p(x_{t-1}\mid x_t)}{q_\phi(x_1 \mid x_0)\prod_{t=2}^{T}q_\phi(x_t \mid x_{t-1}, x_0)}\right]
&& \text{（拆开链条）} \\
&= \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_T)p(x_0 \mid x_1)}{q_\phi(x_1 \mid x_0)}\right]
+ \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\prod_{t=2}^{T}\frac{p(x_{t-1}\mid x_t)}{q_\phi(x_t \mid x_{t-1}, x_0)}\right].
\end{aligned}
\tag{2.22}
$$

考虑第二项：

$$
\begin{aligned}
\prod_{t=2}^{T}\frac{p(x_{t-1}\mid x_t)}{q_\phi(x_t \mid x_{t-1}, x_0)}
&= \prod_{t=2}^{T}\frac{p(x_{t-1}\mid x_t)}{q_\phi(x_{t-1}\mid x_t, x_0)\, q_\phi(x_t \mid x_0)/q_\phi(x_{t-1}\mid x_0)}
&& \text{（贝叶斯法则，式 (2.20)）} \\
&= \prod_{t=2}^{T}\frac{p(x_{t-1}\mid x_t)}{q_\phi(x_{t-1}\mid x_t, x_0)}\times \prod_{t=2}^{T}\frac{q_\phi(x_{t-1}\mid x_0)}{q_\phi(x_t \mid x_0)}
&& \text{（重排分母）} \\
&= \left(\prod_{t=2}^{T}\frac{p(x_{t-1}\mid x_t)}{q_\phi(x_{t-1}\mid x_t, x_0)}\right)\times \frac{q_\phi(x_1 \mid x_0)}{q_\phi(x_T \mid x_0)},
&& \text{（递推消去项）}
\end{aligned}
$$

其中最后一个方程用了如下事实：对任何序列 \(a_1,\ldots,a_T\)，有 \(\prod_{t=2}^{T} a_{t-1}/a_t = a_1/a_T\)。回到式 (2.22)，我们可以看到

$$
\begin{aligned}
&\mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_T)p(x_0 \mid x_1)}{q_\phi(x_1 \mid x_0)}\right]
+ \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\prod_{t=2}^{T}\frac{p(x_{t-1}\mid x_t)}{q_\phi(x_t \mid x_{t-1}, x_0)}\right] \\
&= \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_T)p(x_0 \mid x_1)}{q_\phi(x_1 \mid x_0)} + \log\frac{q_\phi(x_1 \mid x_0)}{q_\phi(x_T \mid x_0)}\right]
+ \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\prod_{t=2}^{T}\frac{p(x_{t-1}\mid x_t)}{q_\phi(x_{t-1}\mid x_t, x_0)}\right] \\
&= \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_T)p(x_0 \mid x_1)}{q_\phi(x_T \mid x_0)}\right]
+ \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\prod_{t=2}^{T}\frac{p(x_{t-1}\mid x_t)}{q_\phi(x_{t-1}\mid x_t, x_0)}\right],
\end{aligned}
$$

其中我们在分子和分母中消去了 \(q_\phi(x_1 \mid x_0)\)，因为对任何正常数 \(a, b, c\) 有 \(\log(a/b) + \log(b/c) = \log(a/c)\)。这将给出

$$
\begin{aligned}
\mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_T)p(x_0 \mid x_1)}{q_\phi(x_T \mid x_0)}\right]
&= \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}[\log p(x_0 \mid x_1)] + \mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\frac{p(x_T)}{q_\phi(x_T \mid x_0)}\right] \\
&= \underbrace{\mathbb{E}_{q_\phi(x_1 \mid x_0)}[\log p(x_0 \mid x_1)]}_{\text{重建}}
- \underbrace{D_{\mathrm{KL}}\bigl(q_\phi(x_T \mid x_0)\,\|\, p(x_T)\bigr)}_{\text{先验匹配}}.
\end{aligned}
$$

最后一项是

$$
\begin{aligned}
&\mathbb{E}_{q_\phi(x_{1:T} \mid x_0)}\left[\log\prod_{t=2}^{T}\frac{p(x_{t-1}\mid x_t)}{q_\phi(x_{t-1}\mid x_t, x_0)}\right] \\
&= \sum_{t=2}^{T}\mathbb{E}_{q_\phi(x_t, x_{t-1} \mid x_0)}\left[\log\frac{p(x_{t-1}\mid x_t)}{q_\phi(x_{t-1}\mid x_t, x_0)}\right] \\
&= \sum_{t=2}^{T}\iint \log\frac{p(x_{t-1}\mid x_t)}{q_\phi(x_{t-1}\mid x_t, x_0)}\cdot q_\phi(x_t, x_{t-1} \mid x_0)\, dx_{t-1}\, dx_t \\
&= \sum_{t=2}^{T}\iint \log\frac{p(x_{t-1}\mid x_t)}{q_\phi(x_{t-1}\mid x_t, x_0)}\cdot q_\phi(x_{t-1}\mid x_t, x_0)\, q_\phi(x_t \mid x_0)\, dx_{t-1}\, dx_t \\
&= -\sum_{t=2}^{T}\underbrace{\mathbb{E}_{q_\phi(x_t \mid x_0)}\bigl[D_{\mathrm{KL}}\bigl(q_\phi(x_{t-1}\mid x_t, x_0)\,\|\, p(x_{t-1}\mid x_t)\bigr)\bigr]}_{\text{一致性}}.
\end{aligned}
$$

最后，把 \(p(x_{t-1}\mid x_t)\) 换成 \(p_\theta(x_{t-1}\mid x_t)\)，把 \(p(x_0 \mid x_1)\) 换成 \(p_\theta(x_0 \mid x_1)\)。完成。

### 2.3 反向过程的分布

既然我们知道了变分扩散模型的新 ELBO，就应当花些时间讨论它的核心成分 \(q_\phi(x_{t-1} \mid x_t, x_0)\)。简而言之，我们想证明的是：

- \(q_\phi(x_{t-1} \mid x_t, x_0)\) 仍是高斯。
- 既然它是高斯，它完全由均值和协方差刻画。结果是

$$
q_\phi(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1} \mid \heartsuit\, x_t + \spadesuit\, x_0,\, \clubsuit\, I), \tag{2.23}
$$

对下面定义的某些神奇标量 \(\heartsuit\)、\(\spadesuit\) 和 \(\clubsuit\)。

**定理 2.5.** 分布 \(q_\phi(x_{t-1} \mid x_t, x_0)\) 取如下形式

$$
q_\phi(x_{t-1} \mid x_t, x_0) = \mathcal{N}\bigl(x_{t-1} \mid \mu_q(x_t, x_0),\, \Sigma_q(t)\bigr), \tag{2.24}
$$

其中

$$
\mu_q(x_t, x_0)
= \frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}}{1-\bar\alpha_t}\, x_t + \frac{(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}\, x_0.
\tag{2.25}
$$

$$
\Sigma_q(t)
= \frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, I \stackrel{\mathrm{def}}{=} \sigma_q^2(t)\, I.
\tag{2.26}
$$

其中 \(\bar\alpha_t = \prod_{i=1}^{t}\alpha_i\)。

式 (2.25) 揭示一个有趣事实：均值 \(\mu_q(x_t, x_0)\) 是 \(x_t\) 和 \(x_0\) 的线性组合。几何上，\(\mu_q(x_t, x_0)\) 落在连接 \(x_t\) 与 \(x_0\) 的直线上，如图 2.9 所示。

**图 2.9：** 根据式 (2.25)，均值 \(\mu_q(x_t, x_0)\) 是 \(x_t\) 与 \(x_0\) 的线性组合。

**定理 2.5 的证明.** 利用式 (2.20) 中陈述的贝叶斯定理，若我们计算下面的高斯乘积，就可以确定 \(q(x_{t-1} \mid x_t, x_0)\)：

$$
q(x_{t-1} \mid x_t, x_0)
= \frac{\mathcal{N}(x_t \mid \sqrt{\alpha_t}\, x_{t-1}, (1-\alpha_t)I)\, \mathcal{N}(x_{t-1} \mid \sqrt{\bar\alpha_{t-1}}\, x_0, (1-\bar\alpha_{t-1})I)}{\mathcal{N}(x_t \mid \sqrt{\bar\alpha_t}\, x_0, (1-\bar\alpha_t)I)}. \tag{2.27}
$$

为简单起见，我们把向量当作标量。于是上述高斯乘积变成

$$
q(x_{t-1} \mid x_t, x_0)
\propto \exp\left\{
-\frac{(x_t - \sqrt{\alpha_t}\, x_{t-1})^2}{2(1-\alpha_t)}
-\frac{(x_{t-1} - \sqrt{\bar\alpha_{t-1}}\, x_0)^2}{2(1-\bar\alpha_{t-1})}
+\frac{(x_t - \sqrt{\bar\alpha_t}\, x_0)^2}{2(1-\bar\alpha_t)}
\right\}. \tag{2.28}
$$

我们考虑如下映射：

$$
x = x_t,\ a = \alpha_t, \qquad y = x_{t-1},\ b = \bar\alpha_{t-1}, \qquad z = x_0,\ c = \bar\alpha_t.
$$

考虑二次函数

$$
f(y) = \frac{(x - \sqrt{a}\, y)^2}{2(1-a)} + \frac{(y - \sqrt{b}\, z)^2}{2(1-b)} - \frac{(x - \sqrt{c}\, z)^2}{2(1-c)}. \tag{2.29}
$$

我们知道，无论怎样重排项，所得函数仍是二次方程。\(f(y)\) 的极小点就是所得高斯的均值。因此我们可以计算 \(f\) 的导数并证明

$$
f'(y) = \frac{1-ab}{(1-a)(1-b)}\left(y - \frac{\sqrt{a}}{1-a}x - \frac{\sqrt{b}}{1-b}z\right).
$$

令 \(f'(y) = 0\) 得到

$$
y = \frac{(1-b)\sqrt{a}}{1-ab}\, x + \frac{(1-a)\sqrt{b}}{1-ab}\, z. \tag{2.30}
$$

我们注意到 \(ab = \alpha_t \bar\alpha_{t-1} = \bar\alpha_t\)。因此

$$
\mu_q(x_t, x_0) = \frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}}{1-\bar\alpha_t}\, x_t + \frac{(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}\, x_0. \tag{2.31}
$$

类似地，对方差，我们可以检查曲率 \(f''(y)\)。我们可以容易证明

$$
f''(y) = \frac{1-ab}{(1-a)(1-b)} = \frac{1-\bar\alpha_t}{(1-\alpha_t)(1-\bar\alpha_{t-1})}.
$$

取倒数给出

$$
\Sigma_q(t) = \frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, I. \tag{2.32}
$$

上述定理中的组合权重值得研究。回忆

$$
\mu_q(x_t, x_0) = \frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}}{1-\bar\alpha_t}\, x_t + \frac{(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}\, x_0.
$$

我们可以问的一个问题是：当 \(t\) 从 \(T\) 走到 \(1\) 时，两个系数如何表现？我们在图 2.10 中展示一个例子。对这个特定例子，我们对所有 \(t\) 使用 \(\alpha_t = 0.9\)。我们把系数画成 \(t\) 的函数。图 2.10 表明，\(x_t\) 的系数随 \(t\) 从 \(t = T\) 减小到 \(t = 1\) 而缩小，而 \(x_0\) 的系数随 \(t\) 减小而增大。

**图 2.10：** \(x_t\) 与 \(x_0\) 的系数随 \(t\) 增长的轨迹。

当 \(t\) 从 \(T\) 走到 \(1\) 时，方差 \(\sigma_q^2(t)\) 也会变化。图 2.11 展示通过按照 \(q_\phi(x_{t-1} \mid x_t, x_0)\) 采样 \(x_t\) 得到的 \(x_t\) 随 \(t\) 的轨迹。在同一张图上，我们展示由 \(\sigma_q^2(t)\) 定义的高斯半径。我们的图表明，当 \(t = T\) 时，方差 \(\sigma_q^2(t)\) 相当大，使 \(x_t\) 更接近白高斯噪声。当 \(t\) 降到 \(t = 1\) 时，方差 \(\sigma_q^2(t)\) 也降到零。这说得通，因为我们最终希望 \(x_0\) 是无噪声的干净图像。

**图 2.11：** \(x_t\) 的轨迹以及相关联的高斯半径 \(\sigma_q^2(t)\)。

**构造 \(p_\theta(x_{t-1} \mid x_t)\)。** 式 (2.24) 有趣的部分是，\(q_\phi(x_{t-1} \mid x_t, x_0)\) 完全由 \(x_t\) 和 \(x_0\) 刻画。不需要神经网络来估计均值和方差！（你可以把它与需要网络的 VAE 比较。）既然不需要网络，就真的没有什么可“学习”的。如果我们知道 \(x_t\) 和 \(x_0\)，分布 \(q_\phi(x_{t-1} \mid x_t, x_0)\) 就自动确定。

这里的认识很重要。看式 (2.21) 中的一致性项：

$$
\begin{aligned}
\mathrm{ELBO}_{\phi,\theta}(x)
&= \underbrace{\mathbb{E}_{q_\phi(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)]}_{\text{与以前相同}}
- \underbrace{D_{\mathrm{KL}}\bigl(q_\phi(x_T \mid x_0)\,\|\, p(x_T)\bigr)}_{\text{新的先验匹配}} \\
&\quad - \sum_{t=2}^{T}\underbrace{\mathbb{E}_{q_\phi(x_t \mid x_0)}\bigl[D_{\mathrm{KL}}\bigl(q_\phi(x_{t-1}\mid x_t, x_0)\,\|\, p_\theta(x_{t-1}\mid x_t)\bigr)\bigr]}_{\text{新的一致性}},
\end{aligned}
$$

（来自式 (2.21)），其中我们刚证明

$$
q_\phi(x_{t-1} \mid x_t, x_0) = \mathcal{N}\bigl(x_{t-1} \mid \mu_q(x_t, x_0),\, \Sigma_q(t)\bigr).
$$

对 \(q_\phi(x_{t-1} \mid x_t, x_0)\) 没有“学习”，因为它一旦超参数 \(\alpha_t\) 被定义就确定了。因此，一致性项是许多 KL 散度项之和，其中第 \(t\) 项是

$$
D_{\mathrm{KL}}\bigl(\underbrace{q_\phi(x_{t-1}\mid x_t, x_0)}_{\text{没有什么可学}} \,\|\, \underbrace{p_\theta(x_{t-1}\mid x_t)}_{\text{需要做些事}}\bigr). \tag{2.33}
$$

因此，要计算 KL 散度，我们需要对 \(p_\theta(x_{t-1}\mid x_t)\) 做些事。

这里的大想法是 \(q_\phi(x_{t-1}\mid x_t, x_0)\) 是高斯。如果我们想快速计算 KL 散度，那么 \(p_\theta(x_{t-1}\mid x_t)\) 也是高斯会很好。所以我们选择 \(p_\theta(x_{t-1}\mid x_t)\) 为高斯。此外，我们应当匹配均值和方差的形式！因此我们定义

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\bigl(x_{t-1} \mid \underbrace{\mu_\theta(x_t)}_{\text{神经网络}},\, \sigma_q^2(t) I\bigr), \tag{2.34}
$$

其中我们假设均值向量可以用神经网络确定。至于方差，我们选择方差为 \(\sigma_q^2(t)\)。这与式 (2.26) 相同！因此，如果把式 (2.24) 与 \(p_\theta(x_{t-1}\mid x_t)\) 并排放置，我们注意到二者之间的平行关系：

$$
q_\phi(x_{t-1} \mid x_t, x_0)
= \mathcal{N}\bigl(x_{t-1} \mid \underbrace{\mu_q(x_t, x_0)}_{\text{已知}},\, \underbrace{\sigma_q^2(t)I}_{\text{已知}}\bigr).
\tag{2.35}
$$

$$
p_\theta(x_{t-1} \mid x_t)
= \mathcal{N}\bigl(x_{t-1} \mid \underbrace{\mu_\theta(x_t)}_{\text{神经网络}},\, \underbrace{\sigma_q^2(t)I}_{\text{已知}}\bigr).
\tag{2.36}
$$

因此 KL 散度简化为

$$
\begin{aligned}
D_{\mathrm{KL}}\bigl(q_\phi(x_{t-1}\mid x_t, x_0)\,\|\, p_\theta(x_{t-1}\mid x_t)\bigr)
&= D_{\mathrm{KL}}\bigl(\mathcal{N}(x_{t-1}\mid \mu_q(x_t, x_0), \sigma_q^2(t)I)\,\|\, \mathcal{N}(x_{t-1}\mid \mu_\theta(x_t), \sigma_q^2(t)I)\bigr) \\
&= \frac{1}{2\sigma_q^2(t)}\, \lVert \mu_q(x_t, x_0) - \mu_\theta(x_t)\rVert^2,
\end{aligned}
\tag{2.37}
$$

其中我们用了如下事实：两个方差相同的高斯之间的 KL 散度就是两个均值向量之间欧氏距离的平方。

把式 (2.37) 代入式 (2.21) 中 ELBO 的定义，我们可以把 ELBO 改写如下。

**定理 2.6.** 式 (2.21) 中变分扩散模型的 ELBO 可以简化为

$$
\begin{aligned}
\mathrm{ELBO}_\theta(x)
&= \mathbb{E}_{q(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)]
- \underbrace{D_{\mathrm{KL}}\bigl(q(x_T \mid x_0)\,\|\, p(x_T)\bigr)}_{\text{没有什么可训练}} \\
&\quad - \sum_{t=2}^{T}\mathbb{E}_{q(x_t \mid x_0)}\left[\frac{1}{2\sigma_q^2(t)}\, \lVert \mu_q(x_t, x_0) - \mu_\theta(x_t)\rVert^2\right],
\end{aligned}
\tag{2.38}
$$

其中 \(x = x_0\)，且 \(x_T \sim \mathcal{N}(0, I)\)。

对定理 2.6 的一个注记是下标 \(\phi\) 被丢掉了，因为定理 2.5 中定义的分布 \(q_\phi\) 完全由 \(x_t\) 和 \(x_0\) 刻画。没有什么可学习，因此优化不需要包含 \(\phi\)。正因为如此，我们可以丢掉式 (2.38) 中的 KL 散度项。这给我们留下重建项 \(\mathbb{E}_{q(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)]\) 和转移项 \(\sum_{t=2}^{T}\mathbb{E}_{q(x_t \mid x_0)}\bigl[\frac{1}{2\sigma_q^2(t)}\lVert \mu_q(x_t, x_0) - \mu_\theta(x_t)\rVert^2\bigr]\)。

重建项可以简化，因为

$$
\begin{aligned}
\log p_\theta(x_0 \mid x_1)
&= \log \mathcal{N}\bigl(x_0 \mid \mu_\theta(x_1), \sigma_q^2(1)I\bigr) \\
&= \log\frac{1}{(2\pi\sigma_q^2(1))^{d/2}}\exp\left(-\frac{\lVert x_0 - \mu_\theta(x_1)\rVert^2}{2\sigma_q^2(1)}\right) \\
&= -\frac{\lVert x_0 - \mu_\theta(x_1)\rVert^2}{2\sigma_q^2(1)} - \frac{d}{2}\log\bigl(2\pi\sigma_q^2(1)\bigr).
\end{aligned}
$$

因此，一旦我们知道 \(x_1\)，就可以把它送进网络 \(\mu_\theta(x_1)\) 以返回一个均值估计。该均值估计随后被用来计算似然。

### 2.4 训练与推理

本节我们讨论如何训练变分扩散模型，并把它变成去噪扩散概率模型。

我们从看定理 2.6 中定义的 ELBO 开始。式 (2.38) 表明我们需要找到一个网络 \(\mu_\theta\)，它能以某种方式最小化损失

$$
\frac{1}{2\sigma_q^2(t)}\, \lVert \underbrace{\mu_q(x_t, x_0)}_{\text{已知}} - \underbrace{\mu_\theta(x_t)}_{\text{网络}}\rVert^2. \tag{2.39}
$$

从式 (2.25) 回忆

$$
\mu_q(x_t, x_0) = \frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}}{1-\bar\alpha_t}\, x_t + \frac{(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}\, x_0. \tag{2.40}
$$

我们看到它是 \(x_t\) 和 \(x_0\) 的函数。因此一旦我们知道 \(x_t\) 和 \(x_0\)，\(\mu_q(x_t, x_0)\) 就已知并确定。

感兴趣的对象是 \(\mu_\theta\)。既然 \(\mu_\theta\) 是我们的设计，没有理由我们不能把它定义得更方便。于是这里有一个选择。我们定义

$$
\mu_\theta(x_t)
\stackrel{\mathrm{def}}{=}
\underbrace{\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}}{1-\bar\alpha_t}\, x_t + \frac{(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}\, \hat{x}_\theta(x_t)}_{\text{一个网络，其中 }\hat{x}_\theta\text{ 是另一个网络}}. \tag{2.41}
$$

把式 (2.40) 和式 (2.41) 代入式 (2.39) 将给出

$$
\begin{aligned}
\frac{1}{2\sigma_q^2(t)}\, \lVert \mu_q(x_t, x_0) - \mu_\theta(x_t)\rVert^2
&= \frac{1}{2\sigma_q^2(t)}\left(\frac{(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}\right)^2 \lVert \hat{x}_\theta(x_t) - x_0\rVert^2 \\
&= \frac{1}{2\sigma_q^2(t)}\frac{(1-\alpha_t)^2\bar\alpha_{t-1}}{(1-\bar\alpha_t)^2}\, \lVert \hat{x}_\theta(x_t) - x_0\rVert^2.
\end{aligned}
\tag{2.42}
$$

因此 ELBO 可以写成

$$
\mathrm{ELBO}_\theta(x)
= \mathbb{E}_{q(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)]
- \sum_{t=2}^{T}\mathbb{E}_{q(x_t \mid x_0)}\left[\frac{1}{2\sigma_q^2(t)}\frac{(1-\alpha_t)^2\bar\alpha_{t-1}}{(1-\bar\alpha_t)^2}\, \lVert \hat{x}_\theta(x_t) - x_0\rVert^2\right], \tag{2.43}
$$

其中我们丢掉了项 \(D_{\mathrm{KL}}(q(x_T \mid x_0)\,\|\, p(x_T))\)。

接下来，我们想简化 ELBO，以便把 \(\mathbb{E}_{q(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)]\) 吸收进求和。下面是结果。

**定理 2.7.** 去噪扩散概率模型的 ELBO 是

$$
\mathrm{ELBO}_\theta(x)
= -\sum_{t=1}^{T}\frac{1}{2\sigma_q^2(t)}\frac{(1-\alpha_t)^2\bar\alpha_{t-1}}{(1-\bar\alpha_t)^2}\, \mathbb{E}_{q(x_t \mid x_0)}\bigl[\lVert \hat{x}_\theta(x_t) - x_0\rVert^2\bigr]. \tag{2.44}
$$

**证明.** 把式 (2.42) 代入式 (2.38)，我们可以看到

$$
\begin{aligned}
\mathrm{ELBO}_\theta(x)
&= \mathbb{E}_{q(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)]
- \sum_{t=2}^{T}\mathbb{E}_{q(x_t \mid x_0)}\left[\frac{1}{2\sigma_q^2(t)}\, \lVert \mu_q(x_t, x_0) - \mu_\theta(x_t)\rVert^2\right] \\
&= \mathbb{E}_{q(x_1 \mid x_0)}[\log p_\theta(x_0 \mid x_1)]
- \sum_{t=2}^{T}\mathbb{E}_{q(x_t \mid x_0)}\left[\frac{1}{2\sigma_q^2(t)}\frac{(1-\alpha_t)^2\bar\alpha_{t-1}}{(1-\bar\alpha_t)^2}\, \lVert \hat{x}_\theta(x_t) - x_0\rVert^2\right].
\end{aligned}
\tag{2.45}
$$

第一项是

$$
\begin{aligned}
\log p_\theta(x_0 \mid x_1)
&= \log \mathcal{N}(x_0 \mid \mu_\theta(x_1), \sigma_q^2(1)I)
\propto -\frac{1}{2\sigma_q^2(1)}\, \lVert \mu_\theta(x_1) - x_0\rVert^2
&& \text{（定义）} \\
&= -\frac{1}{2\sigma_q^2(1)}\left\lVert \frac{(1-\bar\alpha_0)\sqrt{\alpha_1}}{1-\bar\alpha_1}\, x_1 + \frac{(1-\alpha_1)\sqrt{\bar\alpha_0}}{1-\bar\alpha_1}\, \hat{x}_\theta(x_1) - x_0\right\rVert^2
&& \text{（回忆 }\bar\alpha_0 = 1\text{）} \\
&= -\frac{1}{2\sigma_q^2(1)}\frac{(1-\alpha_1)}{1-\bar\alpha_1}\, \text{（化简后）}\ \lVert \hat{x}_\theta(x_1) - x_0\rVert^2 \\
&= -\frac{1}{2\sigma_q^2(1)}\, \lVert \hat{x}_\theta(x_1) - x_0\rVert^2
&& \text{（回忆 }\bar\alpha_1 = \alpha_1\text{）}.
\end{aligned}
\tag{2.46}
$$

把式 (2.46) 代入式 (2.45) 将把 ELBO 简化为

$$
\mathrm{ELBO}_\theta(x)
= -\sum_{t=1}^{T}\mathbb{E}_{q(x_t \mid x_0)}\left[\frac{1}{2\sigma_q^2(t)}\frac{(1-\alpha_t)^2\bar\alpha_{t-1}}{(1-\bar\alpha_t)^2}\, \lVert \hat{x}_\theta(x_t) - x_0\rVert^2\right].
$$

式 (2.44) 中定义的损失函数非常直观。忽略常数和期望，对一个特定的 \(x_t\)，感兴趣的主要对象是

$$
\operatorname*{argmin}_\theta\, \lVert \hat{x}_\theta(x_t) - x_0\rVert^2.
$$

这不过是一个去噪问题，因为我们需要找到一个网络 \(\hat{x}_\theta\)，使去噪后的图像 \(\hat{x}_\theta(x_t)\) 接近真值 \(x_0\)。使它不是典型去噪器的原因如下：

- \(\mathbb{E}_{q(x_t \mid x_0)}\)：我们不是试图去噪任意随机噪声图像。相反，我们小心地把噪声图像选为

$$
x_t \sim q(x_t \mid x_0) = \mathcal{N}\bigl(x_t \mid \sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t)I\bigr)
\iff
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon_t,
\quad \text{其中 } \epsilon_t \sim \mathcal{N}(0, I).
$$

- \(\frac{1}{2\sigma_q^2(t)}\frac{(1-\alpha_t)^2\bar\alpha_{t-1}}{(1-\bar\alpha_t)^2}\)：我们并不对所有步骤等权地加权去噪损失。相反，有一个调度器来控制每个去噪损失的相对强调。考虑到这一点，并用蒙特卡洛近似期望，我们可以把优化问题写成

$$
\begin{aligned}
&\operatorname*{argmax}_\theta \sum_{x_0 \in \mathcal{X}} \mathrm{ELBO}(x_0) \\
&= \operatorname*{argmin}_\theta \sum_{x_0 \in \mathcal{X}}\sum_{t=1}^{T}
\mathbb{E}_{q(x_t \mid x_0)}\left[
\frac{1}{2\sigma_q^2(t)}\frac{(1-\alpha_t)^2\bar\alpha_{t-1}}{(1-\bar\alpha_t)^2}
\bigl\lVert \hat{x}_\theta\bigl(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon_t\bigr) - x_0\bigr\rVert^2
\right] \\
&= \operatorname*{argmin}_\theta \sum_{x_0 \in \mathcal{X}}\sum_{t=1}^{T}\frac{1}{M}\sum_{m=1}^{M}
\frac{1}{2\sigma_q^2(t)}\frac{(1-\alpha_t)^2\bar\alpha_{t-1}}{(1-\bar\alpha_t)^2}
\bigl\lVert \hat{x}_\theta\bigl(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon_t^{(m)}\bigr) - x_0\bigr\rVert^2,
\end{aligned}
\tag{2.47}
$$

其中 \(\epsilon_t^{(m)} \sim \mathcal{N}(0, I)\)，对 \(x_0 \in \mathcal{X}\) 求和意味着我们考虑训练集 \(\mathcal{X}\) 中的所有样本。可以看到，训练这个模型涉及训练一个去噪器 \(\hat{x}_\theta(\cdot)\)。因此，所得模型被称为去噪扩散概率模型（denoising diffusion probabilistic model，DDPM）。

**DDPM 中的前向扩散。** 训练 DDPM 涉及两条并行分支。第一条分支是前向扩散。前向扩散的目标是通过

$$
x_t \sim q(x_t \mid x_0) = \mathcal{N}\bigl(x_t \mid \sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t)I\bigr), \qquad t = 1,\ldots,T-1
$$

生成中间变量 \(x_1,\ldots,x_{T-1}\)。

**图 2.12：** 前向扩散过程。

前向扩散不需要任何训练。如果你给我们干净图像 \(x_0\)，我们可以运行前向扩散并准备图像 \(x_1,\ldots,x_T\)。图示见图 2.12。

**训练 DDPM。** 一旦训练样本 \(x_0,\ldots,x_T\) 准备好，我们就可以训练 DDPM。DDPM 的训练由式 (2.47) 中的优化和图 2.13 总结。目标是为所有噪声水平训练一个去噪器。它是一个适用于所有噪声水平的去噪器，因为每个 \(x_t\) 有不同的方差 \(1-\bar\alpha_t\)。我们不感兴趣于训练许多去噪器，因为计算上不可行。

**图 2.13：** 去噪扩散概率模型的训练。对同一个神经网络 \(\hat{x}_\theta\)，我们把噪声输入 \(x_t\) 送进网络。损失的梯度被反向传播以更新网络。注意噪声图像不是任意的。它们按照前向采样过程生成。

训练去噪器与任何传统监督学习没有不同。给定一对干净图像和噪声图像，在我们的情形中是 \(x_0\) 和 \(x_t\)，我们训练去噪器 \(\hat{x}_\theta(\cdot)\)。式 (2.47) 中的训练损失有三重求和。如果我们运行随机梯度下降，可以把上述优化简化为下面的过程。

**DDPM 的训练算法。** 对训练数据集中的每一幅图像 \(x_0\)：

- 重复下列步骤直到收敛。
- 选取一个随机时间戳 \(t \sim \mathrm{Uniform}[1, T]\)。
- 抽取样本 \(x_t^{(m)} \sim \mathcal{N}(x_t \mid \sqrt{\bar\alpha_t}\, x_0, (1-\bar\alpha_t)I)\)，即

$$
x_t^{(m)} = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon_t^{(m)}, \qquad \epsilon_t^{(m)} \sim \mathcal{N}(0, I).
$$

- 对下式取梯度下降步

$$
\nabla_\theta\left\{\frac{1}{M}\sum_{m=1}^{M}\bigl\lVert \hat{x}_\theta(x_t^{(m)}) - x_0\bigr\rVert^2\right\}.
$$

你可以按批来做，就像训练任何其他神经网络一样。注意，这里你是在为所有噪声条件训练一个去噪网络 \(\hat{x}_\theta\)。

**DDPM 的推理——反向扩散。** 一旦去噪器 \(\hat{x}_\theta\) 训练完成，我们就可以用它做推理。推理是关于在状态序列 \(x_T, x_{T-1}, \ldots, x_1\) 上从分布 \(p_\theta(x_{t-1} \mid x_t)\) 采样图像。由于它是反向扩散过程，我们需要通过下式递归地进行：

$$
x_{t-1} \sim p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\bigl(x_{t-1} \mid \mu_\theta(x_t),\, \sigma_q^2(t)I\bigr).
$$

通过重参数化，我们有

$$
\begin{aligned}
x_{t-1}
&= \mu_\theta(x_t) + \sigma_q(t)\epsilon, \qquad \text{其中 } \epsilon \sim \mathcal{N}(0, I) \\
&= \frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}}{1-\bar\alpha_t}\, x_t + \frac{(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}\, \hat{x}_\theta(x_t) + \sigma_q(t)\epsilon.
\end{aligned}
$$

这导致下面的推理算法。用白话说，我们从白噪声向量 \(x_T\) 顺序地运行去噪器 \(T\) 次，回到生成图像 \(\hat{x}_0\)。图示见图 2.14。

**DDPM 的推理。**

- 你给我们一个白噪声向量 \(x_T \sim \mathcal{N}(0, I)\)。
- 对 \(t = T, T-1, \ldots, 1\) 重复下列步骤。
- 我们用训练好的去噪器计算 \(\hat{x}_\theta(x_t)\)。
- 按照下式更新

$$
x_{t-1}
= \frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}}{1-\bar\alpha_t}\, x_t
+ \frac{(1-\alpha_t)\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}\, \hat{x}_\theta(x_t)
+ \sigma_q(t)\epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I). \tag{2.48}
$$

**图 2.14：** 去噪扩散概率模型的推理。

### 2.5 预测噪声

我们推导的最后一步是把结果接回 Ho 等人的原始论文 [16]，使记号与文献更一致。

**训练。** 如果你熟悉去噪文献，你大概知道预测噪声而不是预测信号的残差型算法。同样的精神适用于去噪扩散，我们可以学习预测噪声。要看为什么如此，我们考虑式 (2.11)。如果重排项，将得到

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon_0
\quad\Longrightarrow\quad
x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_0}{\sqrt{\bar\alpha_t}}.
$$

把这代入 \(\mu_q(x_t, x_0)\)，我们可以证明

$$
\begin{aligned}
\mu_q(x_t, x_0)
&= \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\, x_t + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\, x_0}{1-\bar\alpha_t} \\
&= \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\, x_t + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\cdot \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_0}{\sqrt{\bar\alpha_t}}}{1-\bar\alpha_t} \\
&= \text{再做几步我们将跳过的代数} \\
&= \frac{1}{\sqrt{\alpha_t}}\, x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}\,\sqrt{\alpha_t}}\, \epsilon_0.
\end{aligned}
\tag{2.49}
$$

用文字说，我们把 \(\mu_q(x_t, x_0)\) 从 \(x_0\) 的函数转换成了 \(\epsilon_0\) 的函数。

既然我们做了这一修改，自然也应修改均值估计器 \(\mu_\theta\)。为了使 \(\mu_\theta\) 的形式与 \(\mu_q(x_t, x_0)\) 匹配，我们选择

$$
\mu_\theta(x_t) = \frac{1}{\sqrt{\alpha_t}}\, x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}\,\sqrt{\alpha_t}}\, \hat{\epsilon}_\theta(x_t). \tag{2.50}
$$

把式 (2.49) 和式 (2.50) 代入式 (2.39) 将给我们一个新的 ELBO

$$
\begin{aligned}
\mathrm{ELBO}_\theta(x_0, \epsilon_0)
&= -\sum_{t=1}^{T}\mathbb{E}_{q(x_t \mid x_0)}\left[\frac{1}{2\sigma_q^2(t)}\frac{(1-\alpha_t)^2}{(1-\bar\alpha_t)\alpha_t}\, \lVert \hat{\epsilon}_\theta(x_t) - \epsilon_0\rVert^2\right] \\
&= -\sum_{t=1}^{T}\mathbb{E}_{q(x_t \mid x_0)}\left[\frac{1}{2\sigma_q^2(t)}\frac{(1-\alpha_t)^2}{(1-\bar\alpha_t)\alpha_t}
\left\lVert \hat{\epsilon}_\theta\bigl(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon_0\bigr) - \epsilon_0\right\rVert^2\right].
\end{aligned}
$$

我们注记这个 ELBO 是 \(x_0\) 和 \(\epsilon_0\) 的函数。因此，要训练去噪器，我们需要求解优化

$$
\operatorname*{argmin}_\theta\, \mathbb{E}_{x_0, \epsilon_0}[\mathrm{ELBO}_\theta(x_0, \epsilon_0)]
\approx \operatorname*{argmin}_\theta \sum_{x_0 \sim \mathcal{X}}\frac{1}{M}\sum_{m=1}^{M}\mathrm{ELBO}_\theta(x_0, \epsilon_0^{(m)}),
$$

其中上标表示第 \(m\) 个初始噪声项。如果我们运行随机梯度下降，可以把上述描述简化为下面的过程。

**用 \(\hat{\epsilon}_\theta(x_t)\) 训练 DDPM。** 对训练数据集中的每一幅图像 \(x_0\)：

- 重复下列步骤直到收敛。
- 选取一个随机时间戳 \(t \sim \mathrm{Uniform}[1, T]\)。
- 抽取样本 \(\epsilon_0 \sim \mathcal{N}(0, I)\)。
- 抽取样本 \(x_t \sim \mathcal{N}(x_t \mid \sqrt{\bar\alpha_t}\, x_0, (1-\bar\alpha_t)I)\)，即

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon_0.
$$

- 对下式取梯度下降步

$$
\nabla_\theta\, \lVert \hat{\epsilon}_\theta(x_t) - \epsilon_0\rVert^2.
$$

**推理。** 对推理，我们知道

$$
x_{t-1} \sim p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\bigl(x_{t-1} \mid \mu_\theta(x_t),\, \sigma_q(t)^2 I\bigr).
$$

利用重参数化，我们可以证明

$$
\begin{aligned}
x_{t-1}
&= \mu_\theta(x_t) + \sigma_q(t) z, \qquad \text{其中 } z \sim \mathcal{N}(0, I) \\
&= \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\, \hat{\epsilon}_\theta(x_t)\right) + \sigma_q(t) z.
\end{aligned}
\tag{2.51}
$$

总结在这里，我们有

**用 \(\hat{\epsilon}_\theta(x_t)\) 做 DDPM 推理。**

- 你给我们一个白噪声向量 \(x_T \sim \mathcal{N}(0, I)\)。
- 对 \(t = T, T-1, \ldots, 1\) 重复下列步骤。
- 我们用训练好的去噪器计算 \(\hat{\epsilon}_\theta(x_t)\)。（原文此处写了 \(\hat{x}_\theta(x_t)\)，与噪声预测形式对应的是 \(\hat{\epsilon}_\theta\)。）
- 按照下式更新

$$
x_{t-1}
= \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\, \hat{\epsilon}_\theta(x_t)\right) + \sigma_q(t) z,
\qquad z \sim \mathcal{N}(0, I). \tag{2.52}
$$

### 2.6 去噪扩散隐式模型（DDIM）

**从 DDPM 到 DDIM。** DDPM 最普遍的缺点之一是，它们需要大量迭代才能生成一张看起来相当好的图像。如 Song 等人在 [39] 中提到的，一个 DDPM 在标准 GPU 上生成 5 万张（大小为 \(256\times 256\) 的）图像需要超过 1000 小时。原因是在运行反向扩散步骤时，我们需要执行去噪。如果反向扩散过程本质上需要许多步才能收敛，那么我们就要做许多次去噪步骤。因此，要加快计算，必须减少迭代次数。DDIM 是为克服这一困难而发明的。

从式 (2.1) 回忆，原始 DDPM 转移概率取如下形式

$$
q(x_t \mid x_{t-1}) \stackrel{\mathrm{def}}{=} \mathcal{N}\bigl(x_t \mid \sqrt{\alpha_t}\, x_{t-1},\, (1-\alpha_t)I\bigr).
$$

此外，式 (2.8) 表明给定 \(x_0\) 时 \(x_t\) 的概率是

$$
q(x_t \mid x_0) = \mathcal{N}\bigl(x_t \mid \sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t)I\bigr),
$$

其中对任何递减序列 \(0 < \alpha_t \le 1\) 有 \(\bar\alpha_t = \prod_{i=1}^{t}\alpha_i\)。这里一个重要观察是，转移概率 \(q(x_t \mid x_{t-1})\) 服从马尔可夫链，意味着 \(x_t\) 的概率纯粹依赖于 \(x_{t-1}\)，而不依赖于更早的状态 \(x_{t-2}\) 等等。马尔可夫结构的优点是系统无记忆。一旦我们知道 \(x_{t-1}\)，就知道 \(x_t\)。但缺点是马尔可夫链可能需要许多步才能收敛。DDIM 通过离开马尔可夫结构、走向非马尔可夫来克服这一问题。

**DDIM 中的概率分布。** 为开始讨论，让我们跟随 [39]，选取一组特殊参数，把 \(\alpha_t\) 换成比值 \(\alpha_t / \alpha_{t-1}\)。这意味着

$$
q(x_t \mid x_{t-1})
\stackrel{\mathrm{def}}{=}
\mathcal{N}\!\left(x_t \,\middle|\, \sqrt{\frac{\alpha_t}{\alpha_{t-1}}}\, x_{t-1},\, \left(1 - \frac{\alpha_t}{\alpha_{t-1}}\right)I\right).
$$

这一选择没有特别强的物理意义，只是它使记号更简单。有了这一选择，乘积项简化为 \(\bar\alpha_t = \prod_{i=1}^{t}\alpha_i/\alpha_{i-1} = \alpha_t\)，假设 \(\alpha_0 = 1\)。因此

$$
q(x_t \mid x_0) = \mathcal{N}\bigl(x_t \mid \sqrt{\alpha_t}\, x_0,\, (1-\alpha_t)I\bigr). \tag{2.53}
$$

用重参数化表达式 (2.53)，我们注意到式 (2.53) 中的 \(x_t\) 可以用 \(x_0\) 表示为

$$
x_t = \sqrt{\alpha_t}\, x_0 + \sqrt{1-\alpha_t}\,\epsilon, \qquad \text{其中 } \epsilon \sim \mathcal{N}(0, I).
$$

由同样的论证，我们可以写

$$
x_{t-1} = \sqrt{\alpha_{t-1}}\, x_0 + \sqrt{1-\alpha_{t-1}}\,\epsilon, \qquad \text{其中 } \epsilon \sim \mathcal{N}(0, I). \tag{2.54}
$$

于是这里来了一个有趣的技巧。让我们把 \(\epsilon\) 换成某种东西，使 \(x_{t-1}\) 不再是被白噪声扰动的 \(x_0\)。也许我们可以考虑下面的推导

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t}\, x_0 + \sqrt{1-\alpha_t}\,\epsilon \\
&\Longrightarrow \sqrt{1-\alpha_t}\,\epsilon = x_t - \sqrt{\alpha_t}\, x_0 \\
&\Longrightarrow \epsilon = \frac{x_t - \sqrt{\alpha_t}\, x_0}{\sqrt{1-\alpha_t}}.
\end{aligned}
$$

于是，把 \(\epsilon\) 代入式 (2.54)，我们得到

$$
\begin{aligned}
x_{t-1}
&= \sqrt{\alpha_{t-1}}\, x_0 + \sqrt{1-\alpha_{t-1}}\,\epsilon \\
&= \sqrt{\alpha_{t-1}}\, x_0 + \sqrt{1-\alpha_{t-1}}\left(\frac{x_t - \sqrt{\alpha_t}\, x_0}{\sqrt{1-\alpha_t}}\right).
\end{aligned}
\tag{2.55}
$$

式 (2.55) 与式 (2.54) 的区别在于，在式 (2.54) 中噪声项是 \(\epsilon\)，它是 \(\mathcal{N}(0, I)\)。正是这个高斯使 DDPM 的推导容易，但也正是这个高斯使反向扩散缓慢。相比之下，式 (2.55) 用一个估计替换了高斯。这个估计使用先前信号 \(x_t\) 与初始信号 \(x_0\) 的组合。当然，人们可以争辩说 DDPM（例如式 (2.24)）也使用 \(x_t\) 与 \(x_0\) 的组合。区别在于式 (2.55) 中的组合允许我们做式 (2.24) 做不到的事，即推导边缘分布 \(q(x_{t-1} \mid x_0)\) 并把它变成想要的形式。

让我们更详细地阐述边缘分布。参照式 (2.55)，我们注意到可以选择

$$
q(x_{t-1} \mid x_t, x_0)
= \mathcal{N}\!\left(
\sqrt{\alpha_{t-1}}\, x_0 + \sqrt{1-\alpha_{t-1}}\left(\frac{x_t - \sqrt{\alpha_t}\, x_0}{\sqrt{1-\alpha_t}}\right),
\ \text{某个东西}
\right),
$$

其中“某个东西”代表高斯的方差，它可以做成 \(\sigma_t^2 I\)，对某个超参数 \(\sigma_t\)。DDIM 中一个重要（很可能是最重要）的论证是，我们希望边缘分布 \(q(x_{t-1} \mid x_0)\) 与 \(q(x_t \mid x_0)\) 具有相同形式：

$$
q(x_{t-1} \mid x_0) = \mathcal{N}\bigl(\sqrt{\alpha_{t-1}}\, x_0,\, (1-\alpha_{t-1})I\bigr).
$$

瞄准这一分布的原因是，我们最终关心的是边缘分布 \(q(x_t \mid x_0)\)，我们希望它在 \(t = T\) 时变成纯白噪声，在 \(t = 0\) 时是原始图像。因此，虽然我们可以有成百万种不同的转移分布 \(q(x_{t-1} \mid x_t, x_0)\) 选择，只有某些非常专门的转移概率能确保 \(q(x_{t-1} \mid x_0)\) 取我们喜欢的形式。

**转移分布的推导。** 有了这一目标，我们现在陈述数学问题。假设

$$
\begin{aligned}
q(x_t \mid x_0) &= \mathcal{N}\bigl(\sqrt{\alpha_t}\, x_0,\, (1-\alpha_t)I\bigr), \\
q(x_{t-1} \mid x_t, x_0)
&= \mathcal{N}\!\left(
\sqrt{\alpha_{t-1}}\, x_0 + \sqrt{1-\alpha_{t-1}}\left(\frac{x_t - \sqrt{\alpha_t}\, x_0}{\sqrt{1-\alpha_t}}\right),
\ \sigma_t^2 I
\right),
\end{aligned}
\tag{2.56}
$$

我们能否确保 \(q(x_{t-1} \mid x_0) = \mathcal{N}(\sqrt{\alpha_{t-1}}\, x_0, (1-\alpha_{t-1})I)\)？如果不能，还需要哪些额外改变？

这个数学问题的答案需要教科书中的一些工具。我们回忆 Bishop 教科书 [4] 中的下列结果。

**定理 2.8. Bishop [4, 式 2.115]。** 设两个随机变量 \(x\) 和 \(y\) 服从分布

$$
p(x) = \mathcal{N}(\mu, \Lambda^{-1}), \qquad p(y \mid x) = \mathcal{N}(Ax + b, L^{-1}).
$$

则我们可以证明边缘分布是

$$
p(y) = \int p(y \mid x)\, p(x)\, dx = \mathcal{N}\bigl(A\mu + b,\, L^{-1} + A\Lambda^{-1}A^\top\bigr).
$$

让我们看如何把这一结果应用到我们的问题。看式 (2.56)，我们可以识别出下列量：

$$
A = \sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}}, \quad
\mu = \sqrt{\alpha_t}\, x_0, \quad
b = \sqrt{\alpha_{t-1}}\, x_0 - \sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}}\, \sqrt{\alpha_t}\, x_0.
$$

设 \(q(x_{t-1} \mid x_0) = \mathcal{N}(\mu_{t-1}, \sigma_{t-1}^2 I)\)，其中均值 \(\mu_{t-1}\) 和方差 \(\sigma_{t-1}^2\) 的选择未知。如果我们能证明 \(\mu_{t-1} = \sqrt{\alpha_{t-1}}\, x_0\) 且 \(\sigma_{t-1}^2 = 1-\alpha_{t-1}\)，就完成了。为此我们证明

$$
\begin{aligned}
\mu_{t-1}
&= A\mu + b \\
&= \sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}}\cdot \sqrt{\alpha_t}\, x_0 + \sqrt{\alpha_{t-1}}\, x_0 - \sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}}\, \sqrt{\alpha_t}\, x_0 \\
&= \sqrt{\alpha_{t-1}}\, x_0.
\end{aligned}
$$

哦，好消息！我们证明了 \(\mu_{t-1} = \sqrt{\alpha_{t-1}}\, x_0\)。这意味着对我们所选择的转移分布 \(q(x_{t-1} \mid x_t, x_0)\)，边缘分布具有所期望的均值。

于是剩下检查方差。我们证明

$$
\begin{aligned}
\sigma_{t-1}^2
&= L^{-1} + A\Lambda^{-1}A^\top \\
&= \sigma_t^2 + \sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}}\cdot (1-\alpha_t)\cdot \sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}} \\
&= \sigma_t^2 + (1-\alpha_{t-1}).
\end{aligned}
$$

哦，不！我们不能证明 \(\sigma_{t-1}^2 = 1-\alpha_{t-1}\)。这里多了一项 \(\sigma_t^2\)。但这不是什么大事。我们何不做一个快速修补，把 \(\sigma_t^2\) 加进 \(A\)：

$$
\begin{aligned}
\sigma_{t-1}^2
&= \sigma_t^2 + \sqrt{\frac{1-\alpha_{t-1}-\sigma_t^2}{1-\alpha_t}}\cdot (1-\alpha_t)\cdot \sqrt{\frac{1-\alpha_{t-1}-\sigma_t^2}{1-\alpha_t}} \\
&= \sigma_t^2 + 1 - \alpha_{t-1} - \sigma_t^2 \\
&= 1-\alpha_{t-1}.
\end{aligned}
$$

啊哈！\(\sigma_{t-1}^2\) 现在取所期望的形式，使得 \(\sigma_{t-1}^2 = 1-\alpha_{t-1}\)。让我们快速检查，确保这个额外的 \(\sigma_t^2\) 不影响均值：

$$
\begin{aligned}
\mu_{t-1}
&= A\mu + b \\
&= \sqrt{\frac{1-\alpha_{t-1}-\sigma_t^2}{1-\alpha_t}}\cdot \sqrt{\alpha_t}\, x_0
+ \sqrt{\alpha_{t-1}}\, x_0
- \sqrt{\frac{1-\alpha_{t-1}-\sigma_t^2}{1-\alpha_t}}\, \sqrt{\alpha_t}\, x_0 \\
&= \sqrt{\alpha_{t-1}}\, x_0.
\end{aligned}
$$

因此，尽管我们改变了方差项，均值仍保持所期望的形式。

总结起来，我们可以把 \(q(x_{t-1} \mid x_t, x_0)\) 选择为如下。

**定理 2.9. DDIM 转移分布。** 在 DDIM 中，转移分布定义为

$$
q(x_{t-1} \mid x_t, x_0)
= \mathcal{N}\!\left(
\sqrt{\alpha_{t-1}}\, x_0 + \sqrt{1-\alpha_{t-1}-\sigma_t^2}\left(\frac{x_t - \sqrt{\alpha_t}\, x_0}{\sqrt{1-\alpha_t}}\right),
\ \sigma_t^2 I
\right). \tag{2.57}
$$

若 \(q(x_t \mid x_0) = \mathcal{N}(\sqrt{\alpha_t}\, x_0, (1-\alpha_t)I)\)，则由我们的推导可得 \(q(x_{t-1} \mid x_0) = \mathcal{N}(\sqrt{\alpha_{t-1}}\, x_0, (1-\alpha_{t-1})I)\)。

**DDIM 的推理。** DDIM 的推理基于转移分布推导。从前向过程开始，如果我们想执行反向，需要从下列方程找出 \(x_0\)：

$$
\underbrace{x_t}_{\text{给定}}
= \underbrace{\sqrt{\alpha_t}\, x_0}_{\text{想要找出}}
+ \underbrace{\sqrt{1-\alpha_t}\,\epsilon}_{\text{由网络估计}}.
$$

重排项，我们看到

$$
x_0 = \frac{1}{\sqrt{\alpha_t}}\bigl(x_t - \sqrt{1-\alpha_t}\,\epsilon\bigr)
\quad\Longrightarrow\quad
f_\theta^{(t)}(x_t) \stackrel{\mathrm{def}}{=} \frac{1}{\sqrt{\alpha_t}}\bigl(x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta^{(t)}(x_t)\bigr).
$$

这个方程中有两个新项。第一个是 \(\epsilon_\theta^{(t)}(x_t)\)，它替换 \(\epsilon\)。它是基于当前输入 \(x_t\) 的噪声估计。第二项是 \(f_\theta^{(t)}(x_t)\)，定义为 \(\frac{1}{\sqrt{\alpha_t}}\bigl(x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta^{(t)}(x_t)\bigr)\)。我们可以把它想成真实信号 \(x_0\) 的估计。

回到转移分布，我们回忆它记为 \(q(x_{t-1} \mid x_t, x_0)\)。如果我们得不到 \(x_0\)，可以用 \(f_\theta^{(t)}(x_t)\) 替换它。这意味着

$$
\begin{aligned}
p_\theta(x_{t-1} \mid x_t) &= q(x_{t-1} \mid x_t, x_0) \\
&\stackrel{\mathrm{def}}{\Longrightarrow}
p_\theta(x_{t-1} \mid x_t) = q\bigl(x_{t-1} \mid x_t, f_\theta^{(t)}(x_t)\bigr) \\
&= \mathcal{N}\!\left(
\sqrt{\alpha_{t-1}}\, f_\theta^{(t)}(x_t)
+ \sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot \frac{x_t - \sqrt{\alpha_t}\, f_\theta^{(t)}(x_t)}{\sqrt{1-\alpha_t}},
\ \sigma_t^2 I
\right) \\
&= \mathcal{N}\!\left(
\sqrt{\alpha_{t-1}}\cdot \frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}}
+ \sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot \epsilon_\theta^{(t)}(x_t),
\ \sigma_t^2 I
\right).
\end{aligned}
\tag{2.58}
$$

对 \(t = 1\) 的特殊情形，我们定义 \(p_\theta(x_{0} \mid x_1) = \mathcal{N}(f_\theta^{(1)}(x_1), \sigma_1^2 I)\)，使反向过程处处有支撑。看式 (2.58)，我们用重参数化把它写成如下，并按照 [39] 解释该方程。

$$
\text{（DDIM）}\quad
x_{t-1}
= \underbrace{\sqrt{\alpha_{t-1}}\left(\frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}}\right)}_{\text{预测的 } x_0}
+ \underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot \epsilon_\theta^{(t)}(x_t)}_{\text{指向 } x_t \text{ 的方向}}
+ \underbrace{\sigma_t \epsilon_t}_{\sim \mathcal{N}(0, I)}. \tag{2.59}
$$

把这个方程与式 (2.52) 中的 DDPM 方程比较会有帮助：

$$
\text{（DDPM）}\quad
x_{t-1}
= \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\alpha_t}}\, \epsilon_\theta^{(t)}(x_t)\right) + \sigma_t \epsilon_t,
\qquad \epsilon_t \sim \mathcal{N}(0, I). \tag{2.60}
$$

DDPM 与 DDIM 的主要差别是微妙的。虽然二者都在更新中使用 \(x_t\) 和 \(\epsilon^{(t)}(x_t)\)，具体的更新公式造成不同的收敛速度。事实上，后来在微分方程文献中，人们把 DDIM 和 DDPM 与随机微分方程联系起来，观察到 DDIM 在求解微分方程时采用了某些特殊的加速一阶数值格式。

### 2.7 结语

DDPM 的文献正在迅速膨胀。Sohl-Dickstein 等人 [38] 和 Ho 等人 [16] 的原始论文是理解这一主题的必读文献。对一个更“用户友好”的版本，我们发现 Luo 的教程非常有用 [27]。一些后续工作被高度引用，包括 Song 等人的去噪扩散隐式模型 [39]。在应用方面，人们一直把 DDPM 用于各种图像合成应用，例如 [34, 35]。
---

## 3 分数匹配朗之万动力学（SMLD）

基于分数的生成模型 [42] 是从期望分布生成数据的另一类方法。有几个核心成分：朗之万方程（Langevin equation）、（Stein）分数函数，以及分数匹配损失。本节的重点更多在后两者。我们将对朗之万方程做一些不那么严格的论证，并解释如何使它对我们的生成任务在计算上可行。关于朗之万方程更深入的讨论将推迟到下一节。

### 3.1 从分布中采样

想象我们给定一个分布 \(p(x)\)，想从 \(p(x)\) 中抽取样本。如果 \(x\) 是一维变量，我们可以通过反演累积分布函数（CDF）[7, 第 4 章]，并把一个 \(\mathrm{Uniform}[0, 1]\) 随机变量送过这个反演后的 CDF，轻易达到目标。对高维分布，同样的 CDF 技术不适用。然而，给概率更高的样本更高权重这一直觉仍然有效。例如，如果我们想从橙子数据集中抽取一个样本，几乎可以肯定我们想要一个球形橙子，而不是一个立方体橙子。

**图 3.1：** 如果我们的目标是生成一幅水果图像，那么应当期望底层分布 \(p(x)\) 对那些“看起来正常”的水果图像比对那些“看起来怪异”的图像有更高的值。因此，要从一个分布中采样，更自然的是从分布更高的位置挑选样本。水果图像取自 https://cognitiveseo.com/blog/4224/unnatural-links-definition-examples/

因此，以一种不严格的方式，我们可以论证：如果给定 \(p(x)\)，我们应当旨在从 \(p(x)\) 取值较高的位置抽取样本。寻找更高概率这一想法可以翻译成一个优化

$$
x^\star = \operatorname*{argmax}_x \log p(x),
$$

其中目标是最大化分布 \(p(x)\) 的对数似然。当然，这一最大化并没有提供如何抽取低概率样本的任何线索，我们将通过朗之万方程的视角来解释。现在，我们想对这里的最大化与最大似然估计之间的差别做一点注记。在最大似然中，数据点 \(x\) 固定，但模型参数在变。这里，模型参数固定，但数据点在变。我们被给定一个固定模型。我们的目标是从这个模型中抽取最可能的样本。下表展示我们的采样问题与最大似然估计之间的差别。

| 问题 | 采样 | 最大似然 |
| --- | --- | --- |
| 优化对象 | 一个样本 \(x\) | 模型参数 \(\theta\) |
| 表述 | \(x^\star = \operatorname*{argmax}_x \log p(x; \theta)\) | \(\theta^\star = \operatorname*{argmax}_\theta \log p(x; \theta)\) |

让我们继续关于最大化的论证。如果 \(p(x)\) 是一个简单的参数模型，最大化会有解析解。然而一般而言，高维空间中的优化是病态的，有许多局部极小。因此没有单一算法在所有情形下全局收敛。在计算复杂度、内存需求、实现难度和解的质量之间的一个合理折中是梯度下降算法，它是一阶方法。对我们的优化，目标函数是 \(\log p(x)\)，梯度下降算法通过更新规则定义一列迭代

$$
x_{t+1} = x_t + \tau \nabla_x \log p(x_t),
$$

其中 \(\nabla_x \log p(x_t)\) 表示 \(\log p(x)\) 在 \(x_t\) 处求值的梯度，\(\tau\) 是步长。这里我们用“\(+\)”而不是典型的“\(-\)”，因为我们在求解一个最大化问题。

如果你同意上述方法，我们现在可以提供关于朗之万方程的一个信息性介绍。不必太担心它在物理学中的根源，我们可以把朗之万方程当作允许我们抽取样本的迭代过程。

**定义 3.1.** 从已知分布 \(p(x)\) 采样的（离散时间）朗之万方程是对 \(t = 1,\ldots,T\) 的迭代过程：

$$
x_{t+1} = x_t + \tau \nabla_x \log p(x_t) + \sqrt{2\tau}\, z, \qquad z \sim \mathcal{N}(0, I), \tag{3.1}
$$

其中 \(\tau\) 是用户可以控制的步长，\(x_0\) 是白噪声。

**例 3.1.** 考虑高斯分布 \(p(x) = \mathcal{N}(x \mid \mu, \sigma^2)\)，我们可以证明朗之万方程是

$$
\begin{aligned}
x_{t+1}
&= x_t + \tau \cdot \nabla_x \log \left(\frac{1}{\sqrt{2\pi\sigma^2}} e^{-(x_t-\mu)^2/(2\sigma^2)}\right) + \sqrt{2\tau}\, z \\
&= x_t - \tau \cdot \frac{x_t - \mu}{\sigma^2} + \sqrt{2\tau}\, z, \qquad z \sim \mathcal{N}(0, 1),
\end{aligned}
$$

其中初始状态可以设为 \(x_0 \sim \mathcal{N}(0, 1)\)。

如果我们忽略噪声项 \(\sqrt{2\tau}\, z\)，式 (3.1) 中的朗之万方程恰好是梯度下降，但是针对一个特定函数——随机变量 \(x\) 的对数似然。因此，虽然梯度下降是针对任何目标函数的通用一阶优化算法，朗之万方程在生成模型的语境中聚焦于分布。梯度下降算法加上一个小的噪声扰动，给了我们一个简单总结。

在生成模型中，

> 朗之万方程 = 梯度下降 + 噪声

应用于对数似然函数。

但为什么我们想要梯度下降加噪声，而不是梯度下降？一种解释是我们并不感兴趣于求解优化问题。相反，我们更感兴趣于从分布中采样。通过把随机噪声引入梯度下降步骤，我们随机挑选一个跟随目标函数轨迹、同时并不停在原地的样本。如果我们更靠近峰值，我们会稍微左右移动。如果我们远离峰值，梯度方向会把我们拉向峰值。如果峰值周围的曲率很尖，我们会把大多数稳态点 \(x_T\) 集中在那里。如果峰值周围的曲率很平，我们会散开。因此，通过反复在均匀分布的位置初始化梯度下降（加噪声）算法，我们最终会收集到跟随我们所指定分布的样本。

一种稍更正式的方式来为朗之万方程辩护是福克–普朗克方程。福克–普朗克方程是随机过程的一个基本结果。对任何马尔可夫过程（例如维纳过程和布朗运动），解 \(x_t\) 的动力学由一个随机微分方程（即朗之万方程）描述。然而，由于 \(x_t\) 在任何时刻 \(t\) 都是随机变量，每个 \(x_t\) 都有一个相关联的底层概率分布 \(p(x, t)\)。福克–普朗克方程提供关于该分布的一个数学陈述。该分布必须满足一个偏微分方程。粗略地说，在我们问题的语境中，福克–普朗克方程可以描述如下。

**定理 3.1.** 在我们问题的语境中，朗之万方程的解 \(x_t\) 在时刻 \(t\) 将有一个概率分布 \(p(x, t)\)，满足福克–普朗克方程

$$
\partial_t p(x, t) = -\partial_x \bigl\{[\partial_x \log p(x)]\, p(x, t)\bigr\} + \partial_x^2 p(x, t), \tag{3.2}
$$

其中 \(\log p(x)\) 是真值分布的对数似然。

推导福克–普朗克方程将花费巨大努力。然而，如果给定一个候选解，验证它是否满足福克–普朗克方程并不难。

**定理的验证。** 假设我们已经把朗之万方程运行得足够久，使得当 \(t \to \infty\) 时达到一个收敛解 \(x_t\)。我们论证这个极限分布是 \(p(x)\)。的确，我们可以证明

$$
\partial_x \log p(x) = \frac{\partial_x p(x)}{p(x)}.
$$

然后，当 \(t \to \infty\) 时用 \(p(x)\) 替换 \(p(x, t)\)，我们可以证明

$$
\begin{aligned}
&-\partial_x\bigl\{[\partial_x \log p(x)]\, p(x)\bigr\} + \partial_x^2 p(x) \\
&= \partial_x\bigl\{-[\partial_x \log p(x)]\, p(x) + \partial_x p(x)\bigr\} \\
&= \partial_x\left\{-\frac{\partial_x p(x)}{p(x)}\, p(x) + \partial_x p(x)\right\} \\
&= \partial_x\bigl\{-\partial_x p(x) + \partial_x p(x)\bigr\} = 0.
\end{aligned}
$$

另一方面，当 \(t \to \infty\) 时，有 \(\partial_t p(x) = 0\)。因此福克–普朗克方程得到验证。

**例 3.2.** 考虑高斯混合 \(p(x) = \pi_1 \mathcal{N}(x \mid \mu_1, \sigma_1^2) + \pi_2 \mathcal{N}(x \mid \mu_2, \sigma_2^2)\)。我们可以解析或数值地计算梯度 \(\nabla_x \log p(x)\)。为演示，我们选择 \(\pi_1 = 0.6\)，\(\mu_1 = 2\)，\(\sigma_1 = 0.5\)，\(\pi_2 = 0.4\)，\(\mu_2 = -2\)，\(\sigma_2 = 0.2\)。我们初始化 \(x_0 = 0\)。我们选择 \(\tau = 0.05\)。我们把上述梯度下降迭代运行 \(T = 500\) 次，并画出 \(t = 1,\ldots,T\) 时值 \(p(x_t)\) 的轨迹。如下图所见，序列 \(\{x_1, x_2, \ldots, x_T\}\) 只是跟随高斯的形状并爬向其中一个峰值。

更有趣的是当我们加上噪声项时。序列 \(x_t\) 不是落在峰值上，而是在峰值周围移动，并在峰值附近某处结束。（注：要终止算法，我们可以逐渐使 \(\tau\) 变小，或者可以提前停止。）

$$
x_{t+1} = x_t + \tau \nabla_x \log p(x_t)
\qquad\text{对比}\qquad
x_{t+1} = x_t + \tau \nabla_x \log p(x_t) + \sqrt{2\tau}\, z.
$$

**图 3.2：** 旨在挑选一个最大化似然的样本的确定性算法，对比在每次迭代加入噪声的随机算法。

图 3.3 展示样本轨迹的一个有趣描述。从一个任意位置开始，数据点 \(x_t\) 将按照朗之万动力学方程做随机游走。随机游走的方向并不完全任意。有一定量预先定义的漂移，同时每一步有某种程度的随机性。漂移由 \(\nabla_x \log p(x)\) 决定，随机性来自 \(z\)。

**图 3.3：** 使用朗之万动力学的样本演化轨迹。为更好可视化，我们把高斯混合的两个模态着成不同颜色。这里的设定与上例相同，只是步长为 \(\tau = 0.001\)。

**例 3.3.** 沿用上例，我们再次考虑高斯混合

$$
p(x) = \pi_1 \mathcal{N}(x \mid \mu_1, \sigma_1^2) + \pi_2 \mathcal{N}(x \mid \mu_2, \sigma_2^2).
$$

我们选择 \(\pi_1 = 0.6\)，\(\mu_1 = 2\)，\(\sigma_1 = 0.5\)，\(\pi_2 = 0.4\)，\(\mu_2 = -2\)，\(\sigma_2 = 0.2\)。假设我们初始化 \(M = 10000\) 个均匀分布的样本 \(x_0 \sim \mathrm{Uniform}[-3, 3]\)。我们把朗之万更新运行 \(t = 100\) 步。生成样本的直方图如下图所示。

**图 3.4：** 由朗之万动力学生成的样本。最初样本均匀分布。随着时间推进，样本的分布变成期望的分布。

**注记 1：随机梯度朗之万动力学。** 由朗之万方程支配的 \(x_t\) 的动力学行为通常称为朗之万动力学。朗之万动力学使用梯度下降加噪声。这与随机梯度下降（stochastic gradient descent，SGD）不同。SGD 用小批量近似全梯度。没有噪声。SGD 中的随机性来自从小批量，它们从训练数据集中均匀采样。SGD 可以与朗之万动力学配对，做成随机梯度朗之万动力学，如 [46] 所概述，该文在经典最大后验（MAP）估计的语境中提供了清楚的比较。

$$
\begin{aligned}
\text{随机梯度下降}\quad
&\Delta \theta_t = \frac{\epsilon_t}{2}\nabla_\theta \log p(\theta_t) + \frac{N}{n}\sum_{i=1}^{n}\nabla \log p(y_{t_i} \mid \theta_t), \\
\text{朗之万动力学}\quad
&\Delta \theta_t = \frac{\epsilon}{2}\nabla_\theta \log p(\theta_t) + \sum_{i=1}^{N}\nabla \log p(y_i \mid \theta_t) + \eta_t, \\
\text{随机梯度朗之万动力学}\quad
&\Delta \theta_t = \frac{\epsilon_t}{2}\nabla_\theta \log p(\theta_t) + \frac{N}{n}\sum_{i=1}^{n}\nabla \log p(y_{t_i} \mid \theta_t) + \eta_t,
\end{aligned}
$$

其中 \(\epsilon_t\) 是一列步长，满足 \(\sum_{t=1}^{\infty}\epsilon_t = \infty\) 且 \(\sum_{t=1}^{\infty}\epsilon_t^2 < \infty\)，并且 \(\eta_t \sim \mathcal{N}(0, I)\) 是白噪声。集合 \(\{y_1,\ldots,y_N\}\) 表示训练集。常数 \(N\) 和 \(n\) 分别表示完整训练集和小批量中的训练样本数。

### 3.2 （Stein 的）分数函数

把朗之万动力学和底层福克–普朗克方程放在一边，迭代过程的关键对象是对数似然函数的梯度 \(\nabla_x \log p(x)\)。它有一个正式名称，称为 Stein 分数函数，记为

$$
s_\theta(x) \stackrel{\mathrm{def}}{=} \nabla_x \log p_\theta(x). \tag{3.3}
$$

我们应当小心，不要把 Stein 分数函数与普通分数函数混淆。普通分数函数定义为

$$
s_x(\theta) \stackrel{\mathrm{def}}{=} \nabla_\theta \log p_\theta(x). \tag{3.4}
$$

普通分数函数是对数似然关于 \(\theta\) 的梯度。相比之下，Stein 分数函数是关于数据点 \(x\) 的梯度。最大似然估计使用普通分数函数，而朗之万动力学使用 Stein 分数函数。然而，由于扩散文献中大多数人把 Stein 分数函数称为分数函数，我们遵循这一习惯。

**例 3.4.** 若 \(p(x)\) 是高斯，\(p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-(x-\mu)^2/(2\sigma^2)}\)，则

$$
s(x) = \nabla_x \log p(x) = -\frac{x-\mu}{\sigma^2}.
$$

**例 3.5.** 若 \(p(x)\) 是高斯混合，\(p(x) = \sum_{i=1}^{N}\pi_i \frac{1}{\sqrt{2\pi\sigma_i^2}} e^{-(x-\mu_i)^2/(2\sigma_i^2)}\)，则

$$
s(x) = \nabla_x \log p(x)
= -\frac{\displaystyle\sum_{j=1}^{N}\pi_j \frac{1}{\sqrt{2\pi\sigma_j^2}} e^{-(x-\mu_j)^2/(2\sigma_j^2)}\frac{x-\mu_j}{\sigma_j^2}}{\displaystyle\sum_{i=1}^{N}\pi_i \frac{1}{\sqrt{2\pi\sigma_i^2}} e^{-(x-\mu_i)^2/(2\sigma_i^2)}}.
$$

上面两个例子的概率密度函数及对应分数函数见图 3.5。

**图 3.5：** 分数函数的例子。(a) \(\mathcal{N}(1, 1)\)。(b) \(0.6\mathcal{N}(2, 0.5^2) + 0.4\mathcal{N}(-2, 0.2^2)\)。

**分数函数的几何解释。** 理解分数函数的方式是记住它是关于数据 \(x\) 的梯度。对任何高维分布 \(p(x)\)，梯度将给我们一个向量场。分数函数有几种有用的解释：

- 向量的幅度在 \(\log p(x)\) 变化最大的地方最强。因此，在 \(\log p(x)\) 接近峰值的区域，梯度大多非常弱。
- 向量场表明一个数据点应当如何在等高线图中行进。在图 3.6 中我们展示一个高斯混合（两个高斯）的等高线图。我们画箭头表示向量场。现在如果我们考虑生活在该空间中的一个数据点，朗之万动力学方程基本上会把数据点沿向量场所指的方向移向盆地。
- 在物理学中，分数函数等价于“漂移”。这个名字暗示扩散粒子应当如何流向最低能量状态。

**图 3.6：** 分数函数的等高线图，以及两个样本的对应轨迹。(a) \(\nabla_x \log p(x)\) 的向量场。(b) \(x_t\) 的轨迹。

### 3.3 分数匹配技术

朗之万动力学中最困难的问题是如何得到 \(\nabla_x \log p(x)\)，因为我们无法得到 \(p(x)\)。本节我们简要讨论几种已知技术。

**显式分数匹配 [43]。** 假设给定数据集 \(\mathcal{X} = \{x^{(1)},\ldots,x^{(M)}\}\)。人们想出的解是考虑经典核密度估计，定义一个分布

$$
q_h(x) = \frac{1}{M}\sum_{m=1}^{M}\frac{1}{h} K\left(\frac{x - x^{(m)}}{h}\right), \tag{3.5}
$$

其中 \(h\) 只是核函数 \(K(\cdot)\) 的某个超参数，\(x^{(m)}\) 是训练集中第 \(m\) 个样本。图 3.7 说明核密度估计的想法。在左边所示的卡通图中，我们展示多个以不同数据点 \(x^{(m)}\) 为中心的核 \(K(\cdot)\)。所有这些单个核的和给出总体核密度估计 \(q(x)\)。在右边我们展示一个真实直方图及对应的核密度估计。我们注记 \(q(x)\) 至多是对真实数据分布 \(p(x)\) 的近似，而 \(p(x)\) 永远未知。

**图 3.7：** 核密度估计的说明。

由于 \(q(x)\) 是对永远不可得的 \(p(x)\) 的近似，我们可以基于 \(q(x)\) 学习 \(s_\theta(x)\)。这导致下面可用于训练网络的损失函数定义。

**定理 3.2.** 显式分数匹配损失是

$$
\begin{aligned}
J_{\mathrm{ESM}}(\theta)
&\stackrel{\mathrm{def}}{=} \frac{1}{2}\mathbb{E}_{p(x)}\bigl\lVert s_\theta(x) - \nabla_x \log p(x)\bigr\rVert^2 \\
&\approx \frac{1}{2}\mathbb{E}_{q_h(x)}\bigl\lVert s_\theta(x) - \nabla_x \log q_h(x)\bigr\rVert^2.
\end{aligned}
\tag{3.6}
$$

代入核密度估计，我们可以证明损失是

$$
\begin{aligned}
J_{\mathrm{ESM}}(\theta)
&= \mathbb{E}_{q_h(x)}\bigl\lVert s_\theta(x) - \nabla_x \log q_h(x)\bigr\rVert^2 \\
&= \int \bigl\lVert s_\theta(x) - \nabla_x \log q_h(x)\bigr\rVert^2 q_h(x)\, dx \\
&\approx \frac{1}{M}\sum_{m=1}^{M}\int \bigl\lVert s_\theta(x) - \nabla_x \log q_h(x)\bigr\rVert^2 \frac{1}{h} K\left(\frac{x-x^{(m)}}{h}\right)\, dx.
\end{aligned}
\tag{3.7}
$$

于是我们推导出了一个可以用来训练网络的损失函数。一旦我们训练网络 \(s_\theta\)，就可以在朗之万动力学方程中替换它，得到递推

$$
x_{t+1} = x_t + \tau s_\theta(x_t) + \sqrt{2\tau}\, z. \tag{3.8}
$$

显式分数匹配的问题是，核密度估计是对真实分布相当差的非参数估计。尤其当我们样本数量有限、且样本生活在高维空间时，核密度估计的性能可能很差。

**隐式分数匹配 [18]。** 在隐式分数匹配中，显式分数匹配损失被一个隐式损失替换：

$$
J_{\mathrm{ISM}}(\theta) \stackrel{\mathrm{def}}{=} \mathbb{E}_{p(x)}\left[\mathrm{Tr}(\nabla_x s_\theta(x)) + \frac{1}{2}\lVert s_\theta(x)\rVert^2\right], \tag{3.9}
$$

其中 \(\nabla_x s_\theta(x)\) 表示 \(s_\theta(x)\) 的雅可比。隐式分数匹配损失可以用蒙特卡洛近似

$$
J_{\mathrm{ISM}}(\theta) \approx \frac{1}{M}\sum_{m=1}^{M}\sum_{i}\left(\partial_i s_\theta(x^{(m)}) + \frac{1}{2}\bigl|[s_\theta(x^{(m)})]_i\bigr|^2\right),
$$

其中 \(\partial_i s_\theta(x^{(m)}) = \frac{\partial}{\partial x_i}[s_\theta(x)]_i = \frac{\partial^2}{\partial x_i^2}\log p(x)\)。如果分数函数的模型由深度神经网络实现，迹算子可能难以计算，从而使隐式分数匹配不可扩展 [40]。

**去噪分数匹配。** 鉴于显式和隐式分数匹配的潜在缺点，我们现在介绍一种更流行的分数匹配，即 Vincent [43] 的去噪分数匹配（denoising score matching，DSM）。在 DSM 中，损失函数定义如下：

$$
J_{\mathrm{DSM}}(\theta) \stackrel{\mathrm{def}}{=} \mathbb{E}_{q(x, x')}\left[\frac{1}{2}\bigl\lVert s_\theta(x) - \nabla_x \log q(x \mid x')\bigr\rVert^2\right]. \tag{3.10}
$$

这里的关键差别是我们用条件分布 \(q(x \mid x')\) 替换分布 \(q(x)\)。前者需要近似，例如通过核密度估计，而后者不需要。

在 \(q(x \mid x') = \mathcal{N}(x \mid x', \sigma^2)\) 的特殊情形，我们可以令 \(x = x' + \sigma z\)。这将给出

$$
\begin{aligned}
\nabla_x \log q(x \mid x')
&= \nabla_x \log \frac{1}{(\sqrt{2\pi\sigma^2})^d}\exp\left(-\frac{\lVert x - x'\rVert^2}{2\sigma^2}\right) \\
&= \nabla_x\left(-\frac{\lVert x - x'\rVert^2}{2\sigma^2} - \log(\sqrt{2\pi\sigma^2})^d\right) \\
&= -\frac{x - x'}{\sigma^2} = -\frac{z}{\sigma}.
\end{aligned}
$$

结果，去噪分数匹配的损失函数变成

$$
\begin{aligned}
J_{\mathrm{DSM}}(\theta)
&\stackrel{\mathrm{def}}{=} \mathbb{E}_{q(x, x')}\left[\frac{1}{2}\bigl\lVert s_\theta(x) - \nabla_x \log q(x \mid x')\bigr\rVert^2\right] \\
&= \mathbb{E}_{q(x')}\left[\frac{1}{2}\left\lVert s_\theta(x' + \sigma z) + \frac{z}{\sigma}\right\rVert^2\right].
\end{aligned}
$$

如果我们用 \(x\) 替换哑变量 \(x'\)，并注意到当我们被给定训练数据集时，从 \(q(x)\) 采样可以被从 \(p(x)\) 采样替换，就可以得出下列结论。

**定理 3.3.** 去噪分数匹配有如下定义的损失函数

$$
J_{\mathrm{DSM}}(\theta) = \mathbb{E}_{p(x)}\left[\frac{1}{2}\left\lVert s_\theta(x + \sigma z) + \frac{z}{\sigma}\right\rVert^2\right]. \tag{3.11}
$$

式 (3.11) 的美妙之处在于它高度可解释。量 \(x + \sigma z\) 实际上是给干净图像 \(x\) 加噪声 \(\sigma z\)。分数函数 \(s_\theta\) 应当取这幅噪声图像并预测噪声 \(z/\sigma\)。预测噪声等价于去噪，因为任何去噪后的图像加上预测的噪声将给出噪声观测。因此式 (3.11) 是一个去噪步骤。

下面由 Vincent [43] 证明的定理建立了 DSM 与 ESM 之间的等价性。正是这一等价性允许我们用 DSM 估计分数函数。

**定理 3.4. [Vincent [43]]** 对一个与变量 \(\theta\) 无关的常数 \(C\)，有

$$
J_{\mathrm{DSM}}(\theta) = J_{\mathrm{ESM}}(\theta) + C. \tag{3.12}
$$

**定理 3.4 的证明.** 这里的证明基于 [43]。我们从显式分数匹配损失函数开始，它由下式给出

$$
\begin{aligned}
J_{\mathrm{ESM}}(\theta)
&= \mathbb{E}_{q(x)}\left[\frac{1}{2}\bigl\lVert s_\theta(x) - \nabla_x \log q(x)\bigr\rVert^2\right] \\
&= \mathbb{E}_{q(x)}\left[\frac{1}{2}\lVert s_\theta(x)\rVert^2 - s_\theta(x)^\top \nabla_x \log q(x) + \underbrace{\frac{1}{2}\lVert \nabla_x \log q(x)\rVert^2}_{\stackrel{\mathrm{def}}{=} C_1,\ \text{与 }\theta\text{ 无关}}\right].
\end{aligned}
$$

让我们放大第二项。我们可以证明

$$
\begin{aligned}
\mathbb{E}_{q(x)}\bigl[s_\theta(x)^\top \nabla_x \log q(x)\bigr]
&= \int s_\theta(x)^\top \nabla_x \log q(x)\, q(x)\, dx
&& \text{（期望）} \\
&= \int s_\theta(x)^\top \frac{\nabla_x q(x)}{q(x)}\, q(x)\, dx
&& \text{（梯度）} \\
&= \int s_\theta(x)^\top \nabla_x q(x)\, dx.
\end{aligned}
$$

接下来，我们通过回忆 \(q(x) = \int q(x') q(x \mid x')\, dx'\) 来考虑条件化。这将给出

$$
\begin{aligned}
\int s_\theta(x)^\top \nabla_x q(x)\, dx
&= \int s_\theta(x)^\top \nabla_x\left(\int q(x') q(x \mid x')\, dx'\right) dx
&& \text{（条件）} \\
&= \int s_\theta(x)^\top\left(\int q(x')\nabla_x q(x \mid x')\, dx'\right) dx
&& \text{（移动梯度）} \\
&= \int s_\theta(x)^\top\left(\int q(x')\nabla_x q(x \mid x')\cdot \frac{q(x \mid x')}{q(x \mid x')}\, dx'\right) dx
&& \text{（乘除）} \\
&= \iint s_\theta(x)^\top q(x') \underbrace{\frac{\nabla_x q(x \mid x')}{q(x \mid x')}}_{=\nabla_x \log q(x \mid x')} q(x \mid x')\, dx'\, dx
&& \text{（重排项）} \\
&= \iint q(x \mid x') q(x')\, s_\theta(x)^\top \nabla_x \log q(x \mid x')\, dx'\, dx
&& \text{（移动积分）} \\
&= \mathbb{E}_{q(x, x')}\bigl[s_\theta(x)^\top \nabla_x \log q(x \mid x')\bigr].
\end{aligned}
$$

因此，如果把这一结果代回 ESM 的定义，我们可以证明

$$
J_{\mathrm{ESM}}(\theta)
= \mathbb{E}_{q(x)}\left[\frac{1}{2}\lVert s_\theta(x)\rVert^2\right]
- \mathbb{E}_{q(x, x')}\bigl[s_\theta(x)^\top \nabla_x \log q(x \mid x')\bigr] + C_1.
$$

把这与 DSM 的定义比较，我们可以观察到

$$
\begin{aligned}
J_{\mathrm{DSM}}(\theta)
&\stackrel{\mathrm{def}}{=} \mathbb{E}_{q(x, x')}\left[\frac{1}{2}\bigl\lVert s_\theta(x) - \nabla_x \log q(x \mid x')\bigr\rVert^2\right] \\
&= \mathbb{E}_{q(x, x')}\left[\frac{1}{2}\lVert s_\theta(x)\rVert^2 - s_\theta(x)^\top \nabla_x \log q(x \mid x') + \underbrace{\frac{1}{2}\lVert \nabla_x \log q(x \mid x')\rVert^2}_{\stackrel{\mathrm{def}}{=} C_2,\ \text{与 }\theta\text{ 无关}}\right] \\
&= \mathbb{E}_{q(x)}\left[\frac{1}{2}\lVert s_\theta(x)\rVert^2\right]
- \mathbb{E}_{q(x, x')}\bigl[s_\theta(x)^\top \nabla_x \log q(x \mid x')\bigr] + C_2.
\end{aligned}
$$

因此我们得出

$$
J_{\mathrm{DSM}}(\theta) = J_{\mathrm{ESM}}(\theta) - C_1 + C_2.
$$

分数匹配模型中的训练过程通常通过最小化去噪分数匹配损失函数来完成。如果给定训练数据集 \(\{x^{(m)}\}_{m=1}^{M}\)，优化目标是

$$
\begin{aligned}
\theta^\star
&= \operatorname*{argmin}_\theta \mathbb{E}_{p(x)}\left[\frac{1}{2}\left\lVert s_\theta(x + \sigma z) + \frac{z}{\sigma}\right\rVert^2\right] \\
&\approx \operatorname*{argmin}_\theta \frac{1}{M}\sum_{m=1}^{M}\frac{1}{2}\left\lVert s_\theta\bigl(x^{(m)} + \sigma z^{(m)}\bigr) + \frac{z^{(m)}}{\sigma}\right\rVert^2,
\qquad z^{(m)} \sim \mathcal{N}(0, I).
\end{aligned}
$$

图 3.8 说明分数函数 \(s_\theta(x)\) 的训练过程。

**图 3.8：** 去噪分数匹配中 \(s_\theta\) 的训练。网络 \(s_\theta\) 被训练来估计噪声。

上述训练过程假设噪声水平 \(\sigma\) 固定。把它推广到多个噪声水平并不困难。Song 与 Ermon [40] 的噪声条件分数网络（noise conditioned score network，NCSN）论证，人们可以改为优化下列损失

$$
J_{\mathrm{NCSN}}(\theta) = \frac{1}{L}\sum_{i=1}^{L}\lambda(\sigma_i)\, \ell(\theta; \sigma_i), \tag{3.13}
$$

其中单个损失函数按照噪声水平 \(\sigma_1,\ldots,\sigma_L\) 定义为

$$
\ell(\theta; \sigma) = \mathbb{E}_{p(x)}\left[\frac{1}{2}\left\lVert s_\theta(x + \sigma z) + \frac{z}{\sigma}\right\rVert^2\right].
$$

系数函数 \(\lambda(\sigma_i)\) 常常基于经验发现选为 \(\lambda(\sigma) = \sigma^2\) [40]。噪声水平序列常常满足 \(\sigma_1/\sigma_2 = \cdots = \sigma_{L-1}/\sigma_L > 1\)。

对推理，我们假设已经训练了分数估计器 \(s_\theta\)。要生成一幅图像，我们用朗之万方程通过给图像去噪来迭代抽取样本。在 NCSN 的情形，对应的朗之万方程可以通过退火重要性采样实现：

$$
x_{t+1} = x_t + \frac{\alpha_i}{2}\, s_\theta(x_t, \sigma_i) + \sqrt{\alpha_i}\, z_t, \qquad z_t \sim \mathcal{N}(0, I),
$$

其中 \(\alpha_i = \sigma_i^2 / \sigma_L^2\) 是步长，\(s_\theta(x_t, \sigma_i)\) 表示噪声水平 \(\sigma_i\) 的分数匹配函数。对 \(t\) 的迭代对每个 \(\sigma_i\) 从 \(i = 1\) 到 \(L\) 顺序重复。实现的更多细节，我们请读者参考 Song 与 Ermon [40] 原始论文的算法 1。

### 3.4 结语

关于分数匹配的额外阅读应当从 Vincent 的技术报告 [43] 开始。近期文献中一篇非常流行的论文是 Song 与 Ermon [40]，他们的后续工作 [41]，以及 [42]。在他们的论文中，他们提出一个重要讨论：训练分数函数时，我们需要一个噪声日程，使分数函数被训练得更好。

分数匹配的适用性远超生成图像。它们可以用于求解许多重要的图像复原问题，例如去模糊、去噪、超分辨率等。Kadkhodaie 与 Simoncelli [19] 是较早明确把分数函数作为图像重建过程一部分的论文之一。Kawar 等人 [21] 提出了类似概念，他们从一个包含前向图像形成模型和先验的后验分布中采样。对前向模型结构更好的问题，已经证明采用近端映射的想法会提高性能。这一路线可以建立在算子分裂策略之上，例如即插即用 ADMM [8]，并把它扩展到即插即用扩散模型，例如 Zhu 等人，或 Bouman 与 Buzzard 的生成式即插即用模型 [5]。最近也观察到，当我们无法得到退化过程时，可以直接做向均值的回归 [11]。各种应用论文见 [9, 17, 37]。

---

## 4 随机微分方程（SDE）

前两节我们通过 DDPM 和 SMLD 的视角研究了扩散模型。本节我们将通过微分方程的透镜研究扩散模型。如我们在讨论朗之万方程时提到的，朗之万方程的稳态解是一个随机变量，其概率分布满足福克–普朗克方程。迭代算法与微分方程之间这种不寻常的联系使这一主题格外有趣。本节的目的是提供随机微分方程的基本原理，以及它们如何被用来理解扩散模型。

### 4.1 从迭代算法到常微分方程

第一个、也是最重要的问题，至少对微分方程背景很少的读者而言，是为什么一个迭代算法可以与一个微分方程相关。让我们通过两个例子来理解这一点。

**例 4.1. 简单的一阶 ODE。** 想象我们被给定一个离散时间算法，迭代由递推定义：

$$
x_i = \left(1 - \frac{\beta \Delta t}{2}\right) x_{i-1}, \qquad i = 1, 2, \ldots, N, \tag{4.1}
$$

对某个超参数 \(\beta\) 和步长参数 \(\Delta t\)。我们可以把这个迭代格式变成一个连续时间微分方程。

假设存在一个连续时间函数 \(x(t)\)。我们定义一个离散化方案，令 \(x_i = x(i/N)\)，\(i = 1,\ldots,N\)，且 \(\Delta t = 1/N\)，\(t \in \{0, 1/N, \ldots, (N-1)/N\}\)。于是上述递推可以写成

$$
x(t + \Delta t) = \left(1 - \frac{\beta \Delta t}{2}\right) x(t).
$$

重排项将给出

$$
\frac{x(t + \Delta t) - x(t)}{\Delta t} = -\frac{\beta}{2}\, x(t),
$$

其中在极限 \(\Delta t \to 0\) 时，我们可以把离散方程写成常微分方程（ordinary differential equation，ODE）

$$
\frac{dx(t)}{dt} = -\frac{\beta}{2}\, x(t). \tag{4.2}
$$

不仅如此，我们可以求出该 ODE 的解析解，解由下式给出

$$
x(t) = e^{-\frac{\beta}{2} t}. \tag{4.3}
$$

解的验证可以通过把式 (4.3) 代入式 (4.2) 来完成。

**图 4.1：** 解析解以及数值格式产生的估计。

ODE 的力量在于它给我们一个解析解。不必诉诸迭代格式（那将花费数百到数千次迭代），解析解精确告诉我们解在任何时刻 \(t\) 的行为。为说明这一事实，我们在上图中展示由算法定义的解 \(x_1, x_2, \ldots, x_i, \ldots, x_N\) 的轨迹。这里我们选择 \(\Delta t = 0.1\)。在同一张图中，我们直接画出连续时间解 \(x(t) = \exp\{-\beta t/2\}\)，对任意 \(t\)。如图所示，解析解与迭代格式预测的轨迹完全相同。

我们在这个启发例子中观察到两个有趣事实：

- 离散时间迭代格式可以写成连续时间 ODE。事实上，许多离散时间算法都有与之关联的 ODE。
- 对简单 ODE，我们可以写下闭式解析解。更复杂的 ODE 很难写出解析解。但我们仍然可以用 ODE 工具分析解的行为。我们也可以推导极限解 \(t \to \infty\)。

**例 4.2. 梯度下降。** 回忆对一个（性态良好的）凸函数 \(f\)，梯度下降算法是下列递推。对 \(i = 1, 2, \ldots, N\)，做

$$
x_i = x_{i-1} - \beta_{i-1} \nabla f(x_{i-1}), \tag{4.4}
$$

其中 \(\beta_i\) 是步长参数。使用与上例相同的离散化，我们可以证明（令 \(\beta_{i-1} = \beta(t)\Delta t\)）：

$$
\begin{aligned}
x_i = x_{i-1} - \beta_{i-1}\nabla f(x_{i-1})
&\Longrightarrow x(t + \Delta t) = x(t) - \beta(t)\Delta t\, \nabla f(x(t)) \\
&\Longrightarrow \frac{x(t + \Delta t) - x(t)}{\Delta t} = -\beta(t)\nabla f(x(t)) \\
&\Longrightarrow \frac{dx(t)}{dt} = -\beta(t)\nabla f(x(t)).
\end{aligned}
\tag{4.5}
$$

右边所示的常微分方程有一条解轨迹 \(x(t)\)。这个 \(x(t)\) 称为函数 \(f\) 的梯度流（gradient flow）。

为简单起见，我们可以令 \(\beta(t) = \beta\) 对所有 \(t\)。于是关于这个 ODE 有两个简单事实。首先，我们可以证明

$$
\begin{aligned}
\frac{d}{dt} f(x(t))
&= \nabla f(x(t))^\top \frac{dx(t)}{dt}
&& \text{（链式法则）} \\
&= \nabla f(x(t))^\top [-\beta \nabla f(x(t))]
&& \text{（式 (4.5)）} \\
&= -\beta \nabla f(x(t))^\top \nabla f(x(t)) \\
&= -\beta \lVert \nabla f(x(t))\rVert^2 \le 0
&& \text{（范数平方）}.
\end{aligned}
$$

因此，当我们从 \(x_{i-1}\) 移到 \(x_i\) 时，目标值 \(f(x(t))\) 必须下降。这与我们的期望一致，因为梯度下降算法应当随着迭代进行把代价降下来。其次，在极限 \(t \to \infty\) 时，我们知道 \(dx(t)/dt \to 0\)。因此 \(dx(t)/dt = -\beta \nabla f(x(t))\) 将蕴含

$$
\nabla f(x(t)) \to 0, \qquad \text{当 } t \to \infty. \tag{4.6}
$$

因此解轨迹 \(x(t)\) 将趋近函数 \(f\) 的极小点。

**前向与反向更新。**

让我们用梯度下降的例子再说明 ODE 的一个方面。回到式 (4.4)，我们认出递推可以等价地写成（假设 \(\beta(t) = \beta\)）：

$$
\underbrace{x_i - x_{i-1}}_{\Delta x} = -\underbrace{\beta_{i-1}}_{\beta\Delta t}\nabla f(x_{i-1})
\quad\Longrightarrow\quad
dx = -\beta \nabla f(x)\, dt, \tag{4.7}
$$

其中当我们设 \(\Delta t \to 0\) 且 \(\Delta x \to 0\) 时连续方程成立。这个等式有趣的一点是，它通过把 \(\Delta x\) 写成 \(dt\) 的函数，给了我们更新 \(\Delta x\) 的一个总结。它说如果我们沿时间轴移动 \(dt\)，那么解 \(x\) 将被更新 \(dx\)。

式 (4.7) 定义了变化之间的关系。如果我们考虑一列迭代 \(i = 1, 2, \ldots, N\)，并且如果我们被告知迭代的推进遵循式 (4.7)，那么我们可以写

$$
\begin{aligned}
\text{（前向）}\quad
x_i &= x_{i-1} + \Delta x_{i-1} \approx x_{i-1} + dx \\
&= x_{i-1} - \nabla f(x_{i-1})\, \beta\, dt \\
&\approx x_{i-1} - \beta_{i-1}\nabla f(x_{i-1}).
\end{aligned}
$$

我们称这为前向方程，因为我们通过 \(x + \Delta x\) 更新 \(x\)，假设 \(t \leftarrow t + \Delta t\)。

现在考虑一列迭代 \(i = N, N-1, \ldots, 2, 1\)。如果我们被告知迭代的推进遵循式 (4.7)，那么时间反转的迭代将是

$$
\begin{aligned}
\text{（反向）}\quad
x_{i-1} &= x_i - \Delta x_i \approx x_i - dx \\
&= x_i + \beta \nabla f(x_i)\, dt \\
&\approx x_i + \beta_i \nabla f(x_i).
\end{aligned}
$$

注意在反转推进方向时符号的变化。我们称这为反向方程。

### 4.2 什么是 SDE？

随机微分方程（stochastic differential equation，SDE）可以看作常微分方程（ODE）的扩展。本小节我们简要介绍记号并解释其含义。

**ODE。** 我们考虑如下形式的一阶 ODE：

$$
\frac{dx(t)}{dt} = f(t, x),
$$

其中 \(f(t, x)\) 是 \(t\) 和 \(x\) 的某个函数。假设初始条件是 \(x(0) = x_0\)，解取如下形式

$$
x(t) = x_0 + \int_0^t f(s, x(s))\, ds.
$$

解的积分形式常常伴随着一种称为微分形式的简写记号，可以写成

$$
dx = f(t, x)\, dt. \tag{4.8}
$$

**SDE。** 在 SDE 中，除了确定性函数 \(f(t, x)\)，我们还考虑一个随机扰动。例如，随机扰动可以取如下形式：

$$
\frac{dx(t)}{dt} = f(t, x) + g(t, x)\xi(t), \qquad \text{其中 } \xi(t) \sim \mathcal{N}(0, I),
$$

其中 \(\xi(t)\) 是一个噪声函数，例如白噪声。我们可以定义 \(dw = \xi(t)\, dt\)，其中 \(dw\) 通常称为布朗运动的微分形式。于是这个 SDE 的微分形式可以写成

$$
dx = f(t, x)\, dt + g(t, x)\, dw. \tag{4.9}
$$

因为 \(\xi(t)\) 是随机的，这个微分方程的解也是随机的。为明确解的随机性，我们应当通过积分方程解释微分形式

$$
x(t, \omega) = x_0 + \int_0^t f(s, x(s, \omega))\, ds + \int_0^t g(s, x(s, \omega))\, dw(s, \omega),
$$

其中 \(\omega\) 表示 \(x\) 的状态的指标。因此，当我们挑选随机过程 \(w(s, \omega)\) 的一个特定状态时，我们就求解对应于这个特定 \(\omega\) 的微分方程。

**例 4.3.** 考虑随机微分方程

$$
dx = a\, dw,
$$

对某个常数 \(a\)。基于上面的讨论，解轨迹将取如下形式

$$
x(t) = x_0 + \int_0^t a\, dw(s) = x_0 + a\int_0^t \xi(s)\, ds,
$$

其中最后一个等式用了 \(dw = \xi(t)\, dt\)。我们可以通过离散时间迭代数值实现 \(dx = a\, dw\) 来可视化解轨迹：

$$
x_i - x_{i-1} = a(w_i - w_{i-1})
\quad\Longrightarrow\quad
x_i = x_{i-1} + a z_{i-1},
\qquad z_{i-1} \stackrel{\mathrm{def}}{\sim} \mathcal{N}(0, I).
$$

例如，若 \(a = 0.05\)，\(x(t)\) 的一条可能轨迹将如下所示。初始点 \(x_0 = 0\) 标成红色，表示过程在时间上向前运动。

**图 4.2：** 过程 \(dx = a\, dw\) 的轨迹。

**例 4.4.** 考虑随机微分方程

$$
dx = -\frac{\alpha}{2}\, x\, dt + \beta\, dw.
$$

这个问题的解由 [Risken 式 3.7] 给出

$$
x(t) = x_0 e^{-\frac{\alpha}{2}t} + \beta \int_0^t e^{-\frac{\alpha}{2}(t-s)}\xi(s)\, ds.
$$

为可视化轨迹，我们考虑 \(\alpha = 1\) 和 \(\beta = 0.1\)。离散化将给出

$$
x_i - x_{i-1} = -\frac{\alpha}{2} x_{i-1} + \beta(w_i - w_{i-1})
\quad\Longrightarrow\quad
x_i = \left(1 - \frac{\alpha}{2}\right) x_{i-1} + \beta z_{i-1}.
$$

**图 4.3：** 过程 \(dx = -\frac{\alpha}{2} x\, dt + \beta\, dw\) 的轨迹。

### 4.3 用于 DDPM 与 SMLD 的随机微分方程

不深入 SDE 的物理（我们稍后会做），我们想把 SDE 与前几节研究的去噪扩散概率模型（DDPM）和分数匹配朗之万动力学（SMLD）联系起来。

**前向与反向扩散。** 扩散模型涉及一对方程：前向扩散过程和反向扩散过程。用导数表示，前向扩散过程可以写成

$$
\frac{dx(t)}{dt} = f(x, t) + g(t)\xi(t), \qquad \text{其中 } \xi(t) \sim \mathcal{N}(0, I).
$$

我们强调这是基于布朗运动的一种特定扩散过程。随机扰动 \(\xi\) 被假设为一个随机过程，\(\xi(t)\) 在所有 \(t\) 都是 i.i.d. 高斯随机变量。因此自相关函数是狄拉克函数 \(\mathbb{E}[\xi(t)\xi(t')] = \delta(t - t')\)。对这个扩散过程，我们可以用微分把它写成：

**定义 4.1. 前向扩散。**

$$
dx = \underbrace{f(x, t)\, dt}_{\text{漂移}} + \underbrace{g(t)\, dw}_{\text{扩散}}. \tag{4.10}
$$

这里微分是 \(dw = \xi(t)\, dt\)。这暗示我们可以把 \(\xi(t)\) 看成某种变化率（关于 \(t\)），对 \(\xi(t)\, dt\) 积分将给出 \(dw\)。

两项 \(f(x, t)\) 和 \(g(t)\) 带有物理含义。漂移系数是一个向量值函数 \(f(x, t)\)，定义封闭系统中的分子在没有随机效应时如何运动。对梯度下降算法，漂移由目标函数的负梯度定义。也就是说，我们希望解轨迹跟随目标的梯度。扩散系数 \(g(t)\) 是一个标量函数，描述分子如何从一个位置随机游走到另一个位置。函数 \(g(t)\) 决定随机运动有多强。

扩散方程的反方向是在时间上向后移动。根据 Anderson [2]，反向时间 SDE 如下给出。

**定义 4.2. 反向扩散。**

$$
dx = \underbrace{\bigl[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\bigr]}_{\text{漂移，含分数函数}} dt + \underbrace{g(t)\, dw}_{\text{反向时间扩散}}, \tag{4.11}
$$

其中 \(p_t(x)\) 是 \(x\) 在时刻 \(t\) 的概率分布，\(w\) 是时间向后流动时的维纳过程。

让我们简要谈谈反向时间扩散。反向时间扩散不过是一个按反向时间顺序进行的随机过程。因此当前向扩散定义 \(z_i = w_{i+1} - w_i\) 时，反向扩散定义 \(z_i = w_{i-1} - w_i\)。下面是一个例子。

**例 4.5.** 考虑反向扩散方程

$$
dx = a\, dw. \tag{4.12}
$$

我们可以把离散时间递推写成如下。对 \(i = N, N-1, \ldots, 1\)，做

$$
x_{i-1} = x_i + a(w_{i-1} - w_i) = x_i + a z_i, \qquad z_i \sim \mathcal{N}(0, I).
$$

在下图中我们展示这个反向时间过程的轨迹。注意标成红色的初始点在 \(x_N\)。过程被向后追踪到 \(x_0\)。

**图 4.4：** 反向扩散 \(dx = a\, dw\) 的轨迹。

**DDPM 的随机微分方程。** 为了建立 DDPM 与 SDE 之间的联系，我们考虑离散时间 DDPM 前向迭代。对 \(i = 1, 2, \ldots, N\)：

$$
x_i = \sqrt{1 - \beta_i}\, x_{i-1} + \sqrt{\beta_i}\, z_{i-1}, \qquad z_{i-1} \sim \mathcal{N}(0, I). \tag{4.13}
$$

我们可以证明这个方程可以从下面的前向 SDE 方程推导出来。

**定理 4.1.** DDPM 的前向采样方程可以写成 SDE

$$
dx = \underbrace{-\frac{\beta(t)}{2}\, x\, dt}_{= f(x, t)} + \underbrace{\sqrt{\beta(t)}\, dw}_{= g(t)}. \tag{4.14}
$$

**证明.** 我们定义步长 \(\Delta t = 1/N\)，并考虑辅助噪声水平 \(\{\bar\beta_i\}_{i=1}^{N}\)，其中 \(\beta_i = \bar\beta_i / N\)。于是

$$
\beta_i = \underbrace{\bar\beta_{Ni}}_{\bar\beta_i}\cdot \frac{1}{N} = \beta(t + \Delta t)\, \Delta t,
$$

其中我们假设当 \(N \to \infty\) 时 \(\bar\beta_i \to \beta(t)\)，后者是 \(0 \le t \le 1\) 上的连续时间函数。类似地，我们定义

$$
x_i = x\!\left(\frac{i}{N}\right) = x(t + \Delta t), \qquad z_i = z\!\left(\frac{i}{N}\right) = z(t + \Delta t).
$$

因此我们有

$$
\begin{aligned}
x_i &= \sqrt{1-\beta_i}\, x_{i-1} + \sqrt{\beta_i}\, z_{i-1} \\
&\Longrightarrow x_i = \sqrt{1 - \bar\beta_{Ni}}\, x_{i-1} + \sqrt{\bar\beta_{Ni}}\, z_{i-1} \\
&\Longrightarrow x(t + \Delta t) = \sqrt{1 - \beta(t+\Delta t)\cdot \Delta t}\, x(t) + \sqrt{\beta(t+\Delta t)\cdot \Delta t}\, z(t) \\
&\Longrightarrow x(t + \Delta t) \approx \left(1 - \frac{1}{2}\beta(t+\Delta t)\cdot \Delta t\right) x(t) + \sqrt{\beta(t+\Delta t)\cdot \Delta t}\, z(t) \\
&\Longrightarrow x(t + \Delta t) \approx x(t) - \frac{1}{2}\beta(t)\Delta t\, x(t) + \sqrt{\beta(t)\cdot \Delta t}\, z(t).
\end{aligned}
$$

于是当 \(\Delta t \to 0\) 时，我们有

$$
dx = -\frac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)}\, dw. \tag{4.15}
$$

因此我们证明了 DDPM 前向更新迭代可以等价地写成一个 SDE。

能够把 DDPM 前向更新迭代写成 SDE，意味着 DDPM 估计可以通过求解该 SDE 来确定。换言之，对一个适当定义的 SDE 求解器，我们可以把 SDE 扔进求解器。由适当选择的求解器返回的解将是 DDPM 估计。当然，我们并不被要求使用 SDE 求解器，因为 DDPM 迭代本身就在求解该 SDE。它可能不是最好的 SDE 求解器，因为 DDPM 迭代只是一阶方法。尽管如此，如果我们不感兴趣于使用 SDE 求解器，仍然可以用 DDPM 迭代得到一个解。下面是一个例子。

**例 4.6.** 考虑对所有 \(i = 0,\ldots,N-1\) 有 \(\beta_i = 0.05\) 的 DDPM 前向方程。我们通过从高斯混合中抽取来初始化样本 \(x_0\)，使得

$$
x_0 \sim \sum_{k=1}^{K} \pi_k \mathcal{N}(x_0 \mid \mu_k, \sigma_k^2 I),
$$

其中 \(\pi_1 = \pi_2 = 0.5\)，\(\sigma_1 = \sigma_2 = 1\)，\(\mu_1 = 3\)，\(\mu_2 = -3\)。然后，用方程

$$
x_i = \sqrt{1-\beta_i}\, x_{i-1} + \sqrt{\beta_i}\, z_{i-1}, \qquad z_{i-1} \sim \mathcal{N}(0, I),
$$

我们可以画出如下轨迹和分布。

**图 4.5：** \(x_t\) 轨迹的实现，从高斯混合开始，以单个高斯结束。

反向扩散方程由把适当的量代入式 (4.11) 得到：\(f(x, t) = -\beta(t)/2\) 且 \(g(t) = \sqrt{\beta(t)}\)。这将给出

$$
\begin{aligned}
dx &= \bigl[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\bigr] dt + g(t)\, dw \\
&= \left[-\frac{\beta(t)}{2}\, x - \beta(t)\nabla_x \log p_t(x)\right] dt + \sqrt{\beta(t)}\, dw,
\end{aligned}
$$

这将给我们下列方程：

**定理 4.2.** DDPM 的反向采样方程可以写成 SDE

$$
dx = -\beta(t)\left[\frac{x}{2} + \nabla_x \log p_t(x)\right] dt + \sqrt{\beta(t)}\, dw. \tag{4.16}
$$

**证明.** 迭代更新格式可以通过考虑 \(dx = x(t) - x(t - \Delta t)\) 以及 \(dw = w(t-\Delta t) - w(t) = -z(t)\) 来写。然后令 \(dt = \Delta t\)，我们可以证明

$$
\begin{aligned}
x(t) - x(t-\Delta t)
&= -\beta(t)\Delta t\left[\frac{x(t)}{2} + \nabla_x \log p_t(x(t))\right] - \sqrt{\beta(t)\Delta t}\, z(t) \\
&\Longrightarrow x(t-\Delta t)
= x(t) + \beta(t)\Delta t\left[\frac{x(t)}{2} + \nabla_x \log p_t(x(t))\right] + \sqrt{\beta(t)\Delta t}\, z(t).
\end{aligned}
$$

通过归并项，并假设 \(\beta(t)\Delta t \ll 1\)，我们认出

$$
\begin{aligned}
x(t-\Delta t)
&= x(t)\left(1 + \frac{\beta(t)\Delta t}{2}\right) + \beta(t)\Delta t\, \nabla_x \log p_t(x(t)) + \sqrt{\beta(t)\Delta t}\, z(t) \\
&\approx x(t)\left(1 + \frac{\beta(t)\Delta t}{2}\right) + \beta(t)\Delta t\, \nabla_x \log p_t(x(t)) + (\beta(t)\Delta t)^2 \cdots \\
&= \left(1 + \frac{\beta(t)\Delta t}{2}\right)\bigl(x(t) + \beta(t)\Delta t\, \nabla_x \log p_t(x(t))\bigr) + \sqrt{\beta(t)\Delta t}\, z(t).
\end{aligned}
$$

然后，遵循离散化方案，令 \(t \in \{0,\ldots,(N-1)/N\}\)，\(\Delta t = 1/N\)，\(x(t-\Delta t) = x_{i-1}\)，\(x(t) = x_i\)，且 \(\beta(t)\Delta t = \beta_i\)，我们可以证明

$$
\begin{aligned}
x_{i-1}
&= \left(1 + \frac{\beta_i}{2}\right)\bigl[x_i + \tfrac{\beta_i}{2}\nabla_x \log p_i(x_i)\bigr] + \sqrt{\beta_i}\, z_i \\
&\approx \frac{1}{\sqrt{1-\beta_i}}\bigl[x_i + \tfrac{\beta_i}{2}\nabla_x \log p_i(x_i)\bigr] + \sqrt{\beta_i}\, z_i,
\end{aligned}
\tag{4.17}
$$

其中 \(p_i(x)\) 是 \(x\) 在时刻 \(i\) 的概率密度函数。对实际实现，我们可以用估计的分数函数 \(s_\theta(x_i)\) 替换 \(\nabla_x \log p_i(x_i)\)。

于是我们恢复了与 Song 和 Ermon 在 [42] 中定义的一致的 DDPM 迭代。这是一个有趣的结果，因为它允许我们用分数函数把 DDPM 的迭代联系起来。Song 和 Ermon [42] 把这个 SDE 称为方差保持（variance preserving，VP）SDE。

**例 4.7.** 承接上例，我们用

$$
x_{i-1} = \frac{1}{\sqrt{1-\beta_i}}\left[x_i + \frac{\beta_i}{2}\nabla_x \log p_i(x_i)\right] + \sqrt{\beta_i}\, z_i
$$

执行反向扩散方程，其中 \(z_i \sim \mathcal{N}(0, I)\)。迭代的轨迹如下所示。

**图 4.6：** \(x_t\) 轨迹的实现，从单个高斯开始，以高斯混合结束。

**SMLD 的随机微分方程。** 分数匹配朗之万动力学模型也可以由一个 SDE 描述。首先，我们注意到在 SMLD 设定中，并没有真正的“前向扩散步骤”。然而，我们可以粗略论证：如果把 SMLD 训练中的噪声尺度分成 \(N\) 个水平，那么递推应当遵循马尔可夫链

$$
x_i = x_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2}\, z_{i-1}, \qquad i = 1, 2, \ldots, N. \tag{4.18}
$$

这不太难看出。如果我们假设 \(x_{i-1}\) 的方差是 \(\sigma_{i-1}^2\)，那么可以证明

$$
\mathrm{Var}[x_i] = \mathrm{Var}[x_{i-1}] + (\sigma_i^2 - \sigma_{i-1}^2) = \sigma_{i-1}^2 + (\sigma_i^2 - \sigma_{i-1}^2) = \sigma_i^2.
$$

因此，给定一列噪声水平，式 (4.18) 确实会生成估计 \(x_i\)，使噪声统计满足所期望的性质。

如果我们同意式 (4.18)，就很容易推导与式 (4.18) 相关联的 SDE。我们把结果总结如下。

**定理 4.3.** SMLD 的前向采样方程可以写成 SDE

$$
dx = \sqrt{\frac{d[\sigma(t)^2]}{dt}}\, dw. \tag{4.19}
$$

**证明.** 假设在极限下 \(\{\sigma_i\}_{i=1}^{N}\) 变成 \(0 \le t \le 1\) 上的连续时间 \(\sigma(t)\)，且 \(\{x_i\}_{i=1}^{N}\) 变成 \(x(t)\)，其中若令 \(t \in \{0, 1/N, \ldots, (N-1)/N\}\) 则 \(x_i = x(i/N)\)。于是我们有

$$
\begin{aligned}
x(t + \Delta t)
&= x(t) + \sqrt{\sigma(t+\Delta t)^2 - \sigma(t)^2}\, z(t) \\
&\approx x(t) + \sqrt{\frac{d[\sigma(t)^2]}{dt}\Delta t}\, z(t).
\end{aligned}
$$

在极限 \(\Delta t \to 0\) 时，方程收敛到

$$
dx = \sqrt{\frac{d[\sigma(t)^2]}{dt}}\, dw.
$$

把这映射到式 (4.11)，我们推导反向时间扩散方程。

**定理 4.4.** SMLD 的反向采样方程可以写成 SDE

$$
dx = -\frac{d[\sigma(t)^2]}{dt}\nabla_x \log p_t(x)\, dt + \sqrt{\frac{d[\sigma(t)^2]}{dt}}\, dw. \tag{4.20}
$$

**证明.** 我们认出

$$
f(x, t) = 0, \qquad g(t) = \sqrt{\frac{d[\sigma(t)^2]}{dt}}.
$$

结果，如果我们写反向方程 (4.11)，应当有

$$
\begin{aligned}
dx &= \bigl[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\bigr] dt + g(t)\, dw \\
&= \left(-\frac{d[\sigma(t)^2]}{dt}\nabla_x \log p_t(x(t))\right) dt + \sqrt{\frac{d[\sigma(t)^2]}{dt}}\, dw.
\end{aligned}
$$

对离散时间迭代，我们首先定义 \(\alpha(t) = d[\sigma(t)^2]/dt\)。然后，使用与 DDPM 情形相同的一组离散化设定，我们可以证明

$$
\begin{aligned}
x(t+\Delta t) - x(t)
&= -\bigl(\alpha(t)\nabla_x \log p_t(x)\bigr)\Delta t - \sqrt{\alpha(t)\Delta t}\, z(t) \\
&\Longrightarrow x(t) = x(t+\Delta t) + \alpha(t)\Delta t\, \nabla_x \log p_t(x) + \sqrt{\alpha(t)\Delta t}\, z(t) \\
&\Longrightarrow x_{i-1} = x_i + \alpha_i \nabla_x \log p_i(x_i) + \sqrt{\alpha_i}\, z_i \\
&\Longrightarrow x_{i-1} = x_i + (\sigma_i^2 - \sigma_{i-1}^2)\nabla_x \log p_i(x_i) + \sqrt{\sigma_i^2 - \sigma_{i-1}^2}\, z_i,
\end{aligned}
\tag{4.21}
$$

这与 SMLD 反向更新方程相同。Song 和 Ermon [42] 把这个 SDE 称为方差爆炸（variance exploding，VE）SDE。

**VP 与 VE 推理之间的等价性。** 正如我们刚刚看到的，DDPM 和 SMLD 对应于随机微分方程的两种变体：方差爆炸（VE）和方差保持（VP）。Kawar 等人 [21] 做出的一个观察是，由 VP 和 VE 决定的推理过程是等价的。因此，如果我们想把扩散模型用作另一项任务（例如图像复原）的一部分，用 VE 还是 VP 并不要紧。然而要注意，超参数的具体选择仍会影响训练，从而影响模型本身。

### 4.4 ODE 与 SDE 的数值求解器

SDE 和 ODE 不容易解析求解。在许多情形中，这些微分方程是数值求解的。本小节我们简要强调这些方法的总体想法。

为使讨论稍容易一些，我们将聚焦于一个简单 ODE：

$$
\frac{dx(t)}{dt} = f(t, x(t)). \tag{4.22}
$$

几何上，这个 ODE 意味着 \(x(t)\) 的导数等于函数 \(f(t, x)\) 在 \(x(t)\) 处的值。为此，我们需要 \(x(t)\) 的斜率与函数值匹配。在一维情形，上述几何解释可以翻译成下列方程：

$$
\frac{x(t) - x(t_0)}{t - t_0} = f(t_0, x_0),
$$

其中 \(t_0\) 是相当接近 \(t\) 的一个时刻。重排项，我们看到这个方程等价于

$$
x(t) = x(t_0) + f(t_0, x_0)(t - t_0).
$$

于是我们恢复了与梯度下降非常相似的东西。这就是欧拉方法。

**欧拉方法。** 欧拉方法是求解 ODE 的一阶数值方法。给定 \(dx(t)/dt = f(t, x)\) 且 \(x(t_0) = x_0\)，欧拉方法通过对 \(i = 0, 1, \ldots, N-1\) 的迭代格式求解问题，使得

$$
x_{i+1} = x_i + \alpha \cdot f(t_i, x_i), \qquad i = 0, 1, \ldots, N-1,
$$

其中 \(\alpha\) 是步长。让我们考虑一个简单例子。

**例 4.8. [3, 例 2.2]** 考虑下列 ODE

$$
\frac{dx(t)}{dt} = \frac{x(t) + t^2 - 2}{t + 1}.
$$

如果我们用步长 \(\alpha\) 应用欧拉方法，那么迭代将取如下形式

$$
x_{i+1} = x_i + \alpha \cdot f(t_i, x_i) = x_i + \alpha \cdot \frac{x_i + t_i^2 - 2}{t_i + 1}.
$$

**龙格–库塔（RK）方法。** 另一种常用的 ODE 求解器是龙格–库塔（Runge–Kutta，RK）方法。经典 RK-4 算法通过迭代求解 ODE

$$
x_{i+1} = x_i + \frac{\alpha}{6}\bigl(k_1 + 2k_2 + 2k_3 + k_4\bigr), \qquad i = 1, 2, \ldots, N,
$$

其中量 \(k_1, k_2, k_3, k_4\) 定义为

$$
\begin{aligned}
k_1 &= f(x_i, t_i), \\
k_2 &= f\!\left(t_i + \tfrac{\alpha}{2},\, x_i + \tfrac{\alpha}{2} k_1\right), \\
k_3 &= f\!\left(t_i + \tfrac{\alpha}{2},\, x_i + \tfrac{\alpha}{2} k_2\right), \\
k_4 &= f(t_i + \alpha,\, x_i + \alpha k_3).
\end{aligned}
$$

更多细节可以查阅数值方法教科书，例如 [3]。

**预测–校正算法 [42]。** 由于不同数值求解器在近似误差方面有不同行为，把 ODE（或 SDE）扔进一个现成的数值求解器将导致不同程度的误差 [20]。然而，如果我们专门试图求解反向扩散方程，有可能使用数值 ODE/SDE 求解器以外的技术来做适当校正，如图 4.7 所示。

**图 4.7：** 预测与校正算法。

让我们用 DDPM 作为例子。在 DDPM 中，反向扩散方程由下式给出

$$
x_{i-1} = \frac{1}{\sqrt{1-\beta_i}}\left[x_i + \frac{\beta_i}{2}\nabla_x \log p_i(x_i)\right] + \sqrt{\beta_i}\, z_i.
$$

我们可以把它看作反向扩散的欧拉方法。然而，如果我们已经训练了分数函数 \(s_\theta(x_i, i)\)，就可以运行分数匹配方程，即

$$
x_{i-1} = x_i + \epsilon_i s_\theta(x_i, i) + \sqrt{2\epsilon_i}\, z_i,
$$

做 \(M\) 次以进行校正。下面的算法总结了这一想法。（注意我们已经用估计替换了分数函数。）

**算法 1. DDPM 的预测校正算法。[42]**

- \(x_N \sim \mathcal{N}(0, I)\)。
- 对 \(i = N-1, \ldots, 0\)：
  - （预测）

$$
x_{i-1} = \frac{1}{\sqrt{1-\beta_i}}\left[x_i + \frac{\beta_i}{2} s_\theta(x_i, i)\right] + \sqrt{\beta_i}\, z_i. \tag{4.23}
$$

  - 对 \(m = 1, \ldots, M\)：
    - （校正）

$$
x_{i-1} = x_i + \epsilon_i s_\theta(x_i, i) + \sqrt{2\epsilon_i}\, z_i. \tag{4.24}
$$

对 SMLD 算法，两个方程是：

$$
\begin{aligned}
\text{预测：}\quad
& x_{i-1} = x_i + (\sigma_i^2 - \sigma_{i-1}^2)\, s_\theta(x_i, \sigma_i) + \sqrt{\sigma_i^2 - \sigma_{i-1}^2}\, z, \\
\text{校正：}\quad
& x_{i-1} = x_i + \epsilon_i \nabla_x s_\theta(x_i, \sigma_i) + \sqrt{\epsilon_i}\, z.
\end{aligned}
$$

我们可以像 DDPM 的预测–校正算法那样把它们配对，把校正迭代重复若干次。

### 4.5 结语

迭代算法与随机微分方程之间的联系不仅是一个具有自身科学价值的发现，如一些论文所评论的，这一发现也有实际用途。通过把多个扩散模型统一到同一个 SDE 框架，人们可以比较这些算法。在某些情形中，人们也可以通过借鉴 SDE 文献以及概率采样文献中的想法来改进数值格式。例如，[42] 中的预测–校正格式是一个与马尔可夫链蒙特卡洛耦合的混合 SDE 求解器。按照一些论文如 [1] 的说法，把扩散迭代映射到 SDE 提供了更多设计灵活性。在扩散算法的语境之外，一般的随机梯度下降算法也有对应的 SDE，例如福克–普朗克方程。人们已经演示如何以精确闭式从理论上分析估计的极限分布。这减轻了通过分析一个良定义的极限分布来分析随机算法的困难。

人们对开发加速 SDE 求解器的兴趣正在增长。Lu 等人 [26] 证明，我们在扩散模型中遇到的典型 SDE 可以分解成一个半线性项和一个非线性项。由于半线性项有闭式解，人们只需要用一个专门的数值求解器处理非线性项。进一步证明 DDIM 可以被看作一个加速 SDE 求解器。加速 SDE 求解器的其他途径是借助知识蒸馏 [36, 28]。

---

## 5 朗之万方程与福克–普朗克方程

近期许多关于扩散模型的论文，要么聚焦于提高图像生成质量（通过把它推广到语义上更有意义的图像），要么把它应用到新领域的问题（例如医学数据）。为帮助初学读者更好地理解这些应用的基础，我们想通过讨论与扩散模型相关联的随机微分方程的物理，在时间上往回走。通过研究这些 SDE 的基本原理，我们想更深入地理解这些方程从哪里来。

更具体地说，我们感兴趣于研究与扩散相关的两大组方程：朗之万方程和福克–普朗克方程。本节有几个目标。首先，我们想讨论朗之万方程的起源，并解释它为什么与扩散模型相关。我们想解释马尔可夫过程的一般性质，以及与之关联的微分方程。最后，我们想展示福克–普朗克方程是如何推导的，并讨论它为什么在扩散模型中扮演重要角色。

### 5.1 布朗运动

**历史视角。** 1827 年，植物学家 Robert Brown 观察到一个现象：小花粉粒放在水中时有不规则运动 [6]。花粉粒的运动后来被命名为布朗运动。布朗运动的解释由 Albert Einstein 于 1905 年做出 [14]，并大约同时由 Marian Smoluchowski 独立做出 [44]。Einstein 的主要论证是，花粉粒的运动由水分子的撞击引起。然而，由于系统中有万亿个分子，而且我们永远不知道单个分子的初始状态，用经典分析研究微观状态几乎不可能。Einstein 证明均方位移可以与一个扩散系数相关联，因此他通过考虑分子的统计行为引入了一种概率方法。<sup>4</sup> Einstein 的理论预测随后由 Jean-Baptiste Perrin 经验证实，后者后来于 1926 年获得诺贝尔物理学奖。Einstein 的论文几年之后，1908 年，法国物理学家 Paul Langevin 构造了一个随机马尔可夫力来描述粒子的碰撞与相互作用。这随后导致荷兰物理学家 Adriaan Fokker 于 1914 年、德国物理学家 Max Planck 于 1917 年发展出一个偏微分方程，现在称为福克–普朗克方程。Kramers–Moyal 展开由 Hans Kramers 于 1940 年、José Enrique Moyal 于 1949 年引入，它展示了一种泰勒展开技术来描述概率分布的时间演化。

<sup>4</sup> 这里的一个注记是，Andrey Kolmogorov 的概率书直到 1933 年才出版 [25]。因此在 1905 年，概率论的概念还处于幼年。

**布朗运动的推导。** 那么，什么是布朗运动，它与扩散模型有什么关系？假设有一个悬浮在流体中的粒子。斯托克斯定律指出，施加在粒子上的摩擦力由下式给出

$$
F(t) = -\alpha v(t), \tag{5.1}
$$

其中 \(F\) 是摩擦力，\(v\) 是速度，且 \(\alpha = 6\pi\mu R\)。这里 \(R\) 是粒子半径，\(\mu\) 是流体黏度。由牛顿第二定律，我们进一步知道 \(F(t) = m\dot{v}(t)\)，其中 \(m\) 是粒子质量。令两个方程相等

$$
\begin{cases}
F(t) = -\alpha v(t), \\
F(t) = m \dfrac{dv(t)}{dt},
\end{cases}
$$

我们将得到下列微分方程

$$
m\frac{dv(t)}{dt} + \alpha v(t) = 0.
$$

通过定义 \(\gamma \stackrel{\mathrm{def}}{=} \alpha/m\)，我们可以把上述方程简化为

$$
\frac{dv(t)}{dt} + \gamma v(t) = 0. \tag{5.2}
$$

当粒子显著重于流体分子时，上述确定性方程是准确的。原因是，由动量守恒，当大质量粒子与流体分子碰撞时，它获得的速度很小。然而，由于花粉粒如此轻，水分子的轰击会使它们以一种小但不可忽略的方式加速。这将产生随机涨落。水分子作用在粒子上的总力于是被修改为

$$
F(t) = -\alpha v(t) + F_f(t),
$$

其中 \(F_f(t)\) 是一个随机项。这将给我们一个修改后的微分方程

$$
m\frac{dv(t)}{dt} = -\alpha v(t) + F_f(t).
$$

通过定义 \(\Gamma(t) = F_f(t)/m\)，我们到达一个新的微分方程

$$
\frac{dv(t)}{dt} + \gamma v(t) = \Gamma(t), \tag{5.3}
$$

它可以写成简写记号

$$
\dot{v} + \gamma v = \Gamma(t). \tag{5.4}
$$

在式 (5.4) 中，随机过程 \(\Gamma(t)\) 表示一种称为朗之万力的随机力。它满足两个性质：

(i) 对所有 \(t\) 有 \(\mathbb{E}[\Gamma(t)] = 0\)，从而其均值函数是常数零；

(ii) 对所有 \(t\) 和 \(t'\) 有 \(\mathbb{E}[\Gamma(t)\Gamma(t')] = q\delta(t-t')\)，即自相关函数是幅度为 \(q\) 的狄拉克函数。

**注记。** 性质 (i) 和 (ii) 是宽平稳过程的特例。宽平稳过程是一个随机过程，它有常数均值函数（该常数不一定为零），并且自相关函数 \(R(t, t') \stackrel{\mathrm{def}}{=} \mathbb{E}[\Gamma(t)\Gamma(t')]\) 是差 \(t - t'\) 的函数。这个函数不一定是狄拉克函数。例如 \(R(t, t') = e^{-|t-t'|}\) 可以是宽平稳过程的一个有效自相关函数。

满足性质 (i) 和 (ii) 的随机过程在统计力学文献中有时称为 \(\delta\) 相关过程。构造 \(\delta\) 相关过程有不同方式。例如，我们可以假设对每个 \(t\) 有 \(\Gamma(t) \sim \mathcal{N}(0, 1)\)，或以同样方式定义的任何其他独立同分布。高斯分布更常被使用，因为许多物理现象可以用高斯描述，例如热噪声。满足性质 (i) 和 (ii) 的高斯随机过程称为高斯白噪声。

对任何宽平稳过程，维纳–辛钦定理说功率谱密度可以通过自相关函数的傅里叶变换定义。更具体地，若 \(R(\tau) = \mathbb{E}[\Gamma(t+\tau)\Gamma(t)]\) 是自相关函数（若 \(\Gamma(t)\) 是宽平稳过程，我们可以把 \(R(t, t')\) 写成 \(R(\tau)\)），维纳–辛钦定理指出功率谱密度是

$$
S(\omega) = \int_{-\infty}^{\infty} R(\tau) e^{-j\omega\tau}\, d\tau. \tag{5.5}
$$

因此若 \(R(\tau)\) 是狄拉克函数，\(S(\omega)\) 对所有 \(\omega\) 将有常数值。

**注记。** “高斯白噪声”这个名字来自功率谱密度 \(S(\omega)\) 对每个频率 \(\omega\) 都是均匀的（因此它包含可见光谱中的所有颜色）。白噪声定义为对所有 \(t\) 有 \(\Gamma(t) \sim \mathcal{N}(0, \sigma^2)\)。容易证明这样的 \(\Gamma(t)\) 会满足上述两条准则。

首先，由构造 \(\mathbb{E}[\Gamma(t)] = 0\)（因为 \(\Gamma(t) \sim \mathcal{N}(0, \sigma^2)\)）。其次，若 \(\Gamma(t) \sim \mathcal{N}(0, \sigma^2)\)，则必须有 \(R(\tau) = \mathbb{E}[\Gamma(t+\tau)\Gamma(t)]\) 是狄拉克函数。维纳–辛钦定理于是指出功率谱密度是平坦的，因为它是狄拉克函数的傅里叶变换。下图展示白噪声的随机实现及其自相关函数。

**图 5.1：** (a) 白高斯噪声的自相关函数 \(R(\tau)\)。(b) 随机过程 \(\Gamma(t)\) 的随机实现。

**从物理到生成式 AI。** 由于 \(\Gamma(t)\) 所展现的随机性，由式 (5.4) 给出的微分方程是一个随机微分方程（SDE）。因此这个 SDE 的解是一个随机过程，其中值 \(v(t)\) 在任何时刻 \(t\) 都是随机变量。布朗运动指这个随机过程 \(v(t)\) 作为时间函数的轨迹。式 (5.4) 中所得的 SDE 是朗之万方程的一个特例。我们称它为带 \(\delta\) 相关朗之万力的线性朗之万方程：

**定义 5.1.** 带 \(\delta\) 相关朗之万力的线性朗之万方程是如下形式的随机微分方程

$$
\dot{\xi} + \gamma\xi = \Gamma(t), \tag{5.6}
$$

其中 \(\Gamma(t)\) 是满足两个性质的随机过程：(i) 对所有 \(t\) 有 \(\mathbb{E}[\Gamma(t)] = 0\)；(ii) 对所有 \(t\) 和 \(t'\) 有 \(\mathbb{E}[\Gamma(t)\Gamma(t')] = q\delta(t-t')\)。

至此我们可以把式 (5.6) 中的朗之万方程与一个扩散模型（例如 DDPM）联系起来。

**例 5.1. 前向 DDPM。** 回忆 DDPM 前向扩散方程由下式给出

$$
dx = \underbrace{-\frac{\beta(t)}{2}\, x\, dt}_{= f(t)} + \underbrace{\sqrt{\beta(t)}\, dw}_{= g(t)}.
$$

把它表达成朗之万方程形式，我们可以写成

$$
\dot{\xi}(t) + f(t)\xi(t) = g(t)\Gamma(t), \qquad \text{其中 } \Gamma(t) \sim \mathcal{N}(0, I).
$$

**例 5.2. 反向 DDPM。** 反向 DDPM 扩散由式 (4.16) 给出

$$
dx = \underbrace{-\beta(t)\left[\frac{x}{2} + \nabla_x \log p_t(x)\right] dt}_{= f(\xi, t)} + \underbrace{\sqrt{\beta(t)}\, dw}_{= g(t)}.
$$

把它表达成朗之万方程形式，我们可以写成

$$
\dot{\xi}(t) = f(\xi, t) + g(t)\Gamma(t), \qquad \text{其中 } \Gamma(t) \sim \mathcal{N}(0, I).
$$

我们可以对其他扩散模型（例如 SMLD）继续这些例子。我们把它们留给读者作为练习。我们的底线信息是：前几章中看到的扩散方程都可以借助朗之万方程来表述。因此，如果我们想知道这些扩散方程产生的概率分布，就应当在朗之万方程的文献中寻找工具。

**（线性）朗之万方程的解。** 式 (5.6) 中呈现的线性朗之万方程是一个简单的方程。有可能解析地推导任何时刻 \(t\) 的解 \(\xi(t)\)。

我们从考虑更简单的问题 \(\Gamma(t) = 0\) 开始。在这种情况下，微分方程是

$$
\dot{\xi}(t) + \gamma\xi(t) = 0,
$$

它称为一阶齐次微分方程。这个微分方程的解如下。

**定理 5.1.** 考虑下列微分方程

$$
\dot{\xi}(t) + \gamma\xi(t) = 0,
$$

初始条件为 \(\xi(0) = \xi_0\)。解由下式给出

$$
\xi(t) = \xi_0 e^{-\gamma t}. \tag{5.7}
$$

**证明.** 重排项，我们可以证明

$$
\frac{\dot{\xi}(t)}{\xi(t)} = -\gamma,
$$

其中我们假设对所有 \(t\) 有 \(\xi(t) \neq 0\)，从而可以取 \(1/\xi(t)\)。两边积分将给出

$$
\int_0^t \frac{\dot{\xi}(t')}{\xi(t')}\, dt' = -\int_0^t \gamma\, dt'.
$$

方程左边将给出 \(\log\xi(t) - \log\xi(0)\)，而右边将给出 \(-\gamma t\)。令它们相等将给出

$$
\log\xi(t) - \log\xi(0) = -\gamma t \quad\Longrightarrow\quad \xi(t) = \xi_0 e^{-\gamma t}.
$$

现在让我们考虑 \(\Gamma(t)\) 存在的情形。微分方程变成

$$
\dot{\xi} + \gamma\xi = \Gamma(t),
$$

它称为一阶非齐次微分方程。为求解这个微分方程，我们采用一种称为参数变易法或常数变易法的技术 [29, 定理 1.2.3]。想法可以总结为两步。我们从先前的推导知道齐次方程的解是 \(\xi(t) = \xi_0 e^{-\gamma t}\)。因此让我们对非齐次情形的解做一个有根据的猜测：解取 \(s(t) = A(t)e^{-\gamma t}\) 的形式，对某个 \(A(t)\)。为记号简单，我们定义 \(h(t) = e^{-\gamma t}\)。如果 \(s(t)\) 确实是微分方程的解，那么我们可以计算 \(\dot{s}(t) + \gamma s(t) = \Gamma(t)\)。这个方程的左边是

$$
\begin{aligned}
\dot{s}(t) + \gamma s(t)
&= [A(t)h(t)]' + \gamma[A(t)h(t)] \\
&= A'(t)h(t) + A(t)h'(t) + \gamma A(t)h(t) \\
&= A(t)[h'(t) + \gamma h(t)] + A'(t)h(t) \\
&= A'(t)h(t),
\end{aligned}
$$

其中最后一个等式来自 \(h(t) = e^{-\gamma t}\) 是齐次方程的解，因而 \(h'(t) + \gamma h(t) = 0\)。因此，对 \(\dot{s}(t) + \gamma s(t) = \Gamma(t)\)，必须通过找到适当的 \(A'(t)\) 使 \(A'(t)h(t) = \Gamma(t)\)。但这并不困难。方程 \(A'(t)h(t) = \Gamma(t)\) 可以写成

$$
A'(t)e^{-\gamma t} = \Gamma(t) \quad\Longrightarrow\quad A'(t) = e^{\gamma t}\Gamma(t).
$$

两边积分将给出

$$
A(t) = \int_0^t e^{\gamma t'}\Gamma(t')\, dt',
$$

并且由于 \(s(t) = A(t)e^{-\gamma t}\)，我们可以证明

$$
s(t) = e^{-\gamma t}\int_0^t e^{\gamma t'}\Gamma(t')\, dt' = \int_0^t e^{-\gamma(t-t')}\Gamma(t')\, dt'.
$$

因此完整解（齐次部分与非齐次部分之和）是

$$
\xi(t) = \xi_0 e^{-\gamma t} + \int_0^t e^{-\gamma(t-t')}\Gamma(t')\, dt'.
$$

我们把结果总结如下。

**定理 5.2.** 考虑下列微分方程

$$
\dot{\xi}(t) + \gamma\xi(t) = \Gamma(t),
$$

初始条件为 \(\xi(0) = \xi_0\)。解由下式给出

$$
\xi(t) = \xi_0 e^{-\gamma t} + \int_0^t e^{-\gamma(t-t')}\Gamma(t')\, dt'. \tag{5.8}
$$

**平衡态的分布。** 先前的结果表明解 \(\xi(t)\) 是随机过程 \(\Gamma(t)\) 的函数。由于我们每次做（布朗运动）实验时并不知道 \(\Gamma(t)\) 的特定实现，通过看 \(\xi(t)\) 的概率分布来刻画 \(\xi(t)\) 往往更有用。下面我们跟随 Risken [33]，分析平衡态的概率分布，其中 \(t \to \infty\) 且 \(\xi(t) \to x\)。

**定理 5.3.** 考虑定义 5.1 中的朗之万方程

$$
\dot{\xi} + \gamma\xi = \Gamma(t), \tag{5.9}
$$

且 \(\Gamma(t)\) 是白高斯噪声，从而满足前述性质。令 \(\xi(t) = x\) 为这个 SDE 在平衡态的解，令 \(p(x)\) 为 \(x\) 的概率分布。则有

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\, e^{-x^2/(2\sigma^2)}, \tag{5.10}
$$

其中 \(\sigma = \sqrt{q/(2\gamma)}\)。换言之，平衡态的解 \(\xi(t) = x\) 是零均值高斯随机变量。

**证明.** 令 \(\xi_0 = \xi(0)\) 为初始条件。于是 SDE 的解取如下形式

$$
\xi(t) = \xi_0 e^{-\gamma t} + \int_0^t e^{-\gamma(t-t')}\Gamma(t')\, dt'. \tag{5.11}
$$

在平衡态 \(t \to \infty\) 时，我们可以丢掉 \(\xi_0 e^{-\gamma t}\)。此外，令 \(\tau = t - t'\)，我们可以把解写成

$$
\xi(t) = \int_0^{\infty} e^{-\gamma(t-t')}\Gamma(t')\, dt' = \int_0^{\infty} e^{-\gamma\tau}\Gamma(t-\tau)\, d\tau.
$$

概率密度函数 \(p(\xi)\) 可以通过特征函数的傅里叶逆变换来确定。回忆随机变量 \(v(t)\) 的特征函数是

$$
C(u) = \mathbb{E}[\exp\{iu\cdot\xi(t)\}] = 1 + \sum_{n=1}^{\infty}\frac{(iu)^n}{n!}\mathbb{E}[\xi(t)^n].
$$

因此，要找 \(C(u)\)，我们需要确定矩 \(\mathbb{E}[\xi(t)^n]\)。利用 Risken 中的一个结果（第 3 章式 3.26 和 3.27），我们可以证明

$$
\mathbb{E}[\xi(t)^{2n+1}] = 0.
\tag{5.12}
$$

$$
\begin{aligned}
\mathbb{E}[\xi(t)^{2n}]
&= \frac{(2n)!}{2^n n!}\left[\int_0^{\infty}\int_0^{\infty} e^{-\gamma(\tau_1+\tau_2)} q\delta(\tau_1-\tau_2)\, d\tau_1\, d\tau_2\right]^n \\
&= \frac{(2n)!}{2^n n!}\left[q\int_0^{\infty} e^{-2\gamma\tau_2}\, d\tau_2\right]^n
= \frac{(2n)!}{2^n n!}\left(\frac{q}{2\gamma}\right)^n.
\end{aligned}
\tag{5.13}
$$

把这一结果代入特征函数将给出

$$
\begin{aligned}
C(u)
&= 1 + 0 + \frac{2!}{2}\frac{q}{2\gamma} + 0 + \frac{(2\cdot 2)!}{2^2 2!}\left(\frac{q}{2\gamma}\right)^2 + \cdots \\
&= \sum_{n=0}^{\infty}\frac{(iu)^{2n}}{(2n)!}\mathbb{E}[\xi(t)^{2n}] \\
&= \sum_{n=0}^{\infty}\frac{(iu)^{2n}}{(2n)!}\cdot \frac{(2n)!}{2^n n!}\left(\frac{q}{2\gamma}\right)^n \\
&= \sum_{n=0}^{\infty}\frac{1}{n!}\left(-\frac{u^2 q}{4\gamma}\right)^n
= e^{-u^2 q/(4\gamma)}.
\end{aligned}
\tag{5.14}
$$

认出这是高斯的特征函数，我们可以用傅里叶逆变换取回概率密度函数

$$
p(x) = \sqrt{\frac{\gamma}{\pi q}}\, e^{-\gamma x^2 / q}.
$$

**例 5.3.（平衡态的前向 DDPM 分布）** 让我们做一个健全性检查，把我们的结果应用到前向 DDPM 方程，看看在平衡态会得到什么概率分布。

为简单起见，假设 DDPM 方程有常数学习率

$$
dx = -\frac{\beta}{2}\, x\, dt + \sqrt{\beta}\, dw.
$$

相关联的朗之万方程是

$$
\dot{\xi}(t) + \frac{\beta}{2}\xi(t) = \sqrt{\beta}\, \Gamma(t).
$$

当 \(t \to \infty\) 时，上面的定理表明

$$
p(x) = \sqrt{\frac{\gamma}{\pi q}}\, e^{-\gamma x^2 / q} = \frac{1}{\sqrt{2\pi}}\, e^{-x^2/2},
$$

其中我们代入了 \(\gamma = \beta/2\)，且 \(q = (\sqrt{\beta})^2 = \beta\)。因此当 \(t \to \infty\) 时 \(\xi(t)\) 的概率分布是 \(\mathcal{N}(0, 1)\)。这与我们所期望的一致。

**维纳过程。** 在 \(\gamma = 0\) 的特殊情形，线性朗之万方程简化为

$$
\dot{\xi} = \Gamma(t).
$$

在式 (5.11) 中令 \(\gamma = 0\)，我们可以证明

$$
\xi(t) = \xi_0 + \int_0^t \Gamma(t')\, dt',
$$

这也称为维纳过程。维纳过程解的概率分布可以推导如下。

**定理 5.4. 维纳过程。** 考虑维纳过程

$$
\dot{\xi} = \Gamma(t), \tag{5.15}
$$

其中 \(\Gamma(t)\) 是高斯白噪声，\(\mathbb{E}[\Gamma(t)] = 0\) 且 \(\mathbb{E}[\Gamma(t)\Gamma(t')] = q\delta(t-t')\)。解 \(\xi(t)\)（其中 \(\xi(t) = x\)）的概率分布 \(p(x, t)\) 是

$$
p(x, t) = \frac{1}{\sqrt{2\pi q t}}\, e^{-(x-\xi_0)^2/(2qt)}. \tag{5.16}
$$

**证明.** 这个结果与定理 5.3 的主要差别是，这里我们感兴趣的是任何时刻 \(t\) 的分布。为此，我们注意到 \(\xi(t) = \xi_0 + \int_0^t \Gamma(t')\, dt'\)。因此，为消去非零均值，我们可以考虑 \(\xi(t) - \xi_0\)。把这代入式 (5.13)，我们可以证明

$$
\begin{aligned}
\mathbb{E}[(\xi(t)-\xi_0)^{2n+1}] &= 0, \\
\mathbb{E}[(\xi(t)-\xi_0)^{2n}]
&= \frac{(2n)!}{2^n n!}\left[q\int_0^t 1\, d\tau_2\right]^n
= \frac{(2n)!}{2^n n!}(qt)^n.
\end{aligned}
$$

这蕴含 \(\xi(t) - \xi_0\) 的特征函数是

$$
C(u) = \sum_{n=0}^{\infty}\frac{1}{n!}\left(-\frac{u^2 qt}{2}\right)^n = e^{-u^2 qt / 2}. \tag{5.17}
$$

取傅里叶逆变换将给出 \(\xi(t)\) 的概率分布：

$$
p(x, t) = \frac{1}{\sqrt{2\pi q t}}\, e^{-(x-\xi_0)^2/(2qt)}. \tag{5.18}
$$

为获得关于这个方程的洞见，假设 \(\xi_0 = 0\) 且 \(q = 2k\)，对某个常数 \(k\)。这将给出

$$
p(x, t) = \frac{1}{\sqrt{4\pi k t}}\, e^{-x^2/(4kt)}.
$$

这个结果的一个有趣观察（可以在许多热动力学教科书中找到）是，上面推导的概率分布 \(p(x, t)\) 实际上是热方程的解：

$$
\frac{\partial}{\partial t} p(x, t) = k\frac{\partial^2}{\partial x^2} p(x, t), \tag{5.19}
$$

假设初始条件是 \(p(x, 0) = \delta(x)\)。要看到这一点，我们只需把概率分布代入热方程。然后可以看到

$$
\begin{aligned}
\frac{\partial}{\partial t} p(x, t)
&= \frac{1}{2t}\left(\frac{x^2}{2kt} - 1\right)\frac{1}{\sqrt{4\pi k t}} e^{-x^2/(4kt)}, \\
\frac{\partial^2}{\partial x^2} p(x, t)
&= \frac{1}{2kt}\left(\frac{x^2}{2kt} - 1\right)\frac{1}{\sqrt{4\pi k t}} e^{-x^2/(4kt)}.
\end{aligned}
$$

热方程的解表现得像一个从原点开始、并随时间增加向外扩展的高斯。这个结果的意义在于：虽然相对难以知道由朗之万方程定义的随机过程 \(\xi(t)\) 的精确轨迹，热方程提供了概率分布的完整图景。图 5.2 展示维纳过程 \(\xi(t)\) 的随机实现，及其对应的概率分布 \(p(x, t)\)。

**图 5.2：** 维纳过程的实现。(a) 随机过程遵循随机微分方程。我们展示该随机过程的若干实现。(b) 底层概率分布 \(p(x, t)\)。随着 \(t\) 增加，高斯的方差也增加。

对更复杂的朗之万方程（涉及非线性项），自然会期望有一个类似的偏微分方程刻画概率分布。更具体地说，似乎合理的是期望方程的一边有 \(\partial/\partial t\)，另一边有 \(\partial^2/\partial x^2\)。如我们将稍后展示的，福克–普朗克方程将有与此相似的形式。的确，人们可以从福克–普朗克方程推导出热方程。

**注记：** 对如下形式的齐次微分方程组

$$
\dot{\xi}_i + \sum_{j=1}^{N}\gamma_{ij}\xi_j = \Gamma_i(t), \qquad i = 1,\ldots,N,
$$

其中 \(\mathbb{E}[\Gamma_i(t)] = 0\)，\(\mathbb{E}[\Gamma_i(t)\Gamma_j(t')] = q_{ij}\delta(t-t')\)，且 \(q_{ij} = q_{ji}\)，对应的随机过程称为 Ornstein–Uhlenbeck 过程。

### 5.2 主方程

到目前为止我们一直在研究线性朗之万方程 \(\dot{\xi}(t) + \gamma\xi(t) = \Gamma(t)\)。这个方程使我们能够处理大多数前向扩散，其目标是给样本加噪声（即把输入分布转换成高斯）。对反向扩散，例如反向 DDPM 方程和反向 SMLD 方程，我们需要某种更一般的东西。我们现在考虑的方程是非线性朗之万方程，表达如下。

**定义 5.2.** 非线性朗之万方程取如下形式

$$
\dot{\xi} = h(\xi, t) + g(\xi, t)\Gamma(t), \tag{5.20}
$$

其中 \(h(\xi, t)\) 和 \(g(\xi, t)\) 是分别表示漂移和扩散的函数。和以前一样，我们假设 \(\Gamma(t)\) 是高斯白噪声，从而对所有 \(t\) 有 \(\mathbb{E}[\Gamma(t)] = 0\)，且 \(\mathbb{E}[\Gamma(t)\Gamma(t')] = 2\delta(t-t')\)。

读者可以参照例 5.2，看反向 DDPM 如何适合这个方程。

分析非线性朗之万方程的困难在于没有简单的闭式解。因此我们需要发展一些数学工具来帮助我们理解非线性朗之万方程。

**马尔可夫性质。** 让我们首先定义一个马尔可夫过程。假设 \(\xi(t)\) 在时刻 \(t_n\) 有值 \(x_n = \xi(t_n)\)，并令 \(t_1 \le t_2 \cdots \le t_n\)。我们将用记号 \(p(x_n, t_n)\) 描述 \(\xi(t_n) = x_n\) 的概率密度。我们也引入下列简写记号

$$
\mathbf{x}_n = [x_n,\ldots,x_1], \qquad \mathbf{t}_n = [t_n,\ldots,t_1].
$$

因此 \(p(\mathbf{x}_n, \mathbf{t}_n) = p(x_n, t_n, \ldots, x_1, t_1)\) 是 \((\xi(t_n),\ldots,\xi(t_1))\) 的联合分布。

让我们定义一个马尔可夫过程。我们说随机过程 \(\xi(t)\) 是马尔可夫的，如果下列无记忆条件成立。

**定义 5.3.** 随机过程 \(\xi(t)\) 称为马尔可夫过程，如果

$$
p(x_n, t_n \mid \mathbf{x}_{n-1}, \mathbf{t}_{n-1}) = p(x_n, t_n \mid x_{n-1}, t_{n-1}). \tag{5.21}
$$

也就是说，给定所有先前状态时在 \(t_n\) 得到状态 \(x_n\) 的概率，与我们只以紧邻的前一状态 \(x_{n-1}\)（在 \(t_{n-1}\)）为条件时相同。

满足定义 5.2 中非线性朗之万方程的随机过程 \(\xi(t)\) 是马尔可夫的，只要 \(\Gamma(t)\) 是 \(\delta\) 相关的。这意味着 \(t_n\) 处的条件概率只依赖于 \(t_{n-1}\) 处的那个值。原因由 Risken [33] 总结：(i) 一阶微分方程由其初值唯一确定；(ii) 在更早时刻 \(t < t_{n-1}\) 的 \(\delta\) 相关朗之万力 \(\Gamma(t)\) 不能改变更晚时刻 \(t > t_{n-1}\) 的条件概率。Risken 进一步阐述，如果 \(\Gamma(t)\) 不再是 \(\delta\) 相关的，马尔可夫性质就被破坏。例如，若 \(\Gamma(t)\) 使得 \(\mathbb{E}[\Gamma(t)\Gamma(t')] = \frac{q}{2}\gamma e^{-\gamma|t-t'|}\)，那么由 \(\dot{\xi}(t) = h(\xi) + \Gamma(t)\) 描述的过程将是非马尔可夫的。从现在起，我们将只聚焦于马尔可夫过程。

**Chapman–Kolmogorov 方程。** 考虑一个马尔可夫过程 \(\xi(t)\)。我们可以推导一个称为 Chapman–Kolmogorov 方程的有用结果。Chapman–Kolmogorov 方程指出，\(t_3\) 与 \(t_1\) 处的联合分布可以通过积分 \(t_3\) 给定 \(t_2\)、然后 \(t_2\) 给定 \(t_1\) 的条件概率来找到。这里的两个关键论证是贝叶斯定理加上边缘化的定义，以及马尔可夫过程的无记忆性质。

**定理 5.5. Chapman–Kolmogorov 方程。** 令 \(\xi(t)\) 为马尔可夫过程，令 \(x_n = \xi(t_n)\) 为 \(\xi(t)\) 在时刻 \(t_n\) 的状态。则

$$
p(x_3, t_3 \mid x_1, t_1) = \int p(x_3, t_3 \mid x_2, t_2)\, p(x_2, t_2 \mid x_1, t_1)\, dx_2, \tag{5.22}
$$

假设 \(t_1 \le t_2 \le t_3\)。

**证明.** 为记号简单，令 \(\mathbf{x}_n = \{x_n,\ldots,x_1\}\) 且 \(\mathbf{t}_n = \{t_n,\ldots,t_1\}\)。若 \(\xi(t)\) 是马尔可夫的，则由马尔可夫的无记忆性质，我们有

$$
\begin{aligned}
p(\mathbf{x}_n, \mathbf{t}_n)
&= p(x_n, t_n \mid \mathbf{x}_{n-1}, \mathbf{t}_{n-1})\, p(\mathbf{x}_{n-1}, \mathbf{t}_{n-1}) \\
&= p(x_n, t_n \mid x_{n-1}, t_{n-1})\, p(\mathbf{x}_{n-1}, \mathbf{t}_{n-1}).
\end{aligned}
$$

重复上述论证将给出

$$
p(\mathbf{x}_n, \mathbf{t}_n) = p(x_n, t_n \mid x_{n-1}, t_{n-1})\cdots p(x_2, t_2 \mid x_1, t_1)\cdot p(x_1, t_1).
$$

因此，通过对 \(x_2\) 边缘化 \(p(x_3, t_3)\)，我们可以证明

$$
\begin{aligned}
p(x_3, t_3, x_1, t_1)
&= \int p(x_3, t_3)\, dx_2 \\
&= \int p(x_3, t_3 \mid x_2, t_2)\, p(x_2, t_2 \mid x_1, t_1)\, p(x_1, t_1)\, dx_2.
\end{aligned}
$$

通过写 \(p(x_3, t_3, x_1, t_1) = p(x_3, t_3 \mid x_1, t_1)\, p(x_1, t_1)\)，我们可以证明

$$
p(x_3, t_3 \mid x_1, t_1) = \int p(x_3, t_3 \mid x_2, t_2)\, p(x_2, t_2 \mid x_1, t_1)\, dx_2. \tag{5.23}
$$

证明完成。

作为 Chapman–Kolmogorov 方程的一个推论，我们可以令 \(x = x_3\) 为当前状态，\(x_0 = x_1\) 为初始状态，\(x' = x_2\) 为中间状态。于是方程可以写成

$$
p(x, t \mid x_0, t_0) = \int p(x, t \mid x', t')\, p(x', t' \mid x_0, t_0)\, dx'. \tag{5.24}
$$

如果我们进一步丢掉对 \(x_0\) 的条件化，可以写

$$
p(x, t) = \int p(x, t \mid x', t')\, p(x', t')\, dx'.
$$

**主方程。** 基于 Chapman–Kolmogorov 方程，我们可以推导马尔可夫过程的一个基本方程。这个方程称为主方程（Masters Equation）。

**定理 5.6.** 令 \(\xi(t)\) 为马尔可夫过程。主方程指出

$$
\frac{\partial}{\partial t} p(x, t) = \int \bigl[W(x \mid x')\, p(x', t) - W(x' \mid x)\, p(x, t)\bigr]\, dx', \tag{5.25}
$$

其中 \(W(x \mid x')\) 是单位时间的概率密度函数。

**证明.** 回忆 Chapman–Kolmogorov 方程

$$
p(x_3, t_3 \mid x_1, t_1) = \int p(x_3, t_3 \mid x_2, t_2)\, p(x_2, t_2 \mid x_1, t_1)\, dx_2. \tag{5.26}
$$

我们考虑下列映射：

$$
(x_1, t_1) \to (x_0, t_0), \qquad (x_2, t_2) \to (x', t), \qquad (x_3, t_3) \to (x, t+\Delta t).
$$

于是 Chapman–Kolmogorov 方程可以写成

$$
p(x, t+\Delta t \mid x_0, t_0) = \int p(x, t+\Delta t \mid x', t)\, p(x', t \mid x_0, t_0)\, dx'. \tag{5.27}
$$

由于我们的目标是得到时间上的偏导数，我们考虑 \(p(x, t \mid x_0, t_0)\) 的时间导数：

$$
\begin{aligned}
\frac{\partial}{\partial t} p(x, t \mid x_0, t_0)
&= \lim_{\Delta t \to 0}\frac{p(x, t+\Delta t \mid x_0, t_0) - p(x, t \mid x_0, t_0)}{\Delta t} \\
&= \lim_{\Delta t \to 0}\frac{\int p(x, t+\Delta t \mid x', t)\, p(x', t \mid x_0, t_0)\, dx' - p(x, t \mid x_0, t_0)}{\Delta t}.
\end{aligned}
$$

我们注意到在上述方程右边有一个积分。如果我们交换变量 \(x\) 和 \(x'\)，可以使用下列观察：

$$
\int p(x', t+\Delta t \mid x, t)\, dx' = 1.
$$

我们可以把它插入上述方程并得到

$$
\lim_{\Delta t \to 0}\frac{1}{\Delta t}\left[
\int p(x, t+\Delta t \mid x', t)\, p(x', t \mid x_0, t_0)\, dx'
- \int p(x', t+\Delta t \mid x, t)\, p(x, t \mid x_0, t_0)\, dx'
\right].
$$

接下来，我们可以把极限移进积分。让我们定义

$$
\begin{aligned}
W(x, t \mid x', t) &= \lim_{\Delta t \to 0}\frac{1}{\Delta t} p(x, t+\Delta t \mid x', t), \\
W(x', t \mid x, t) &= \lim_{\Delta t \to 0}\frac{1}{\Delta t} p(x', t+\Delta t \mid x, t).
\end{aligned}
$$

于是我们有

$$
\frac{\partial}{\partial t} p(x, t \mid x_0, t_0)
= \int \bigl[W(x, t \mid x', t)\, p(x', t \mid x_0, t_0) - W(x', t \mid x, t)\, p(x, t \mid x_0, t_0)\bigr]\, dx'. \tag{5.28}
$$

如果我们固定 \((x_0, t_0)\)，就可以丢掉条件化。这将给出

$$
\frac{\partial}{\partial t} p(x, t)
= \int \bigl[W(x, t \mid x', t)\, p(x', t) - W(x', t \mid x, t)\, p(x, t)\bigr]\, dx'. \tag{5.29}
$$

在上述推导中，项 \(W(x, t \mid x', t)\) 和 \(W(x', t \mid x, t)\) 称为转移率。它们是单位时间的转移概率，单位为 \([\mathrm{time}^{-1}]\)。因此，如果我们对它们关于时间积分，将得到

$$
\int W(x, t \mid x', t)\, dt = p(x, t \mid x', t), \qquad
\int W(x', t \mid x, t)\, dt = p(x', t \mid x, t).
$$

可视化主方程的一种方式是把 \(\int W(x, t \mid x', t)\, p(x', t)\, dx'\) 看作流入，把 \(\int W(x', t \mid x, t)\, p(x, t)\, dx'\) 看作从状态 \(x'\) 到 \(x\)（以及从 \(x\) 到 \(x'\)）的转移概率的流出。因此如果我们把概率看成房间里粒子的密度，那么主方程说密度的变化率是粒子流入与流出之差：

$$
\underbrace{\frac{\partial}{\partial t} p(x, t)}_{\text{变化率}}
= \underbrace{\int W(x, t \mid x', t)\, p(x', t)\, dx'}_{\text{概率的流入}}
- \underbrace{\int W(x', t \mid x, t)\, p(x, t)\, dx'}_{\text{概率的流出}}. \tag{5.30}
$$

主方程在化学、生物学和许多学科中被广泛使用。粒子流入与流出的概念对研究系统动力学特别有用。主方程的另一个重要方面是它把时间 \(\partial_t\) 与状态 \(dx'\) 联系起来。这将在福克–普朗克方程中变得突出。

我们可以抱怨上述证明的一件事是：虽然它严格，但证明过程中缺乏物理直觉。下面我们呈现一个更直观但不那么严格的替代证明。该证明基于 Luca Donati 的讲义 [13]。

**直观证明。** 考虑一个只能取两个状态 1 或 2 的粒子。落在特定状态上的概率是 \(p(x_1, t)\) 和 \(p(x_2, t)\)，使得对任何 \(t\) 有 \(p(x_1, t) + p(x_2, t) = 1\)。现在考虑一个小区间 \((t, t+dt)\)。在这个区间里，粒子要么停留在当前状态，要么可以跳到另一个状态。这意味着在 \(t+dt\) 结束时，我们可以写

$$
\begin{aligned}
p(x_1, t+dt)
&= p(x_1, t)\, \mathbb{P}[\text{停留在 } x_1] + p(x_2, t)\, \mathbb{P}[\text{从 } x_2 \text{ 移到 } x_1] \\
&= p(x_1, t)\bigl(1 - \mathbb{P}[\text{从 } x_1 \text{ 移到 } x_2]\bigr) + p(x_2, t)\, \mathbb{P}[\text{从 } x_2 \text{ 移到 } x_1].
\end{aligned}
$$

定义率 \(W(x_2 \mid x_1)\) 和 \(W(x_1 \mid x_2)\)，使得

$$
\mathbb{P}[\text{从 } x_1 \text{ 移到 } x_2] = W(x_2 \mid x_1)\, dt, \qquad
\mathbb{P}[\text{从 } x_2 \text{ 移到 } x_1] = W(x_1 \mid x_2)\, dt.
$$

注意这里我们隐含假设转移分布是马尔可夫的，从而当前状态只依赖于其前一状态，而不依赖于整个历史。于是上述方程可以写成

$$
p(x_1, t+dt) = p(x_1, t)\bigl(1 - W(x_2 \mid x_1)\, dt\bigr) + p(x_2, t)\, W(x_1 \mid x_2)\, dt + O(dt^2).
$$

高阶项是为了计入 \((t, t+dt)\) 期间的多次跳跃，例如在该区间内从 \(x_1\) 跳到 \(x_2\) 再从 \(x_2\) 跳回 \(x_1\)。然而，若 \(dt \to 0\)，这一项将消失。重排项，我们可以写

$$
\frac{dp(x_1, t)}{dt} = -W(x_2 \mid x_1)\, p(x_1, t) + W(x_1 \mid x_2)\, p(x_2, t).
$$

我们可以把这个结果推广到进出 \(x_1\) 的多个状态。例如，

$$
\frac{dp(x_1, t)}{dt} = \sum_{j \neq 1}\bigl[-W(x_j \mid x_1)\, p(x_1, t) + W(x_1 \mid x_j)\, p(x_j, t)\bigr].
$$

为使它更一般，我们可以考虑 \(x_j\) 的连续统。重排项，我们将得到

$$
\frac{dp(x, t)}{dt} = \int \bigl[W(x \mid x')\, p(x', t) - W(x' \mid x)\, p(x, t)\bigr]\, dx',
$$

这就是主方程。

### 5.3 Kramers–Moyal 展开

有了主方程，我们现在可以处理非线性朗之万方程。回忆非线性朗之万方程没有闭式解，因此我们不能解析地写下解的概率分布。主方程允许我们通过一个称为福克–普朗克方程的偏微分方程写下概率分布的条件。福克–普朗克方程的推导需要一个称为 Kramers–Moyal 展开的数学结果。

**定理 5.7.** 令 \(\xi(t)\) 为马尔可夫过程，令 \(p(x, t)\) 为 \(\xi(t)\) 在时刻 \(t\) 取值 \(x\) 的概率分布。Kramers–Moyal 展开指出

$$
\frac{\partial}{\partial t} p(x, t) = \sum_{m=1}^{\infty}\left(-\frac{\partial}{\partial x}\right)^m \bigl[D^{(m)}(x, t)\, p(x, t)\bigr],
$$

其中 Kramers–Moyal 展开系数定义为

$$
D^{(m)}(x, t) = \frac{1}{m!}\lim_{\Delta t \to 0}\frac{1}{\Delta t}\mathbb{E}\bigl[(\xi(t+\Delta t) - x)^m\bigr]_{\xi(t)=x}.
$$

**证明.** 让我们从主方程开始。主方程指出

$$
\begin{aligned}
\frac{\partial}{\partial t} p(x, t \mid x_0, t_0)
&= \lim_{\Delta t \to 0}\frac{1}{\Delta t}\Biggl[
\int p(x, t+\Delta t \mid x', t)\, p(x', t \mid x_0, t_0)\, dx' \\
&\qquad - \int p(x', t+\Delta t \mid x, t)\, p(x, t \mid x_0, t_0)\, dx'
\Biggr].
\end{aligned}
$$

我们可以注入一个测试函数 \(\varphi(x)\)，使得

$$
\begin{aligned}
\frac{\partial}{\partial t}\int \varphi(x)\, p(x, t \mid x_0, t_0)\, dx
&= \lim_{\Delta t \to 0}\frac{1}{\Delta t}\Biggl(
\int\varphi(x)\int p(x, t+\Delta t \mid x', t)\, p(x', t \mid x_0, t_0)\, dx'\, dx \\
&\qquad - \int\varphi(x)\int p(x', t+\Delta t \mid x, t)\, p(x, t \mid x_0, t_0)\, dx'\, dx
\Biggr).
\end{aligned}
$$

测试函数的泰勒展开将给我们一个无穷级数

$$
\varphi(x) = \varphi(x') + \sum_{m=1}^{\infty}\frac{(x-x')^m}{m!}\left.\frac{\partial^m}{\partial x^m}\varphi(x)\right|_{x=x'}.
$$

把展开代入主方程将给出

$$
\begin{aligned}
&\frac{\partial}{\partial t}\int \varphi(x)\, p(x, t \mid x_0, t_0)\, dx \\
&= \lim_{\Delta t \to 0}\frac{1}{\Delta t}\Biggl[
\iint \varphi(x')\, p(x, t+\Delta t \mid x', t)\, p(x', t \mid x_0, t_0)\, dx'\, dx \\
&\quad + \iint\sum_{m=1}^{\infty}\frac{(x-x')^m}{m!}\left.\frac{\partial^m}{\partial x^m}\varphi\right|_{x=x'}
p(x, t+\Delta t \mid x', t)\, p(x', t \mid x_0, t_0)\, dx'\, dx \\
&\quad - \iint \varphi(x)\, p(x', t+\Delta t \mid x, t)\, p(x, t \mid x_0, t_0)\, dx'\, dx
\Biggr].
\end{aligned}
\tag{5.31}
$$

我们注意到上述方程中最后一个二重积分有哑变量 \(x'\) 和 \(x\)。我们可以交换哑变量，并写

$$
\iint \varphi(x)\, p(x', t+\Delta t \mid x, t)\, p(x, t \mid x_0, t_0)\, dx'\, dx
= \iint \varphi(x')\, p(x, t+\Delta t \mid x', t)\, p(x', t \mid x_0, t_0)\, dx'\, dx.
$$

于是式 (5.31) 中第一和第三个二重积分可以消去。这将给我们留下

$$
\begin{aligned}
&\frac{\partial}{\partial t}\int \varphi(x)\, p(x, t \mid x_0, t_0)\, dx \\
&= \lim_{\Delta t \to 0}\frac{1}{\Delta t}
\iint\sum_{m=1}^{\infty}\frac{(x-x')^m}{m!}\left.\frac{\partial^m \varphi}{\partial x^m}\right|_{x=x'}
p(x, t+\Delta t \mid x', t)\, p(x', t \mid x_0, t_0)\, dx'\, dx.
\end{aligned}
\tag{5.32}
$$

现在让我们定义

$$
D^{(m)}(x', t) = \frac{1}{m!}\lim_{\Delta t \to 0}\frac{1}{\Delta t}\int (x-x')^m p(x, t+\Delta t \mid x', t)\, dx.
$$

于是主方程变成

$$
\begin{aligned}
\frac{\partial}{\partial t}\int \varphi(x)\, p(x, t \mid x_0, t_0)\, dx
&= \sum_{m=1}^{\infty}\int D^{(m)}(x', t)\left.\frac{\partial^m}{\partial x^m}\varphi(x)\right|_{x=x'} p(x', t \mid x_0, t_0)\, dx'
&& \text{代入 } D^{(m)} \\
&= \sum_{m=1}^{\infty}\int D^{(m)}(x', t)\, p(x', t \mid x_0, t_0)\left.\frac{\partial^m \varphi}{\partial x^m}\right|_{x=x'}\, dx'
&& \text{交换求和与积分} \\
&= \sum_{m=1}^{\infty}(-1)^m\int \varphi(x')\, \frac{\partial^m}{\partial x^m}\bigl[D^{(m)}(x, t)\, p(x, t, x_0, t_0)\bigr]\, dx'
&& \text{广义分部积分},
\end{aligned}
$$

其中最后一步称为广义分部积分，它指出对任何连续可微函数 \(f\) 和 \(g\)，

$$
\int g\cdot \frac{\partial^m f}{\partial x^m}\, dx = (-1)^m\int f\cdot \frac{\partial^m g}{\partial x^m}\, dx.
$$

把这些合在一起，并认出上述结果对任何任意 \(\varphi(x)\) 成立，于是有

$$
\frac{\partial}{\partial t} p(x, t \mid x_0, t_0)
= \sum_{m=1}^{\infty}(-1)^m \frac{\partial^m}{\partial x^m}\bigl[D^{(m)}(x, t)\, p(x, t, x_0, t_0)\bigr]. \tag{5.33}
$$

如果我们进一步丢掉对 \((x_0, t_0)\) 的条件化，将得到

$$
\frac{\partial}{\partial t} p(x, t)
= \sum_{m=1}^{\infty}(-1)^m \frac{\partial^m}{\partial x^m}\bigl[D^{(m)}(x, t)\, p(x, t)\bigr], \tag{5.34}
$$

证明完成。

Kramers–Moyal 展开通过空间导数 \(\partial_x\) 表达任何马尔可夫过程（包括非线性朗之万方程的解）的概率分布的时间导数 \(\partial_t\)。然而展开有无穷多项。现在一个重要问题是：我们是否被允许截断这些项中的任何项。如果是，可以截断多少项？Pawula 定理为这个问题提供了一个答案 [30]：

**定理 5.8. Pawula 定理。** Kramers–Moyal 展开可以在下列三种情形之一停止：

- \(m = 1\)：所得微分方程称为刘维尔方程，它是一个确定性过程。
- \(m = 2\)：所得微分方程称为福克–普朗克方程。
- \(m = \infty\)，即展开不能被截断。

**证明.** 回忆 Kramers–Moyal 系数定义为

$$
D^{(m)}(x, t) = \frac{1}{m!}\lim_{\Delta t \to 0}\frac{1}{\Delta t}\mathbb{E}\bigl[(\xi(t+\Delta t)-x)^m\bigr]_{\xi(t)=x}.
$$

记 \(x' = \xi(t+\Delta t)\)，这里感兴趣的对象是第 \(m\) 阶矩 \(\mathbb{E}[(x'-x)^m]\)。

我们应用 Cauchy–Schwarz 不等式，它指出对任何函数 \(f\) 和 \(g\) 以及随机变量 \(X\) 和 \(Y\)，有

$$
\mathbb{E}[f(X)g(Y)]^2 \le \mathbb{E}[f(X)^2]\, \mathbb{E}[g(Y)^2].
$$

我们考虑两种可能性：

- 假设 \(m \ge 3\) 且 \(m\) 为奇数，则

$$
\mathbb{E}[(x'-x)^m]^2
= \mathbb{E}\left[(x'-x)^{(m-1)/2}(x'-x)^{(m+1)/2}\right]^2
\le \mathbb{E}[(x'-x)^{m-1}]\, \mathbb{E}[(x'-x)^{m+1}].
$$

- 假设 \(m \ge 4\) 且 \(m\) 为偶数，则

$$
\mathbb{E}[(x'-x)^m]^2
= \mathbb{E}\left[(x'-x)^{(m-2)/2}(x'-x)^{(m+2)/2}\right]^2
\le \mathbb{E}[(x'-x)^{m-2}]\, \mathbb{E}[(x'-x)^{m+2}].
$$

注意我们不能对 \(m = 0, 1, 2\) 应用上述论证，因为它们会给出平凡等式。

从这两个关系出发，并假设我们记 \(D_m = D^{(m)}(x, t)\)，则上述两种情形可以写成

$$
D_m^2 \le D_{m-1} D_{m+1}, \quad m \text{ 为奇数且 } m \ge 3,
\qquad
D_m^2 \le D_{m-2} D_{m+2}, \quad m \text{ 为偶数且 } m \ge 4.
$$

我们现在的目标是证明这个递推关系将给出对任何 \(m \ge 3\) 有 \(D_m = 0\)。

首先假设 \(D_4 = 0\)。则 \(D_6 \le D_4 D_8\) 蕴含 \(D_6 = 0\)。但若 \(D_6 = 0\)，则 \(D_8 \le D_6 D_{10}\) 蕴含 \(D_8 = 0\)。重复这一过程将给出对 \(m = 4, 6, 8, 10, \ldots\) 有 \(D_m = 0\)。类似地，假设 \(D_6 = 0\)。则 \(D_4 \le D_2 D_6\) 蕴含 \(D_4 = 0\)。但若 \(D_4 = 0\)，我们可以回到第一种情形并证明对 \(m = 4, 6, 8, 10, \ldots\) 有 \(D_m = 0\)。一般而言，若这些偶数 \(m \ge 4\) 中任何一个为零，则所有偶数 \(m \ge 4\) 都应为零。对奇数 \(m\)：若 \(D_4 = 0\)，则 \(D_3 \le D_2 D_4\) 蕴含 \(D_3 = 0\)。类似地，若 \(D_6 = 0\)，我们将有 \(D_5 = 0\)。因此若 \(D_4 = D_6 = D_8 = \cdots = 0\)，则 \(D_3 = D_5 = D_7 = \cdots = 0\)。因此，若对任何偶数 \(m\) 使得 \(m \ge 4\) 有 \(D_m = 0\)，则对所有整数 \(m \ge 3\) 有 \(D_m = 0\)。

上述分析表明，如果 Kramers–Moyal 展开被截断到 \(m = 3\)，使得 \(D_3 \neq 0\) 且 \(D_4 = D_5 = \cdots = 0\)，那么 \(D_4 = 0\) 将迫使 \(D_3 = 0\)。于是我们将对所有 \(m \ge 3\) 有 \(D_m = 0\)。类似地，如果 Kramers–Moyal 展开被截断到 \(m = 4\)，使得 \(D_4 \neq 0\) 且 \(D_5 = D_6 = \cdots = 0\)，那么 \(D_6 = 0\) 将迫使 \(D_4 = 0\)。于是我们将再次对所有 \(m \ge 3\) 有 \(D_m = 0\)。

通过对其他 \(m \ge 3\) 重复上述论证，我们看到不可能让 Kramers–Moyal 展开对任何 \(m \ge 3\) 被截断。换言之，我们要么在 \(m = 1\)、\(m = 2\) 处截断展开，要么永远不截断它。

Pawula 定理并不是说福克–普朗克方程（把 Kramers–Moyal 展开截断到 \(m = 2\)）是对底层主方程的好近似。它只是说我们要么可以用 \(m = 1\) 或 \(m = 2\) 精确近似主方程，要么根本不能近似。

### 5.4 福克–普朗克方程

我们现在可以讨论福克–普朗克方程。福克–普朗克方程是用 \(m = 2\) 截断 Kramers–Moyal 展开。

**定义 5.4.** 福克–普朗克方程通过把 Kramers–Moyal 展开截断到 \(m = 2\) 得到。也就是说，对任何马尔可夫过程 \(\xi(t)\)，\(\xi(t) = x\) 在时刻 \(t\) 的概率分布 \(p(x, t)\) 将满足下列偏微分方程：

$$
\frac{\partial}{\partial t} p(x, t)
= -\frac{\partial}{\partial x}\bigl[D^{(1)}(x, t)\, p(x, t)\bigr]
+ \frac{\partial^2}{\partial x^2}\bigl[D^{(2)}(x, t)\, p(x, t)\bigr]. \tag{5.35}
$$

福克–普朗克方程是任何马尔可夫随机过程的一般结果，因为它是 Chapman–Kolmogorov 方程和主方程的推论。我们在本教程中研究的过程，例如朗之万方程，是这个大随机过程家族的特例。因此，如果我们有一个朗之万方程，其解 \(\xi(t)\) 必然有一个满足福克–普朗克方程的概率分布。

**非线性朗之万方程。** 如果我们聚焦于非线性朗之万方程

$$
\dot{\xi} = h(\xi, t) + g(\xi, t)\Gamma(t),
$$

我们可以计算 Kramers–Moyal 系数 \(D^{(m)}(x, t)\)。下面的定理总结这些系数。我们注记，在这个定理的证明过程中，为什么 \(D^{(m)}(x, t)\) 只限于 \(m = 1\) 和 \(m = 2\) 将变得清楚。

**定理 5.9. 非线性朗之万方程的福克–普朗克方程。** 考虑非线性朗之万方程

$$
\dot{\xi} = h(\xi, t) + g(\xi, t)\Gamma(t),
$$

其中函数为 \(h(\xi, t)\) 和 \(g(\xi, t)\)。这个非线性朗之万方程的福克–普朗克方程将有 Kramers–Moyal 系数：

$$
\text{（漂移）}\quad D^{(1)}(x, t) = h(x, t) + g'(x, t)\, g(x, t).
\tag{5.36}
$$

$$
\text{（扩散）}\quad D^{(2)}(x, t) = g^2(x, t).
\tag{5.37}
$$

**证明.** 回忆 Kramers–Moyal 系数的定义：

$$
D^{(m)}(x, t) = \frac{1}{m!}\lim_{\tau \to 0}\frac{1}{\tau}\mathbb{E}\bigl[(\xi(t+\tau)-x)^m\bigr]_{\xi(t)=x}.
$$

困难的部分是如何计算矩 \(\mathbb{E}[(\xi(t+\tau)-x)^m]\)。

我们从看非线性朗之万方程开始：

$$
\dot{\xi} = h(\xi, t) + g(\xi, t)\Gamma(t).
$$

把 \(\xi(t+\tau) - \xi(t) = \int_t^{t+\tau}\dot{\xi}(t')\, dt'\) 对小的 \(\tau\) 写出，并令 \(\xi(t) = x\)，我们可以写

$$
\xi(t+\tau) - x = \int_t^{t+\tau}\bigl[h(\xi(t'), t') + g(\xi(t'), t')\Gamma(t')\bigr]\, dt'.
$$

我们假设 \(h\) 和 \(g\) 可以展开为

$$
\begin{aligned}
h(\xi(t'), t') &= h(x, t') + h'(x, t')(\xi(t')-x) + \cdots, \\
g(\xi(t'), t') &= g(x, t') + g'(x, t')(\xi(t')-x) + \cdots.
\end{aligned}
$$

这将给出

$$
\begin{aligned}
\xi(t+\tau) - x
&= \int_t^{t+\tau} h(x, t')\, dt' + \int_t^{t+\tau} h'(x, t')(\xi(t')-x)\, dt' + \cdots \\
&\quad + \int_t^{t+\tau} g(x, t')\Gamma(t')\, dt' + \int_t^{t+\tau} g'(x, t')(\xi(t')-x)\Gamma(t')\, dt' + \cdots.
\end{aligned}
$$

现在，我们迭代上述方程，并用积分替换 \(\xi(t') - x\)。这将给出只涉及 \(h\)、\(g\) 和 \(\Gamma\) 的项（涉及 \(\xi(t'')-x\) 的项并不丢掉）：

$$
\begin{aligned}
\xi(t+\tau) - x
&= \int_t^{t+\tau} h(x, t')\, dt'
+ \int_t^{t+\tau} h'(x, t')\left(\int_t^{t'} h(x, t'')\, dt''\right) dt' \\
&\quad + \int_t^{t+\tau} h'(x, t')\left(\int_t^{t'} g(x, t'')\Gamma(t'')\, dt''\right) dt' + \cdots \\
&\quad + \int_t^{t+\tau} g(x, t')\Gamma(t')\, dt'
+ \int_t^{t+\tau} g'(x, t')\left(\int_t^{t'} h(x, t'')\, dt''\right)\Gamma(t')\, dt' \\
&\quad + \int_t^{t+\tau} g'(x, t')\left(\int_t^{t'} g(x, t'')\Gamma(t'')\, dt''\right)\Gamma(t')\, dt' + \cdots.
\end{aligned}
\tag{5.38}
$$

取期望，并注意到 \(\mathbb{E}[\Gamma(t)] = 0\)，我们可以证明式 (5.38) 中只有前两项和最后一项会存活。因此我们有

$$
\begin{aligned}
\mathbb{E}[\xi(t+\tau)-x]
&= \int_t^{t+\tau} h(x, t')\, dt'
+ \int_t^{t+\tau}\int_t^{t'} h'(x, t') h(x, t'')\, dt''\, dt' + \cdots \\
&\quad + \int_t^{t+\tau} g'(x, t')\left(\int_t^{t'} g(x, t'')\, 2\delta(t''-t')\, dt''\right) dt' + \cdots,
\end{aligned}
\tag{5.39}
$$

其中内层积分等于 \(g(x, t')\)，我们遵循 Risken 的定义 \(\int_t^{t'} 2\delta(t''-t')\, dt'' = 1\) [33]。当 \(\tau \to 0\) 时，式 (5.39) 的第一项和第三项是

$$
\lim_{\tau \to 0}\int_t^{t+\tau} h(x, t')\, dt' = h(x, t),
\qquad
\lim_{\tau \to 0}\int_t^{t+\tau} g'(x, t') g(x, t')\, dt' = g'(x, t) g(x, t).
$$

对式 (5.39) 中的第二项，我们可以证明它在 \(\tau \to 0\) 时为零：若 \(h\) 关于 \(t\) 可积，定义 \(H(x, t) = \int_0^t h(x, t')\, dt'\)，则

$$
\lim_{\tau \to 0}\int_t^{t+\tau} h'(x, t')\bigl(H(x, t') - H(x, t)\bigr)\, dt' = h'(x, t)\bigl(H(x, t) - H(x, t)\bigr) = 0.
$$

因此我们到达

$$
D^{(1)}(x, t) = h(x, t) + g'(x, t)\, g(x, t).
$$

\(D^{(2)}(x, t)\) 的推导本质上遵循同一组论证。这里要注意的关键是，当我们在 \(\mathbb{E}[(\xi(t+\tau)-x)^2]\) 中取平方时，式 (5.39) 中的积分将给出与 \(\tau^2\) 成比例的贡献。当 \(\tau \to 0\) 时，所有这些项将消失，因为 \(D^{(2)}(x, t)\) 的定义中只有一个 \(1/\tau\)。结果，唯一能存活的项是

$$
\begin{aligned}
D^{(2)}(x, t)
&= \frac{1}{2}\lim_{\tau \to 0}\frac{1}{\tau}\int_t^{t+\tau}\int_t^{t+\tau} g(x, t') g(x, t'')\, 2\delta(t'-t'')\, dt'\, dt'' \\
&= g^2(x, t),
\end{aligned}
$$

证明完成。

**例 5.4.** 考虑下列朗之万方程

$$
\dot{\xi} = A(\xi)\xi + \sigma\Gamma(t).
$$

于是解 \(\xi(t)\) 的概率分布 \(p(x, t)\) 将满足下列福克–普朗克方程

$$
\begin{aligned}
\frac{\partial}{\partial t} p(x, t)
&= -\frac{\partial}{\partial x}\bigl[(h(x, t) + g'(x, t) g(x, t))\, p(x, t)\bigr]
+ \frac{\partial^2}{\partial x^2}\bigl[g^2(x, t)\, p(x, t)\bigr] \\
&= -\frac{\partial}{\partial x}\bigl[(A(x) + 0\cdot\sigma)\, p(x, t)\bigr]
+ \frac{\partial^2}{\partial x^2}\bigl[\sigma^2 p(x, t)\bigr] \\
&= -\frac{\partial}{\partial x}\bigl[A(x)\, p(x, t)\bigr] + \sigma^2\frac{\partial^2}{\partial x^2}[p(x, t)].
\end{aligned}
$$

**例 5.5.** 对 \(A(x) = 0\) 的特殊情形，朗之万方程简化为维纳过程：

$$
\dot{\xi} = \sigma\Gamma(t).
$$

对应的福克–普朗克方程是

$$
\frac{\partial}{\partial t} p(x, t) = \sigma^2\frac{\partial^2}{\partial x^2} p(x, t).
$$

这个方程称为热方程或扩散方程。若初始条件是 \(p(x, 0) = \delta(x)\)，解是（见下面的推导）

$$
p(x, t) = \frac{1}{\sqrt{4\pi\sigma^2 t}}\, e^{-x^2/(4\sigma^2 t)}.
$$

**热方程的解。** 热方程可以用傅里叶变换求解。为记号简单，我们记 \(u_t = \partial u/\partial t\) 且 \(u_{xx} = \partial^2 u/\partial x^2\)。考虑一个一般热方程：

$$
u_t(x, t) = k u_{xx}(x, t), \qquad x \in \mathbb{R},\ t > 0,
$$

初始条件为 \(u(x, 0) = \varphi(x)\)。我们可以在两边取傅里叶变换（在 \(x \leftrightarrow \omega\) 之间映射），定义

$$
\hat{u}_t(\omega, t) = \mathcal{F}\{u_t(x, t)\} = \int_{-\infty}^{\infty} u_t(x, t) e^{j\omega x}\, dx,
\qquad
\hat{u}_{xx}(\omega, t) = \mathcal{F}\{u_{xx}(x, t)\} = \int_{-\infty}^{\infty} u_{xx}(x, t) e^{j\omega x}\, dx.
$$

这将给出

$$
\hat{u}_t(\omega, t) = k\hat{u}_{xx}(\omega, t).
$$

利用傅里叶变换的微分性质，我们可以把方程右边写成

$$
k\hat{u}_{xx}(\omega, t) = k(j\omega)^2\hat{u}(\omega, t) = -k\omega^2\hat{u}(\omega, t),
$$

这将给我们一个常微分方程

$$
\hat{u}_t(\omega, t) = -k\omega^2\hat{u}(\omega, t). \tag{5.40}
$$

这个（关于 \(t\) 的）微分方程的解由下式给出

$$
\hat{u}(\omega, t) = \hat{\varphi}(\omega)\, e^{-k\omega^2 t}. \tag{5.41}
$$

（注：式 (5.40) 只是一个简单微分方程 \(f'(t) = a f(t)\)，其解可以通过积分找到。）因此，如果我们对 \(\hat{u}(\omega, t)\) 取傅里叶逆变换（关于 \(\omega \leftrightarrow x\)），将有

$$
u(x, t) = \mathcal{F}^{-1}\{\hat{u}(\omega, t)\} = \mathcal{F}^{-1}\{\hat{\varphi}(\omega) e^{-k\omega^2 t}\},
$$

这是 \(\hat{\varphi}(\omega)\) 与 \(\hat{f}(\omega) \stackrel{\mathrm{def}}{=} e^{-k\omega^2 t}\) 之积的傅里叶逆变换。由于傅里叶域中的乘法是空间域中的卷积，于是 \(u(x, t)\) 是 \(\varphi(x)\) 与 \(f(x) = \mathcal{F}^{-1}(e^{-k\omega^2 t})\) 的卷积。而 \(\mathcal{F}^{-1}(e^{-k\omega^2 t}) = \frac{1}{\sqrt{2kt}} e^{-x^2/(4kt)}\)。因此我们可以证明解是

$$
\begin{aligned}
u(x, t)
&= \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\varphi(x-x') f(x')\, dx' \\
&= \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\varphi(x-x')\frac{1}{\sqrt{2kt}} e^{-(x')^2/(4kt)}\, dx'.
\end{aligned}
$$

因此，若 \(\varphi(x) = \delta(x)\)，从而 \(\varphi(x-x') = \delta(x-x')\)，则有

$$
u(x, t) = \frac{1}{\sqrt{4k\pi t}}\, e^{-x^2/(4kt)}. \tag{5.42}
$$

**概率流。** 福克–普朗克方程有一些有趣的物理解释。回忆福克–普朗克方程是

$$
\frac{\partial}{\partial t} p(x, t)
= -\frac{\partial}{\partial x}\bigl[D^{(1)}(x, t)\, p(x, t)\bigr]
+ \frac{\partial^2}{\partial x^2}\bigl[D^{(2)}(x, t)\, p(x, t)\bigr]. \tag{5.43}
$$

让我们定义一个量

$$
S(x, t) = \left[D^{(1)}(x, t) - \frac{\partial}{\partial x} D^{(2)}(x, t)\right] p(x, t). \tag{5.44}
$$

于是福克–普朗克方程可以写成

$$
\frac{\partial p}{\partial t} + \frac{\partial S}{\partial x} = 0. \tag{5.45}
$$

解释式 (5.45) 的一种方式是它是概率流。

**式 (5.45) 的直观推导。** 能量守恒告诉我们，如果某个空间区域中有增加/减少（例如粒子或电荷），数量的变化应当等于其表面处的变化。因此，如果 \(p(x, t)\) 表示某种密度，那么 \(p(x, t)\, dx\) 将是时刻 \(t\) 位于 \((x, x+dx)\) 之间的粒子数量。这里 \(S(x, t)\) 可以看作单位时间穿过 \(x\) 的粒子流。对时间区间 \((t, t+dt)\) 和空间区间 \((x, x+dx)\)，粒子数量的变化是

$$
\text{增加/减少的粒子数} = [p(x, t+dt) - p(x, t)]\, dx.
$$

由于能量守恒，要使这一变化发生，必须有某种流的流动。区间上的流是

$$
\text{流入/流出的粒子数} = [S(x, t) - S(x+dx, t)]\, dt.
$$

令二者相等，我们将得到

$$
[p(x, t+dt) - p(x, t)]\, dx = [S(x, t) - S(x+dx, t)]\, dt.
$$

一阶泰勒近似将给出 \(p(x, t+dt) - p(x, t) = \partial p/\partial t\) 且 \(S(x, t) - S(x+dx, t) = -\partial S/\partial x\)。这将给我们

$$
\frac{\partial p}{\partial t} + \frac{\partial S}{\partial x} = 0. \tag{5.46}
$$

因此，福克–普朗克方程可以被看作能量守恒的一种形式，其中 \(p\) 的变化（关于时间）应当等于 \(S\) 的变化（关于空间）。

**平衡解。** 在平衡态，流消失，因此我们有 \(S = 0\)。于是我们可以证明下列结果。

**定理 5.10.** 在平衡态，由于概率流消失，概率分布 \(p(x)\) 满足

$$
D^{(1)}(x, t)\, p(x) = \frac{\partial}{\partial x}\bigl[D^{(2)}(x, t)\, p(x)\bigr].
$$

**例 5.6.** 例如，若 \(D^{(1)} = -\gamma x\) 且 \(D^{(2)} = \gamma kT/m\)，则 \(S = 0\) 将给出

$$
\left(-\gamma x - \frac{\gamma kT}{m}\frac{\partial}{\partial x}\right) p(x) = 0.
$$

于是 \(-\gamma x\, p(x) = \frac{\gamma kT}{m}\partial_x p(x)\)。求解这个微分方程将给出

$$
p(x) = \sqrt{\frac{m}{2\pi kT}}\, e^{-m x^2/(2kT)}.
$$

**例 5.7. 与 SMLD 的联系。** 让我们试着把我们的结果与定义 3.1 中定义的式 (3.1) 对应起来。我们将考虑一维情形。考虑下列朗之万方程

$$
\underbrace{\frac{\partial x}{\partial t}}_{\dot{\xi}}
= \underbrace{\tau\frac{\partial}{\partial x}\log p(x)}_{h(\xi, t)}
+ \underbrace{\sigma}_{g(\xi, t)}\Gamma(t).
$$

为避免记号混淆，我们令 \(W(x, t)\) 为这个朗之万方程的解 \(x(t)\) 的概率分布。这个朗之万方程的 Kramers–Moyal 系数是

$$
\begin{aligned}
D^{(1)}(x, t) &= h(x, t) + g'(x, t) g(x, t) = \tau\frac{\partial}{\partial x}\log p(x) \stackrel{\mathrm{def}}{=} A(x), \\
D^{(2)}(x, t) &= g(x, t)^2 = \sigma^2.
\end{aligned}
$$

因此对应的福克–普朗克方程是

$$
\begin{aligned}
\frac{\partial}{\partial t} W(x, t)
&= -\frac{\partial}{\partial x}\bigl[D^{(1)}(x, t)\, W(x, t)\bigr]
+ \frac{\partial^2}{\partial x^2}\bigl[D^{(2)}(x, t)\, W(x, t)\bigr] \\
&= -\frac{\partial}{\partial x}\bigl[A(x)\, W(x, t)\bigr] + \sigma^2\frac{\partial^2}{\partial x^2} W(x, t).
\end{aligned}
$$

在平衡态 \(t \to \infty\) 时，概率分布 \(W(x, t)\) 可以写成 \(W(x)\)。由于概率流消失，于是有

$$
A(x)\, W(x) = \sigma^2\frac{\partial}{\partial x} W(x).
$$

回忆 \(A(x) = \tau \partial_x \log p(x)\)，于是有

$$
\tau\frac{\partial}{\partial x}\log p(x) = \sigma^2\frac{\partial}{\partial x} W(x).
$$

由于我们有选择 \(\sigma\) 的自由，我们就令 \(\sigma = \sqrt{\tau}\)。于是上述方程简化为 \(\partial_x \log p(x) = \partial_x W(x)\)。两边关于 \(x\) 积分将给出

$$
\log p(x) = W(x) + C, \tag{5.47}
$$

对某个常数 \(C\)。令 \(U(x) = e^{W(x)}\) 为一个概率分布，从而 \(W(x) = \log U(x)\) 是对数似然，式 (5.47) 将给出 \(p(x) = U(x) e^{C}\)。由于 \(p(x)\) 和 \(U(x)\) 都是概率分布，我们必须有 \(\int p(x)\, dx = 1\) 且 \(\int U(x)\, dx = 1\)。因此我们可以证明 \(C = 0\)。

因此我们得出：如果我们把朗之万方程运行到收敛，解的概率分布 \(W(x)\) 恰好是真值分布 \(p(x)\)。此外，噪声水平 \(\sigma\) 与步长 \(\tau\) 由 \(\sigma = \sqrt{\tau}\) 相关联。

### 5.5 结语

本节我们讨论了布朗运动、朗之万方程和福克–普朗克方程背后的物理，并演示了统计物理文献中的若干经典定理。我们的许多结果是一般的，因为它们可以应用于任何马尔可夫随机过程。超出马尔可夫过程可以通过把空间/时间相关限制在一个小区间来完成。

关于这一主题有大量参考文献，其中许多起源于物理学。Risken 的教科书 [33] 是这一主题的经典参考，它包含本质上所有成分。对寻找稍更一般内容的读者，Reichl 的统计物理书 [31] 可以满足目的。

---

## 6 结论

本教程涵盖了近期文献中基于扩散的生成模型所依托的若干基本概念。我们发现，深入这些基本原理，而不是停留在 Python 编程的表面，特别重要。

在撰写本教程时，我们学到的几条教训值得分享。DDPM 的发展在许多意义上是 VAE 的扩展，无论是结构、证据下界的使用，还是重参数化技巧。虽然 DDPM 仍然是最先进的方法之一，如果未来的模型能够避免任何迭代，会更理想。一些近期论文开始探索用知识蒸馏减少迭代次数的可行性。一些论文通过借鉴微分方程文献中的想法来研究加速方法。

对感兴趣于成像的读者，分数匹配朗之万动力学很可能继续在求解反问题中扮演基础角色。分数匹配函数的训练几乎与训练一个图像去噪器相同，而应用一个分数匹配步骤与一个去噪步骤相同。因此，一旦我们知道如何把反问题拆成前向模型和先验分布，就能够利用分数匹配来做后验采样。近期方法也开始探索贝叶斯扩散模型的不同形式，以连接物理模型与数据驱动模型。

SDE 和福克–普朗克方程提供了很好的理论直觉，说明 SMLD 方程为什么以某种方式被推导出来。从历史上说，SMLD 的发展似乎并不是从 SDE 开始的，但今天的认识帮助我们理解解的统计行为。

展望未来，扩散模型最大的挑战是与物理世界的一致性，更不必说它们很高的计算复杂度。类别特定的学习将继续有影响力。例如，人们可以用手机相册中的图像训练一个定制的扩散模型。时间一致性会要求更大的模型、用更多内存来存储帧。需要新的架构来利用越来越多的时间输入。另一个开放问题是如何把语言和语义带入图像生成。图像是否应当继续被表示成像素（或特征像素）的数组，或者是否有新的方式用几个词描述场景而不丢失信息？最后，信息取证将是未来几十年最大的挑战，直到我们能够发展出有效的对策（或政策）。

## 致谢

本工作部分由美国国家科学基金会奖项 2030570、2134209 和 2133032 支持，以及 SRC JUMP 2.0 中心和 Samsung Research America 的研究奖。自这份草稿于 2024 年 3 月发布到互联网以来，我们收到了来自世界各地读者的大量建设性反馈。感谢大家的意见。也感谢普渡大学许多研究生，他们就教程内容分享了好的想法。我们要特别感谢 William Chi-Kin Yau，他不知疲倦地与我们一起完成了关于朗之万方程和福克–普朗克方程的那一节。

---

## References

参考文献按原文保留，不翻译。

[1] Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying
     framework for flows and diffusions. https://arxiv.org/abs/2303.08797.
 [2] Brian Anderson. Reverse-time diffusion equation models. Stochastic Process. Appl., 12(3):313–326, May
     1982. https://www.sciencedirect.com/science/article/pii/0304414982900515.

 [3] Kendall Atkinson, Weimin Han, and David Stewart. Numerical solution of ordinary differential equa-
     tions. Wiley, 2009. https://homepage.math.uiowa.edu/~atkinson/papers/NAODE_Book.pdf.
 [4] Christopher Bishop. Pattern Recognition and Machine Learning. Springer, 2006.
 [5] Charles A. Bouman and Gregery T. Buzzard. Generative plug and play: Posterior sampling for inverse
     problems. In 2023 59th Annual Allerton Conference on Communication, Control, and Computing
     (Allerton), pages 1–7, 2023. https://arxiv.org/abs/2306.07233.
 [6] Robert Brown. A brief account of microscopical observations on the particles contained in the pollen
     of plants and the general existence of active molecules in organic and inorganic bodies. Edinburgh New
     Philosophical Journal, pages 358–371, 1828.

 [7] Stanley H. Chan. Introduction to Probability for Data Science. Michigan Publishing, 2021. https:
     //probability4datascience.com/.
 [8] Stanley H. Chan, Xiran Wang, and Omar Elgendy. Plug-and-Play ADMM for image restoration: Fixed
     point convergence and applications. IEEE Trans. Computational Imaging, 3(5):84–98, Mar 2017. https:
     //arxiv.org/abs/1605.01710.

 [9] Hyungjin Chung and Jong Chul Ye. Score-based diffusion models for accelerated mri. Medical Image
     Analysis, 80:102479, 2022. https://arxiv.org/abs/2110.05243.
[10] Peter Dayan, Geoffrey E. Hinton, Radford M. Neal, and Richard S. Zemel. The Helmholtz machine.
     Neural Computation, 7(5):889–904, 1995.

[11] Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising
     diffusion for image restoration. Transactions on Machine Learning Research (TMLR), 2023. https:
     //openreview.net/forum?id=VmyFF5lL3F.
[12] Carl Doersch. Tutorial on variational autoencoders, 2016. https://arxiv.org/abs/1606.05908.
[13] Luca Donati. From Chapman-Kolmogorov equation to Master equation and Fokker-Planck equation.
     https://www.zib.de/userpage/donati/stochastics2023/03/lecture_notes/L03_dCKeq.pdf.
[14] Albert Einstein. On the movement of small particles suspended in stationary liquids required by the
     molecular-kinetic theory of heat. Annalen der Physik, pages 549–560, 1905.
[15] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair,
     Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information
     Processing Systems (NeurIPS), volume 27, 2014. https://arxiv.org/abs/1406.2661.
[16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in
     Neural Information Processing Systems (NeurIPS), 2020. https://arxiv.org/abs/2006.11239.
[17] Jason Hu, Bowen Song, Xiaojian Xu, Liyue Shen, and Jeffrey A. Fessler. Learning image priors through
     patch-based diffusion models for solving inverse problems, 2024. https://arxiv.org/abs/2406.02462.
[18] Aapo Hyvärinen. Estimation of non-normalized statistical models by score matching. Journal
     of Machine Learning Research (JMLR), 6(24):695–709, 2005. https://jmlr.org/papers/volume6/
     hyvarinen05a/hyvarinen05a.pdf.

[19] Zahra Kadkhodaie and Eero P. Simoncelli. Solving linear inverse problems using the prior implicit in a
     denoiser, 2020. https://arxiv.org/abs/2007.13640.
[20] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-
     based generative models. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
     https://arxiv.org/abs/2206.00364.
[21] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models.
     In Advances in Neural Information Processing Systems (NeurIPS), 2022. https://arxiv.org/abs/
     2201.11793.
[22] Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In
     Advances in Neural Information Processing Systems (NeurIPS), 2021. https://arxiv.org/abs/2107.
     00630.
[23] Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. In International Conference
     on Learning Representations (ICLR), 2014. https://openreview.net/forum?id=33X9fd2-9FyZd.

[24] Diederik P. Kingma and Max Welling. An introduction to variational autoencoders. Foundations and
     Trends in Machine Learning, 12(4):307–392, 2019. https://arxiv.org/abs/1906.02691.
[25] Andrey Kolmogorov.     Foundations of the Theory of Probability. Dover, 2018. The orig-
     inal version was published in 1933 in German. https://dn790007.ca.archive.org/0/items/
     foundationsofthe00kolm/foundationsofthe00kolm.pdf.

[26] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver: A fast ODE
     solver for diffusion probabilistic model sampling in around 10 steps. In Advances in Neural Information
     Processing Systems (NeurIPS), 2022. https://arxiv.org/abs/2206.00927.
[27] Calvin Luo. Understanding diffusion models: A unified perspective, 2022. https://arxiv.org/abs/
     2208.11970.

[28] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim
     Salimans. On distillation of guided diffusion models. In IEEE Conference on Computer Vision and
     Pattern Recognition (CVPR), pages 14297–14306, 2023. https://arxiv.org/abs/2210.03142.
[29] Gabriel Nagy. MTH 235 differential equations, 2024. https://users.math.msu.edu/users/gnagy/
     teaching/ade.pdf.
[30] R. Pawula. Generalizations and extensions of the Fokker-Planck-Kolmogorov equations. IEEE Trans-
     actions on Information Theory, 13(1):33–41, 1967.
[31] L. E. Reichl. A Modern Course in Statistical Physics. John Wiley and Sons, Inc, 2 edition, 1998.

[32] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In Proceedings of
     International Conference on Machine Learning (ICML), pages 1530–1538, 2015. https://arxiv.org/
     abs/1505.05770.
[33] Hannes Risken. The Fokker-Planck Equations: Methods of solutions and applications. Springer, 2
     edition, 1989.

[34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-
     resolution image synthesis with latent diffusion models. In IEEE Conference on Computer Vision
     and Pattern Recognition (CVPR), pages 10684–10695, 2022. https://arxiv.org/abs/2112.10752.
[35] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar
     Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J
     Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language un-
     derstanding. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages
     36479–36494, 2022. https://arxiv.org/abs/2205.11487.

[36] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In
     International Conference on Learning Representations (ICLR), 2022. https://arxiv.org/abs/2202.
     00512.
[37] Yash Sanghvi, Yiheng Chi, and Stanley H. Chan. Kernel diffusion: An alternate approach to blind
     deconvolution. In European Conference on Computer Vision (ECCV), 2024. https://arxiv.org/abs/
     2312.02319.
[38] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised
     learning using nonequilibrium thermodynamics. In Proceedings of International Conference on Machine
     Learning (ICML), volume 37, pages 2256–2265, 2015. https://arxiv.org/abs/1503.03585.

[39] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Interna-
     tional Conference on Learning Representations (ICLR), 2023. https://openreview.net/forum?id=
     St1giarCHLP.
[40] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution.
     In Advances in Neural Information Processing Systems (NeurIPS), 2019. https://arxiv.org/abs/
     1907.05600.
[41] Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. In
     Advances in Neural Information Processing Systems (NeurIPS), 2020. https://arxiv.org/abs/2006.
     09011.

[42] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and
     Ben Poole. Score-based generative modeling through stochastic differential equations. In Interna-
     tional Conference on Learning Representations (ICLR), 2021. https://openreview.net/forum?id=
     PxTIG12RRHS.
[43] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Compu-
     tation, 23(7):1661–1674, 2011. https://www.iro.umontreal.ca/~vincentp/Publications/smdae_
     techreport.pdf.
[44] M. von Smoluchowski. Zur kinetischen theorie der brownschen molekularbewegung und der suspensio-
     nen. Annalen der Physik, pages 756–780, 1906.
[45] Martin J. Wainwright and Michael I. Jordan. Graphical models, exponential families, and variational
     inference. Foundations and Trends in Machine Learning, 1(1-2):1–305, 2008. https://cba.mit.edu/
     events/03.11.ASE/docs/Wainwright.1.pdf.
[46] Max Welling and Yee Whye Teh. Bayesian learning via stochastic gradient Langevin dynamics. In
     Proceedings of International Conference on Machine Learning (ICML), pages 681–686, 6 2011. https:
     //www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf.
