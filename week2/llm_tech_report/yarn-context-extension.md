# YaRN：高效扩展大语言模型上下文窗口

## 论文信息

- 论文：**YaRN: Efficient Context Window Extension of Large Language Models**
- 作者：Bowen Peng、Jeffrey Quesnelle、Honglu Fan、Enrico Shippole
- arXiv：<https://arxiv.org/abs/2309.00071>
- 本地 PDF：`2309.00071.pdf`
- 代码：<https://github.com/jquesnelle/yarn>
- 版本：论文 PDF 的 arXiv v3

## 一句话总结

YaRN 是一种针对 RoPE 模型的长上下文扩展方法。它不是重新训练一个新的 Transformer，而是在原模型的 RoPE 上组合三个关键思想：

1. 低频维度做插值，避免位置角度超出预训练范围；
2. 高频维度尽量保持原样，保留模型对局部相对距离的辨别能力；
3. 对 attention logits 做温度校准，恢复扩展上下文后注意力分布的合适尖锐程度。

论文还提出了推理时的 **Dynamic Scaling**，使位置缩放因子随当前序列长度变化。

最终，YaRN 在 LLaMA/Llama 2 上用很少的长上下文微调数据把上下文从 2K/4K 扩展到 32K、64K 甚至 128K，并且比此前的 RoPE 插值方法更省训练计算。

---

## 1. 论文要解决什么问题

### 1.1 预训练长度就是上下文上限

假设一个模型预训练时最多看到长度为 \(L\) 的序列，例如：

```text
原始上下文长度 L = 4K
目标上下文长度 L' = 128K
扩展倍数 s = L' / L = 32
```

如果直接把 128K 的 token 输入原模型，模型虽然在代码层面可能能够接收这些 token，但它没有学习过这么大的位置范围。最先出问题的通常是位置编码，而不是词表或 Transformer 结构本身。

### 1.2 为什么 RoPE 会在长上下文外失效

RoPE 给每个 token 的 Q、K 施加与位置有关的旋转。每一对维度有自己的旋转频率：

$$
\theta_d = b^{-2d/D}, \qquad b=10000,
$$

位置 (m) 对应的旋转角度大致是：

$$
m\theta_d.
$$

训练时，模型只见过 \(m\in[0,L)\)。当推理位置达到 \(L'\) 时，角度分布可能落到训练时没有出现过的区域。高频维度会快速旋转，低频维度则可能出现很长的相位跨度。

关键矛盾是：

- 直接外推：保留原始相对距离，但会产生训练分布外的旋转角度；
- 所有维度都做同样的缩放：避免角度越界，但会破坏一部分本来很重要的高频位置信息。

YaRN 的核心就是不再把所有 RoPE 维度一视同仁。

---

## 2. RoPE 背景

### 2.1 Q、K 的位置旋转

标准 self-attention 先将隐藏状态投影为 Q 和 K：

$$
q_m=W_qx_m, \qquad k_n=W_kx_n.
$$

RoPE 再根据 token 位置旋转它们：

$$
q_m'=R(m\boldsymbol\theta)q_m,
\qquad
k_n'=R(n\boldsymbol\theta)k_n.
$$

其中 (R) 可以看成对每一对坐标使用一个 2D 旋转矩阵：

$$
R(\phi)=
\begin{bmatrix}
\cos\phi & -\sin\phi\\
\sin\phi & \cos\phi
\end{bmatrix}.
$$

attention 分数为：

$$
\operatorname{score}(m,n)=
\frac{q_m'^{\mathsf T}k_n'}{\sqrt D}.
$$

由于旋转矩阵满足正交性，两个位置的相互作用会包含角度差：

$$
m\theta_d-n\theta_d=(m-n)\theta_d.
$$

所以 RoPE 能把相对位置 (m-n) 传给 attention。

### 2.2 频率与波长

论文把第 (d) 个 RoPE 维度的波长定义为：

$$
\lambda_d=\frac{2\pi}{\theta_d}
       =2\pi b^{2d/D}.
$$

波长的含义是：需要经过多少个 token，该维度才完整旋转一周 (2\pi)。

- 波长短、(	heta_d) 大：高频，变化快，擅长表达局部相对距离；
- 波长长、(	heta_d) 小：低频，变化慢，适合表达较长范围的位置变化。

这里的“高频/低频”不是 token 的频率，而是 RoPE 某一对维度的旋转频率。

### 2.3 一个四维直观例子

假设一个 head 只有 4 个维度，那么可以分成两对：

```text
(q0, q1) 使用频率 theta0
(q2, q3) 使用频率 theta1
```

位置从 10 变为 11 时，两对维度都旋转一次自己的角度；但高频对旋转得更快，低频对旋转得更慢。

如果把长度扩展 32 倍：

- 所有维度统一除以 32：高频维度的旋转速度也被大幅降低；
- 所有维度保持不变：位置 100000 处可能出现模型训练时没见过的角度；
- YaRN：长波长维度主要除以 32，短波长维度基本保持原频率，中间维度平滑混合。

---

## 3. 论文讨论的几种方法

YaRN 不是凭空出现的。论文先解释已有方法各自解决了什么问题，然后将它们组合起来。

### 3.1 直接外推

最简单的做法是继续使用原始 RoPE：

$$
g(m)=m.
$$

也就是位置 100000 就使用 100000 对应的旋转角度。

问题是，模型训练时没有见过这些角度分布。即使 RoPE 理论上表达的是相对位置，实际网络也会利用旋转向量在训练区间内的分布来推断位置，因此直接外推通常会在超过原始长度后快速退化。

### 3.2 Position Interpolation（PI）

PI 的思想是把更长的序列压回原始位置范围：

$$
m' = m\frac{L}{L'}=\frac{m}{s},
$$

其中 \(s=L'/L\)。例如原始长度是 4K、目标长度是 128K，则目标位置 64000 会映射到原位置 2000：

$$
64000\times\frac{4K}{128K}=2000.
$$

于是新序列所有旋转角度都落在原模型见过的范围内，通常只需少量长序列微调就能工作。

但是 PI 对所有频率使用同一个缩放倍数：

$$
\theta_d' = \frac{\theta_d}{s}.
$$

扩展倍数很大时，高频维度也被压得非常慢，导致模型失去区分邻近 token 的细粒度位置信息。论文指出，单独使用 PI 时，扩展到约 \(s=8\) 以后性能会明显恶化。

> **符号说明：**论文正文中 PI 的 \(g(m)\) 有一处写成了 \(s\cdot m\)，但附录的位置变换公式和 PI 的实际含义都是 \(mL/L'=m/s\)。实现时应以“把目标位置压回原始长度范围”为准。

### 3.3 NTK-aware interpolation

NTK-aware 的观察是：不必让每个频率都承受相同的插值压力。

- 高频维度少缩放，尽量保留高频信息；
- 低频维度多缩放，使长距离位置不要越界。

论文给出的一个简单实现是增大 RoPE base：

$$
b'=b\cdot s^{D/(D-2)}.
$$

再用新的 base 计算：

$$
\theta_d'={b'}^{-2d/D}.
$$

这相当于把频率缩放压力分散到不同维度，而不是对所有维度统一除以 \(s\)。

优点是非微调模型通常比 PI 更稳定；缺点是 base 的最佳取值很难理论上精确确定，通常需要经验搜索。另外，有些维度可能仍然发生轻微外推，所以它在微调场景不一定优于 PI。

### 3.4 NTK-by-parts

NTK-by-parts 进一步按照每个维度在原始训练长度内转了多少圈来决定处理方式。

定义：

$$
r(d)=\frac{L}{\lambda_d}
    =\frac{L}{2\pi b^{2d/D}}.
$$

\(r(d)\) 可以理解为该维度在原始上下文 \(L\) 内完成的旋转圈数。

- (r(d)) 小：波长长，训练区间内没有完整转很多圈，更容易携带绝对位置线索；
- (r(d)) 大：波长短，训练区间内转了很多圈，主要携带局部相对距离。

论文使用两个阈值 \(\alpha,\beta\)，并定义平滑 ramp：

$$
\gamma(r)=
\begin{cases}
0,&r<\alpha\\
1,&r>\beta\\
\dfrac{r-\alpha}{\beta-\alpha},&\text{otherwise}.
\end{cases}
$$

新的频率为：

$$
\theta_d'
=\left(1-\gamma(r(d))\right)\frac{\theta_d}{s}
 +\gamma(r(d))\theta_d.
$$

因此：

| 维度类型 | \(r(d)\) | \(\gamma\) | 处理方式 | 目的 |
|---|---:|---:|---|---|
| 长波长/低频 | 小于 \(\alpha\) | 0 | \(\theta_d/s\) | 完全插值，避免外推 |
| 中间频率 | \(\alpha\) 到 \(\beta\) | 0 到 1 | 两者平滑混合 | 平衡长距离和局部信息 |
| 短波长/高频 | 大于 \(\beta\) | 1 | 原始 \(\theta_d\) | 保留局部相对距离 |

对 LLaMA 家族，论文实验中使用：

$$
\alpha=1,\qquad\beta=32.
$$

这就是论文所说的“by parts”：不同频率分段处理，而不是所有维度采用同一规则。

---

## 4. YaRN 的完整方法

YaRN = **NTK-by-parts + attention scaling**。

### 4.1 第一部分：按频率插值

首先根据 NTK-by-parts 为每个维度得到 (	heta_d')，再对位置 (m) 使用：

$$
\phi_{m,d}=m\theta_d'.
$$

长波长维度的角度会被压缩，高频维度的角度基本保持原样。

### 4.2 第二部分：调整 attention logits

标准 attention 的权重为：

$$
\operatorname{softmax}
\left(\frac{q_m^{\mathsf T}k_n}{\sqrt D}\right).
$$

YaRN 改为：

$$
\operatorname{softmax}
\left(\frac{q_m^{\mathsf T}k_n}{t\sqrt D}\right).
$$

当 \(t<1\) 时，logit 被放大，softmax 会更尖锐；当 \(t>1\) 时，分布会更平滑。

为什么需要这一步？位置插值会改变 Q/K 的表示分布。上下文变长后，attention logits 的熵会发生变化，模型可能变得过于平均地关注很多 token。论文发现，对 logits 做一个全局温度校准，可以在不同样本和不同 token 位置上比较稳定地改善困惑度。

### 4.3 不修改 attention 代码的实现技巧

论文利用：

$$
\frac{q^{\mathsf T}k}{t\sqrt D}
=\frac{(\sqrt{1/t}q)^{\mathsf T}(\sqrt{1/t}k)}{\sqrt D}.
$$

所以可以把 Q 和 K 都乘以：

$$
c=\sqrt{1/t}.
$$

在 RoPE 的复数形式中，直接把旋转后的 embedding 乘以 (c)，就等价于给 attention logits 加温度，而不必改动 attention kernel。

论文对 LLaMA/Llama 2 推荐：

$$
\sqrt{\frac1t}=0.1\ln(s)+1.
$$

几个例子：

| 扩展倍数 (s) | (c=\sqrt{1/t}) | (t) 约为 |
|---:|---:|---:|
| 2 | 1.069 | 0.875 |
| 8 | 1.208 | 0.686 |
| 16 | 1.277 | 0.613 |
| 32 | 1.347 | 0.551 |

### 4.4 YaRN 的计算流程

以一个标准 RoPE attention head 为例：

```text
输入隐藏状态 x[m]
        |
        +-- q[m] = Wq x[m]
        +-- k[m] = Wk x[m]
        |
        | 对每个 RoPE 维度 d：
        |   1. 计算波长 lambda[d]
        |   2. 计算旋转圈数 r[d] = L / lambda[d]
        |   3. 计算 gamma(r[d])
        |   4. 得到 theta_new[d]
        |
        +-- 用 m * theta_new[d] 旋转 q[m]
        +-- 用 m * theta_new[d] 旋转 k[m]
        |
        +-- q_rot = c * q_rot
        +-- k_rot = c * k_rot
        |
        +-- attention = softmax(q_rot @ k_rot.T / sqrt(D))
```

固定上下文长度时，(cos)、(sin) 缓存可以预先生成，因此 YaRN 不增加 attention 的渐进复杂度，也不需要额外的网络层。

### 4.5 伪代码

下面是表达算法含义的简化伪代码：

```python
scale = target_length / pretrained_length

for d in rope_dimensions:
    theta = base ** (-2 * d / head_dim)
    wavelength = 2 * pi / theta
    rotations = pretrained_length / wavelength

    if rotations < alpha:
        gamma = 0.0
    elif rotations > beta:
        gamma = 1.0
    else:
        gamma = (rotations - alpha) / (beta - alpha)

    theta_new[d] = (1 - gamma) * theta / scale + gamma * theta

q = rotate(q, position_ids, theta_new)
k = rotate(k, position_ids, theta_new)

attention_scale = 0.1 * log(scale) + 1
q = q * attention_scale
k = k * attention_scale

output = softmax(q @ k.transpose(-1, -2) / sqrt(head_dim)) @ v
```

真实工程实现还要注意：频率索引、偶数/奇数维度的交错方式、cos/sin cache、KV cache 和 Flash Attention 的接口。

---

## 5. Dynamic Scaling

### 5.1 为什么需要动态缩放

自回归生成通常会经历很多次 forward：

```text
第 1 次：序列长度 1
第 2 次：序列长度 2
...
第 n 次：序列长度 n
```

如果从一开始就固定使用目标长度 \(L'\) 的缩放因子，那么短序列也会被强行压缩，可能损伤原模型在短上下文上的表现。

论文比较两种做法：

1. 整个生成过程固定 \(s=L'/L\)；
2. 每次 forward 根据当前长度更新：

$$
s_{current}=\max\left(1,\frac{l'}{L}\right).
$$

其中 (l') 是当前序列长度。

第二种就是 Dynamic Scaling。序列没有超过原始长度时使用 \(s=1\)，超过后才逐渐增加缩放压力，因此性能退化更平滑。

### 5.2 Dynamic Scaling 和 Dynamic NTK

Dynamic Scaling 是推理时的通用策略：每次根据当前长度计算 scale。

如果它和 NTK-aware 一起使用，称为 **Dynamic NTK**；如果它和 YaRN 一起使用，可以称为 **Dynamic-YaRN**。论文附录的实验显示，在没有长上下文微调的原始 Llama 2 上，Dynamic-YaRN 比 Dynamic-PI 的长序列困惑度更好。

### 5.3 Dynamic Scaling 对 KV cache 的注意事项

普通 KV cache 通常缓存已经计算好的 K、V。可是当 \(s\) 每次变化时，所有历史 token 的 RoPE 角度都会变化。

因此 Dynamic Scaling 的正确缓存顺序是：

```text
Wk x -> 先缓存未施加 RoPE 的 K
当前 forward 确定 s_current
对历史 K 按当前 s_current 重新施加 RoPE
```

如果缓存的是已经旋转过的 K，那么后续 scale 变化时不能正确更新历史 token 的位置表示。

---

## 6. 实验设置

### 6.1 模型和训练

论文主要使用 LLaMA/Llama 2 7B 和 13B：

| 模型 | 原始长度 | 目标长度 | scale |
|---|---:|---:|---:|
| LLaMA 7B | 2K | 32K | 16 |
| Llama 2 7B/13B | 4K | 64K | 16 |
| Llama 2 7B/13B | 4K | 128K | 32 |

训练使用 PG19 长文档数据：

- 64K 模型使用 64K token 的连续片段；
- (s=16) 模型微调 400 steps；
- (s=32) 模型从 (s=16) checkpoint 继续训练 200 steps；
- global batch size 为 64；
- learning rate 为 (2\times10^{-5})；
- 使用 AdamW、FSDP 和 Flash Attention 2。

重要的是，YaRN 没有修改 Llama 的 Transformer 架构，主要修改的是 RoPE 频率计算和等价的 Q/K scaling。

### 6.2 评测指标

论文使用四类评测：

1. **长序列语言建模**：在 Proof-pile、GovReport 上计算 sliding-window perplexity；
2. **Passkey retrieval**：把一个五位数字放入长文本，测试模型能否找回；
3. **短上下文通用能力**：ARC-Challenge、HellaSwag、MMLU、TruthfulQA；
4. **训练效率**：比较训练 steps、token 数和 A100 GPU-hours。

sliding window 评测的步长是 (S=256)。它不是只评估最后一个窗口，而是让整个长文档中的 token 都对困惑度有贡献。

---

## 7. 主要实验结果

### 7.1 从 4K 扩展到 64K/128K

下表是论文对十个至少 128K token 的 Proof-pile 文档计算的 sliding-window perplexity。数值越低越好。

#### Llama 2 7B

| 方法 | 训练数据长度 | 评测 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|---:|
| YaRN (s=16) | 64K | 3.51 | 2.99 | 2.65 | 2.42 | >10 |
| YaRN (s=32) | 64K | 3.56 | 3.04 | 2.70 | 2.45 | 2.37 |

#### Llama 2 13B

| 方法 | 训练数据长度 | 评测 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|---:|
| YaRN (s=16) | 64K | 3.25 | 2.79 | 2.50 | 2.29 | >10 |
| YaRN (s=32) | 64K | 3.29 | 2.83 | 2.53 | 2.31 | 2.24 |

最值得注意的是：(s=32) 模型只使用 64K 长度的数据训练，却能在 128K 上取得良好困惑度。这是论文所说的 **train short, test long**，即训练长度和最终可用长度不完全相同。

### 7.2 消融实验：每个组件的作用

在 LLaMA 7B 从 2K 扩展到 32K 的实验中，400 steps 微调结果如下：

| 方法 | 2K | 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|---:|
| PI | 5.70 | 4.95 | 4.64 | 3.97 | 3.57 |
| NTK-aware | 4.39 | 3.92 | 3.73 | 3.21 | 8.49 |
| NTK-by-parts | 4.14 | 3.75 | 3.62 | 3.12 | 2.81 |
| YaRN | 4.19 | 3.77 | **3.30** | **3.09** | **2.77** |

从结果可以看出：

- PI 在较大扩展倍数下退化明显；
- NTK-aware 对非微调模型有帮助，但并不总是适合微调；
- NTK-by-parts 明显改善长上下文表现；
- YaRN 在 NTK-by-parts 基础上加入 attention scaling 后，整体最好。

### 7.3 Passkey retrieval

在 128K 模型上，YaRN 的 passkey retrieval 准确率很高：

- 7B，(s=32)：99.4%；
- 13B，(s=32)：99.4%。

这说明模型不仅在平均语言建模指标上工作，也能在长文本中定位一个远距离的关键信息。

论文还提醒：困惑度不是长上下文能力的全部。某个模型的 perplexity 在 100K 以后上升，并不一定代表它完全无法从 128K 位置找回信息。

### 7.4 短上下文能力是否被破坏

Llama 2 7B 原模型和 YaRN 模型的 Open LLM benchmark：

| 模型 | ARC-C | HellaSwag | MMLU | TruthfulQA |
|---|---:|---:|---:|---:|
| 原始 Llama 2 7B | 53.1 | 77.8 | 43.8 | 39.0 |
| YaRN (s=16) | 52.3 | 78.8 | 42.5 | 38.2 |
| YaRN (s=32) | 52.1 | 78.4 | 41.7 | 37.3 |

总体上短上下文通用能力只有小幅下降。论文将这种差异的一部分归因于微调数据 PG19 和原模型预训练数据不同。

### 7.5 训练效率

论文的比较显示，YaRN 达到相近效果所需的训练量更小：

- YaRN 从 4K 到 64K 的 Llama 2 7B 模型约使用 256 A100 GPU-hours；
- 从 64K 继续扩展到 128K 只增加约 128 A100 GPU-hours；
- 旧的 PI 方法在类似扩展任务中需要更多 training steps；
- 某些 NTK-aware 长上下文模型的公开训练成本明显更高。

论文摘要将总体收益概括为：相比此前方法，YaRN 约少用 10 倍 token 和 2.5 倍训练 steps。

---

## 8. 为什么 YaRN 有效

可以把 YaRN 的有效性归纳成三个互补的修复：

### 修复一：避免角度越界

对低频/长波长维度进行插值，把长序列的位置压回模型熟悉的旋转范围。

### 修复二：保留局部相对距离

高频维度本来就主要用来区分附近 token 的相对顺序。如果也把它们除以很大的 \(s\)，相邻位置之间的旋转差变小，局部结构会丢失。NTK-by-parts 对这些维度保持原始频率。

### 修复三：调整注意力熵

仅修改位置角度会改变 Q/K 的统计分布和 attention logits 的熵。YaRN 使用 Q/K 等比例放大，等价于在 softmax 前降低温度，使注意力分布恢复到更适合模型的锐度。

这三个部分分别处理：

```text
位置范围问题       -> 插值
局部距离问题       -> 高频维度保留
attention 分布问题 -> logit/温度 scaling
```

---

## 9. 和其他长上下文方法的对比

| 方法 | 核心操作 | 主要优点 | 主要问题 |
|---|---|---|---|
| 直接外推 | 保持原始 RoPE | 不改变训练内位置 | 超过训练长度后容易崩溃 |
| PI | 所有位置压缩到原窗口 | 简单、易实现、微调有效 | 高频信息损失较大 |
| NTK-aware | 改变 RoPE base，分散缩放压力 | 非微调表现较好 | base 难选，可能有轻微外推 |
| NTK-by-parts | 按频率分段插值 | 兼顾绝对位置和局部距离 | \(\alpha,\beta\) 需要调参 |
| YaRN | NTK-by-parts + attention scaling | 效果和训练效率都更好 | 仍依赖 RoPE 和长上下文微调 |
| ReRoPE | 修改 attention 机制 | 可以尝试更强的长度泛化 | 论文时不兼容 Flash Attention 2，需要两次 attention |

YaRN 的工程优势是：它主要改 RoPE embedding 和 Q/K 的尺度，不需要重新设计 attention kernel，因此比较容易接入已有 LLaMA 实现和 Flash Attention 2。

---

## 10. 实现时最容易混淆的点

### 10.1 YaRN 不是把 theta 全部变小

错误理解：

```text
theta_new = theta / s
```

这是 PI 风格的统一缩放，不是完整 YaRN。

YaRN 是按每个频率计算：

```text
低频：theta / s
高频：theta
中间：两者之间的平滑混合
```

### 10.2 “低频”不等于“低维编号”

在常见 RoPE 定义中，(d) 越大，(	heta_d) 越小、波长越长，所以较大的维度编号往往是低频维度。但工程中的维度排列可能交错，不能只根据 tensor 的前后位置猜测，应该以具体 RoPE 实现的频率生成方式为准。

### 10.3 temperature scaling 不是除以 (\sqrt D)

attention 中的两种缩放作用不同：

```text
除以 sqrt(D)：控制点积随维度增大而变大的数值尺度
YaRN 的 t：校准扩展 RoPE 后 attention logits 的分布/熵
```

YaRN 的实现等价于同时把 Q、K 乘以 (sqrt{1/t})，不是删除标准的 (1/\sqrt D)。

### 10.4 Dynamic Scaling 和固定 Scaling 不一样

固定 Scaling 在整个生成过程中使用目标上下文的 scale；Dynamic Scaling 每次根据当前序列长度更新 scale。后者需要谨慎设计 KV cache，因为 RoPE 作用后的历史 K 会随 scale 改变。

### 10.5 YaRN 不是“只改配置就无限扩展”

增大 `max_position_embeddings` 或改变 `rope_theta` 只能改变位置编码的计算范围，不能自动让模型学会长距离语言建模。论文仍然进行了长上下文微调，并通过长文档和 passkey 任务验证效果。

---

## 11. 论文贡献与局限

### 贡献

1. 系统总结 PI、NTK-aware、NTK-by-parts 和 Dynamic Scaling 等 RoPE 上下文扩展方法；
2. 识别统一插值造成的高频信息损失；
3. 提出按波长分段处理的 NTK-by-parts；
4. 提出 attention scaling，并与 NTK-by-parts 组合为 YaRN；
5. 在 LLaMA/Llama 2 上展示高效扩展到 32K、64K 和 128K；
6. 展示只用 64K 训练数据也可以外推到 128K；
7. 提供开源实现和复现实验代码。

### 局限

- 方法针对使用 RoPE 的模型，不能直接套用到完全不同的位置编码；
- \(\alpha,\beta\) 以及 scaling 公式主要通过实验得到，不是完全无超参数的方法；
- 长上下文微调仍然需要计算资源；
- PPL、passkey retrieval 和通用 benchmark 只能从不同角度反映能力，不能保证所有长文档任务都有效；
- 论文实验主要基于当时的 LLaMA/Llama 2，换模型家族时需要重新验证频率划分和温度系数。

---

## 12. 阅读论文时的定位

按照论文结构，可以这样读：

| 论文部分 | 重点 |
|---|---|
| Section 1 Introduction | 长上下文问题和本文贡献 |
| Section 2.1 Rotary Position Embeddings | RoPE、频率、相对位置性质 |
| Section 2.2/2.3 Related Work | PI、ReRoPE、LM-Infinite 等 |
| Section 3.1 NTK-aware | 高频信息为什么会丢失 |
| Section 3.2 NTK-by-parts | 按波长分段插值的核心公式 |
| Section 3.3 YaRN | attention scaling 和完整 YaRN |
| Section 3.4 Dynamic Scaling | 推理时动态更新缩放因子、KV cache |
| Section 4 Experiments | 训练设置、长序列、passkey 和基准测试 |
| Appendix A | PI、NTK-aware 的补充公式 |
| Appendix B | temperature scaling 的困惑度分析 |
| Appendix C/D | 消融、效率和 Dynamic Scaling 结果 |

最重要的公式顺序是：

```text
RoPE:             theta_d, lambda_d
PI:               m -> m / s
NTK-aware:        b -> b'
NTK-by-parts:     r(d), gamma(r), theta'_d
YaRN:             theta'_d + q/k scaling
Dynamic Scaling:  s -> max(1, current_length / L)
```

## 最后总结

YaRN 的本质不是“把 RoPE 的 base 调大”，而是根据不同频率承载的信息类型进行有选择的修改：长波长维度负责把位置范围压回训练分布，短波长维度负责保留局部相对距离，最后再通过 attention scaling 修正注意力分布。这样既减少了长上下文外推时的角度越界，又避免了统一插值对局部位置信息的破坏。
