# Week Two Notes

## 1. Key points of lecs and paper reading

### (1) Pre-LayerNorm

阅读 Lecture 3 推荐的文章：
[On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745)

#### Post-LN 与 Pre-LN 的结构

Post-LN 把 LayerNorm 放在残差相加之后：

$$
x_{l+1}
=
\operatorname{LN}\left(x_l+F_l(x_l)\right).
$$

梯度反向传播经过许多 Post-LN block 时，会反复乘上
LayerNorm 和子层的 Jacobian

对于post-LN，如果训练一开始就采用目标学习率，参数更新可能过大，因此通常需要 learning-rate
warm-up，让学习率从较小值逐渐上升。

Pre-LN 把 LayerNorm 移到子层 \(F_l\) 的内部、残差相加之前：

$$
x_{l+1}
=
x_l+F_l\left(\operatorname{LN}(x_l)\right).
$$
这样就可以在反向传播时将其中一个乘数变为加上一个 I 单位矩阵，不会导致数值过大，也就可以省去lr warmup的步骤，并且加快学习效率
