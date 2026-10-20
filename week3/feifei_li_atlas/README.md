# Fei-Fei Li / World Labs Atlas

本目录整理 Fei-Fei Li 参与的 World Labs 在 2026 年 9 月发布的 Atlas，以及名称相近的 Gaussian Atlas 论文。主报告见 [atlas_technical_report.md](./atlas_technical_report.md)。

## 资料清单

### 核心资料

1. [Atlas: A World Model for Spatial Intelligence](https://www.worldlabs.ai/blog/atlas)（World Labs 官方技术介绍，2026-09-01）
   - 当前最新、最直接的 Atlas 资料。
   - 内容包括模型定位、统一架构、空间重建、时空仿真、机器人 Real-to-Sim、评测与扩展性。
   - 截至本次整理，页面说明 Atlas 处于 select partners early access，未公开权重、代码和完整实验表。

### 背景资料

2. [Marble: A Multimodal World Model](https://www.worldlabs.ai/blog/marble-world-model)（World Labs 官方技术介绍，2025-11-12）
   - Atlas 的产品和应用背景；Marble 支持从文本、图像、视频和粗略 3D 布局生成、编辑和导出 3D 世界。

3. [A Functional Taxonomy of World Models](https://www.worldlabs.ai/blog/taxonomy-of-world-models)（World Labs，2026）
   - 讨论 world model、空间/时间结构、生成式模拟器和物理一致性问题。

### 同名但不同工作的论文

4. [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](https://arxiv.org/abs/2503.15877)（arXiv:2503.15877，2025-03-20）
   - 作者包括 Li Fei-Fei；提出将 3D Gaussian 表示展平为稠密 2D 网格，以复用预训练 2D diffusion model。
   - 这是 3D 物体生成论文，不是 2026 年 World Labs 发布的 Atlas 世界模型。

### Atlas 官方评测中提到的对照论文

5. [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651)；[VGGT-Ω](https://arxiv.org/abs/2605.15195)
   - 代表面向视觉几何重建的 feed-forward 模型，Atlas 官方将其列为 3D reconstruction 对照方法。

6. [π³: Permutation-Equivariant Visual Geometry Learning](https://arxiv.org/abs/2507.13347) 及其 [Pi3/Pi3X 实现](https://github.com/yyfz/Pi3)
   - 研究不依赖固定参考视图的多视图几何重建；Pi3X 被 Atlas 官方评测页列为对照模型。

7. [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647)
   - 从任意数量视图预测空间一致的几何，是 Atlas 官方列出的另一类专业重建基线。

8. [MapAnything: Universal Feed-Forward Metric 3D Reconstruction](https://arxiv.org/abs/2509.13414)
   - 将图像、相机、深度等几何输入统一到 metric 3D reconstruction 中，也是 Atlas 官方列出的比较对象。

## 阅读建议

先读主报告，再按“空间上下文 - 多模态自回归扩散 - 显式 3D 输出 - Real-to-Sim”的顺序阅读 Atlas 官方页面。若需要实现层面的复现，应等待官方发布论文、代码或 API 文档；目前不宜根据宣传页面臆测参数规模、训练数据组成或损失函数细节。
