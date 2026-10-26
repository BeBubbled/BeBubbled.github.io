# Papers of Interest

按研究主题整理的阅读清单：附件中的 61 篇论文与原有补充合并去重，共 64 篇；更早的阅读条目保留在页末。

**阅读优先级：** A = 优先阅读；B = 重要基线或机制参考；— = 原有补充，尚未排序。优先级沿用阅读笔记，表示个人研究相关性，不代表论文质量或已读状态。年份和会议沿用题录；仅列年份的条目可能对应预印本。

## 1. 权重排列、排序与模型合并

| 论文 | 年份 / 会议 | 阅读重点 | 优先级 |
| --- | --- | --- | :---: |
| [Train-by-Reconnect: Decoupling Locations of Weights from Their Values (LaPerm)](https://arxiv.org/abs/2003.02570) | NeurIPS 2020 | 固定权重数值、学习连接排列；区分排列学习与数值优化 | A |
| [Neural Networks Trained by Weight Permutation are Universal Approximators](https://arxiv.org/abs/2407.01033) | 2025 | 置换训练的通用逼近能力及适用条件 | B |
| [Transfusion](https://arxiv.org/abs/1902.07208) | 2019 | 权重统计信息与训练收敛 | B |
| [Monotonic Differentiable Sorting Networks](https://arxiv.org/abs/2203.09630) | 2022 | 可微排序机制 | B |
| [PLeaS: Merging Models with Permutations and Least Squares](https://openaccess.thecvf.com/content/CVPR2025/html/Nasery_PLeaS_-_Merging_Models_with_Permutations_and_Least_Squares_CVPR_2025_paper.html) | CVPR 2025 | 置换匹配与最小二乘数值优化 | A |

LaPerm 对应最初的“先排序、后数值”线索，但不应将其概括为普遍成立的两阶段收敛定理。

## 2. 模型量化与压缩（PTQ）

| 论文 | 年份 / 会议 | 阅读重点 | 优先级 |
| --- | --- | --- | :---: |
| [Deep Compression](https://arxiv.org/abs/1510.00149) | 2016 | 剪枝、共享码本、权重聚类与编码 | B |
| [AdaRound](https://proceedings.mlr.press/v119/nagel20a.html) | 2020 | 数据驱动的量化舍入与局部重建 | A |
| [BRECQ](https://arxiv.org/abs/2102.05426) | 2021 | 块级依赖与量化重建 | A |
| [PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression](https://arxiv.org/abs/2405.14852) | NeurIPS 2024 · Oral | 离散分配与连续参数优化；极低比特压缩 | A |
| [Permute, Quantize, and Fine-tune](https://arxiv.org/abs/2010.15703) | 2021 | 通道置换与量化结合 | A |
| [VQRound](https://arxiv.org/abs/2602.02151) | 2026 | 向量化重参数化与舍入优化 | B |
| [COMQ](https://arxiv.org/abs/2403.07134) | 2024 | 无反向传播的坐标更新量化 | A |
| [JARQ](https://arxiv.org/abs/2609.38599) | 2026 | 量化参数与整数编码联合优化；题录链接待复核 | A |

PV-Tuning：[正式论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/091166620a04a289c555f411d8899049-Abstract-Conference.html) · [代码](https://github.com/Vahe1994/AQLM)。

## 3. Massive Activations 与扩散模型表征

| 论文 | 年份 / 会议 | 阅读重点 | 优先级 |
| --- | --- | --- | :---: |
| [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762) | COLM 2024 | 巨激活、隐式偏置与注意力集中 | A |
| [DiTF](https://arxiv.org/abs/2505.18584) | 2025 | 巨激活调制与视觉对应 | A |
| [Massive Activations are the Key to Local Detail Synthesis in Diffusion Transformers](https://arxiv.org/abs/2510.11538) | 2025 | 巨激活与局部细节生成 | A |
| [Few Channels Draw The Whole Picture: Revealing Massive Activations in Diffusion Transformers](https://arxiv.org/abs/2605.13974) | 2026 | 少量通道中的空间语义及语义迁移 | A |
| [Awakening Diffusion Transformers](https://arxiv.org/abs/2607.02968) | 2026 | 巨激活调制、生成与视觉理解 | A |
| [From Image to Video: An Empirical Study of Diffusion Representations](https://arxiv.org/abs/2502.07001) | ICCV 2025 | 扩散表征用于图像与视频理解；DeepMind 相关论文的候选匹配 | — |

Massive Activations：[项目](https://eric-mingjie.github.io/massive-activations/index.html) · [代码](https://github.com/locuslab/massive-activations)。From Image to Video：[正式论文](https://openaccess.thecvf.com/content/ICCV2025/html/Velez_From_Image_to_Video_An_Empirical_Study_of_Diffusion_Representations_ICCV_2025_paper.html)。

## 4. 网络扩宽、加深与模型增长

| 论文 | 年份 / 会议 | 阅读重点 | 优先级 |
| --- | --- | --- | :---: |
| [Net2Net: Accelerating Learning via Knowledge Transfer](https://arxiv.org/abs/1511.05641) | 2016（预印本 2015） | 保持函数的扩宽与恒等层插入 | B |
| [LiGO](https://arxiv.org/abs/2303.00980) | 2023 | 用可学习映射扩展预训练模型 | B |
| [Escaping the Subspace Trap](https://proceedings.mlr.press/v306/chen26eq.html) | 2026 | 扩宽后的优化几何与新增参数利用 | A |
| [LLaMA Pro: Progressive LLaMA with Block Expansion](https://aclanthology.org/2024.acl-long.352/) | ACL 2024 | 插入 Transformer 块，冻结原块并训练新增块 | — |

“从 layer 中间扩展 layer”的原始线索尚未唯一确定：LLaMA Pro 是块插入方向的候选；Net2Net 是保函数扩展方向的候选。[LLaMA Pro 代码](https://github.com/TencentARC/LLaMA-Pro)。

## 5. Attention、MLP 与结构重参数化

| 论文 | 年份 / 会议 | 阅读重点 | 优先级 |
| --- | --- | --- | :---: |
| [HyperMLP: An Integrated Perspective for Sequence Modeling](https://proceedings.mlr.press/v306/lu26aq.html) | ICML 2026 | 将自回归注意力解释为上下文动态实例化的两层 MLP | A |
| [Mixture-of-Top-k Attention: Efficient Attention via Scalable Fast Weights (MiTA)](https://arxiv.org/abs/2602.01219) | 2026 | 快权重 MLP 视角、路由与高效注意力 | — |
| [Attention-Only Transformers and Implementing MLPs with Attention Heads](https://arxiv.org/abs/2309.08593) | 2023 | 用 Attention Head 实现 MLP 神经元的条件 | B |
| [RepVGG](https://arxiv.org/abs/2101.03697) | CVPR 2021 | 训练多分支、推理单分支的等价参数融合 | A |
| [RepMLPNet](https://arxiv.org/abs/2112.11081) | CVPR 2022 | 将局部卷积结构融合进 MLP 参数 | A |
| [RePaViT](https://proceedings.mlr.press/v267/xu25o.html) | ICML 2025 | 可重参数化线性通路与 ViT FFN 加速 | B |
| [ReplaceMe](https://proceedings.nips.cc/paper_files/paper/2025/hash/1c10d0c087c14689628124bbc8fa69f6-Abstract-Conference.html) | NeurIPS 2025 | 校准拟合线性映射以替代 Transformer Block | A |
| [Neural Block Linearization](https://papers.neurips.cc/paper_files/paper/2025/hash/088463cd3126aef2002ffc69da42ec59-Abstract-Conference.html) | NeurIPS 2025 | 预训练 Attention 层的线性近似 | A |
| [RepSpec](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d70ea003729b440b89a2f958a5554c1f-Abstract-Conference.html) | ICLR 2026 | 训练冗余结构与推理融合 | B |

“2026 年 2 月将 Attention 变为 MLP”的线索保留 HyperMLP 与 MiTA 两个候选。HyperMLP 的[预印本](https://arxiv.org/abs/2602.12601)发表于 2 月；这里的 MLP 权重依赖上下文，并非普通固定权重 MLP。

## 6. 特征压缩与面向机器视觉的图像编码

| 论文 | 年份 / 会议 | 阅读重点 | 优先级 |
| --- | --- | --- | :---: |
| [Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396) | 2019 | 面向内积检索目标的向量量化 | B |
| [EQUSS](https://arxiv.org/abs/2312.07342) | AAAI 2024 | 产品量化与无监督语义分割 | A |
| [Deep Image Compression Toward Machine Vision: A Unified Optimization Framework](https://ieeexplore.ieee.org/document/9992257/) | 2023 | 面向机器视觉任务的率—精度联合优化 | A |
| [FSIC: Frequency-separated Image Compression for Small Object Detection](https://www.sciencedirect.com/science/article/pii/S1051200424004470) | 2025 | 小目标检测与频率感知图像压缩 | A |
| [Task-Oriented Lossy Compression with Data, Perception, and Classification Constraints](https://ieeexplore.ieee.org/document/10960410/) | 2025（预印本 2024） | 码率、感知质量与分类性能约束 | B |

## 7. 视觉特征、几何估计与点跟踪

| 论文 | 年份 / 会议 | 阅读重点 | 优先级 |
| --- | --- | --- | :---: |
| [INSID3](https://arxiv.org/abs/2603.28480) | 2026 | 冻结视觉特征与细粒度理解 | B |
| [Rolling Shutter Relative Pose Estimation Made Practical](https://arxiv.org/abs/2606.26863) | 2026 | 高效滚动快门相对位姿求解 | A |
| [VSAC](https://arxiv.org/abs/2106.10240) | 2022 | 鲁棒几何估计与 Independent Inliers | A |
| [Track-On-R](https://arxiv.org/abs/2603.12217) | CVPR 2026 | 轨迹验证与可靠结果选择 | A |
| [TAPNext++](https://arxiv.org/abs/2604.10582) | 2026 | 遮挡后的点重新检测及评测 | A |
| [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) | 2023 | 显式、可渲染的三维场景表示 | B |
| [2D Gaussian Splatting](https://arxiv.org/abs/2403.17888) | 2024 | 面向表面几何的高斯表示 | B |
| [VGGT](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html) | CVPR 2025 | 前馈几何理解与多任务预测 | A |

## 8. 持续学习与增量学习

| 论文 | 年份 / 会议 | 阅读重点 | 优先级 |
| --- | --- | --- | :---: |
| [Learning to Prompt (L2P)](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.html) | CVPR 2022 | 基于 Prompt 的持续学习 | B |
| [CODA-Prompt](https://openaccess.thecvf.com/content/CVPR2023/html/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.html) | CVPR 2023 | 条件化 Prompt 与无回放持续学习 | B |
| [InfLoRA](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html) | CVPR 2024 | 低秩子空间与任务干扰控制 | A |
| [EASE](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Expandable_Subspace_Ensemble_for_Pre-Trained_Model-Based_Class-Incremental_Learning_CVPR_2024_paper.html) | CVPR 2024 | 扩展特征子空间与类别增量学习 | A |
| [SEMA](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Self-Expansion_of_Pre-trained_Models_with_Mixture_of_Adapters_for_Continual_CVPR_2025_paper.html) | CVPR 2025 | 自适应 Adapter 扩展 | B |
| [BiMC](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Enhancing_Few-Shot_Class-Incremental_Learning_via_Training-Free_Bi-Level_Modality_Calibration_CVPR_2025_paper.html) | CVPR 2025 | 免训练少样本类别增量校准 | B |
| [Computationally Budgeted Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/html/Prabhu_Computationally_Budgeted_Continual_Learning_What_Does_Matter_CVPR_2023_paper.html) | CVPR 2023 | 真实计算预算下的持续学习方法比较 | A |
| [Real-Time Evaluation in Online Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/html/Ghunaim_Real-Time_Evaluation_in_Online_Continual_Learning_A_New_Hope_CVPR_2023_paper.html) | CVPR 2023 | 在线计算约束与实时评价 | A |
| [Re-evaluating Continual VQA](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Re-evaluating_Continual_VQA_Toward_Fair_and_Robust_Evaluation_for_Multimodal_CVPR_2026_paper.html) | CVPR 2026 | 数据集偏差与多模态遗忘评价 | A |

## 9. 视频深度、时序一致性与视频编辑

| 论文 | 年份 / 会议 | 阅读重点 | 优先级 |
| --- | --- | --- | :---: |
| [RollingDepth: Video Depth without Video Models](https://arxiv.org/abs/2411.19189) | CVPR 2025 | 片段级估计、对齐与视频深度一致性 | B |
| [Video Depth Anything](https://arxiv.org/abs/2501.12375) | CVPR 2025 | 高效、长时序一致的视频深度估计 | A |
| [Video Depth Propagation (VeloDepth)](https://arxiv.org/abs/2512.10725) | 2025 | 光流传播与深度残差修正 | A |
| [DynamiCrafter](https://arxiv.org/abs/2310.12190) | ECCV 2024 | 图像到视频生成、运动与内容保持 | B |
| [AnyV2V](https://arxiv.org/abs/2403.14468) | 2024 | 免训练视频编辑与时序一致性 | B |

## 10. 扩散模型剪枝与自适应推理

| 论文 | 年份 / 会议 | 阅读重点 | 优先级 |
| --- | --- | --- | :---: |
| [TinyFusion](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_TinyFusion_Diffusion_Transformers_Learned_Shallow_CVPR_2025_paper.html) | CVPR 2025 | DiT 结构剪枝与性能恢复 | B |
| [Adaptive Time-Stepping Schedules for Diffusion Models](https://proceedings.mlr.press/v244/chen24c.html) | UAI 2024 | 采样误差与时间步调度 | B |
| [AdaDiff](https://ojs.aaai.org/index.php/AAAI/article/view/33075) | AAAI 2025 | 按输入自适应选择采样步数 | B |
| [Not All Prompts Are Made Equal](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4f1c0a593705124f3061ca7d48b92898-Abstract-Conference.html) | ICLR 2025 | Prompt 条件化的适应性剪枝 | A |
| [ALTER](https://papers.nips.cc/paper_files/paper/2025/hash/baca5eb9e789dbfc45446715bc9f692e-Abstract-Conference.html) | NeurIPS 2025 | 层剪枝、时间步路由与参数适配 | A |

## Earlier reading list

以下保留此前的阅读条目；未提供链接的标题暂按原始记录保留。

### 复数网络与量子理论

- Ruling out real-valued standard formalism of quantum theory
- A Survey of **Complex-Valued** Neural Networks
- [DEEP COMPLEX NETWORKS](https://arxiv.org/abs/1705.09792)

### 科学计算与偏微分方程

- DGM: A deep learning algorithm for solving **partial differential**

### 持续学习、增量学习与剪枝

- Continual learning in neural networks
- Achieving Forgetting Prevention and Knowledge Transfer in **Continual Learning**
- **Few-Shot** Incremental Learning with Continually Evolved Classifiers
- PackNet Adding Multiple Tasks to a Single Network by Iterative **Pruning**

### 联邦学习与安全

- [Local model poisoning attacks to {Byzantine-Robust} federated learning](https://www.usenix.org/conference/usenixsecurity20/presentation/fang)
