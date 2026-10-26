# Papers of Interest

A reading list of 64 papers organized by research topic, followed by earlier reading selections.

Years refer to the listed publication or preprint. Venue information is included where available.

## 1. Weight Permutations, Sorting, and Model Merging

| Paper | Year / Venue |
| --- | --- |
| [Train-by-Reconnect: Decoupling Locations of Weights from Their Values (LaPerm)](https://arxiv.org/abs/2003.02570) | NeurIPS 2020 |
| [Neural Networks Trained by Weight Permutation are Universal Approximators](https://arxiv.org/abs/2407.01033) | 2025 |
| [Transfusion](https://arxiv.org/abs/1902.07208) | 2019 |
| [Monotonic Differentiable Sorting Networks](https://arxiv.org/abs/2203.09630) | 2022 |
| [PLeaS: Merging Models with Permutations and Least Squares](https://openaccess.thecvf.com/content/CVPR2025/html/Nasery_PLeaS_-_Merging_Models_with_Permutations_and_Least_Squares_CVPR_2025_paper.html) | CVPR 2025 |

LaPerm is relevant to learning weight permutations before numerical refinement; it does not establish a universal two-stage convergence theorem.

## 2. Model Quantization and Compression (PTQ)

| Paper | Year / Venue |
| --- | --- |
| [Deep Compression](https://arxiv.org/abs/1510.00149) | 2016 |
| [AdaRound](https://proceedings.mlr.press/v119/nagel20a.html) | 2020 |
| [BRECQ](https://arxiv.org/abs/2102.05426) | 2021 |
| [PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression](https://arxiv.org/abs/2405.14852) | NeurIPS 2024 · Oral |
| [Permute, Quantize, and Fine-tune](https://arxiv.org/abs/2010.15703) | 2021 |
| [VQRound](https://arxiv.org/abs/2602.02151) | 2026 |
| [COMQ](https://arxiv.org/abs/2403.07134) | 2024 |
| [JARQ](https://arxiv.org/abs/2609.38599) | 2026 |

PV-Tuning: [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/091166620a04a289c555f411d8899049-Abstract-Conference.html) · [Code](https://github.com/Vahe1994/AQLM).

## 3. Massive Activations and Diffusion Representations

| Paper | Year / Venue |
| --- | --- |
| [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762) | COLM 2024 |
| [DiTF](https://arxiv.org/abs/2505.18584) | 2025 |
| [Massive Activations are the Key to Local Detail Synthesis in Diffusion Transformers](https://arxiv.org/abs/2510.11538) | 2025 |
| [Few Channels Draw The Whole Picture: Revealing Massive Activations in Diffusion Transformers](https://arxiv.org/abs/2605.13974) | 2026 |
| [Awakening Diffusion Transformers](https://arxiv.org/abs/2607.02968) | 2026 |
| [From Image to Video: An Empirical Study of Diffusion Representations](https://arxiv.org/abs/2502.07001) | ICCV 2025 |

Massive Activations: [Project](https://eric-mingjie.github.io/massive-activations/index.html) · [Code](https://github.com/locuslab/massive-activations). From Image to Video: [Proceedings](https://openaccess.thecvf.com/content/ICCV2025/html/Velez_From_Image_to_Video_An_Empirical_Study_of_Diffusion_Representations_ICCV_2025_paper.html).

## 4. Network Widening, Deepening, and Model Growth

| Paper | Year / Venue |
| --- | --- |
| [Net2Net: Accelerating Learning via Knowledge Transfer](https://arxiv.org/abs/1511.05641) | 2016 (preprint 2015) |
| [LiGO](https://arxiv.org/abs/2303.00980) | 2023 |
| [Escaping the Subspace Trap](https://proceedings.mlr.press/v306/chen26eq.html) | 2026 |
| [LLaMA Pro: Progressive LLaMA with Block Expansion](https://aclanthology.org/2024.acl-long.352/) | ACL 2024 |

LLaMA Pro and Net2Net remain candidate matches for the original layer-insertion reference. [LLaMA Pro code](https://github.com/TencentARC/LLaMA-Pro).

## 5. Attention, MLPs, and Structural Reparameterization

| Paper | Year / Venue |
| --- | --- |
| [HyperMLP: An Integrated Perspective for Sequence Modeling](https://proceedings.mlr.press/v306/lu26aq.html) | ICML 2026 |
| [Mixture-of-Top-k Attention: Efficient Attention via Scalable Fast Weights (MiTA)](https://arxiv.org/abs/2602.01219) | 2026 |
| [Attention-Only Transformers and Implementing MLPs with Attention Heads](https://arxiv.org/abs/2309.08593) | 2023 |
| [RepVGG](https://arxiv.org/abs/2101.03697) | CVPR 2021 |
| [RepMLPNet](https://arxiv.org/abs/2112.11081) | CVPR 2022 |
| [RePaViT](https://proceedings.mlr.press/v267/xu25o.html) | ICML 2025 |
| [ReplaceMe](https://proceedings.nips.cc/paper_files/paper/2025/hash/1c10d0c087c14689628124bbc8fa69f6-Abstract-Conference.html) | NeurIPS 2025 |
| [Neural Block Linearization](https://papers.neurips.cc/paper_files/paper/2025/hash/088463cd3126aef2002ffc69da42ec59-Abstract-Conference.html) | NeurIPS 2025 |
| [RepSpec](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d70ea003729b440b89a2f958a5554c1f-Abstract-Conference.html) | ICLR 2026 |

HyperMLP and MiTA remain candidate matches for the February 2026 attention-as-MLP reference. The HyperMLP [preprint](https://arxiv.org/abs/2602.12601) appeared in February; its MLP weights depend on the context.

## 6. Feature Compression and Image Coding for Machine Vision

| Paper | Year / Venue |
| --- | --- |
| [Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396) | 2019 |
| [EQUSS](https://arxiv.org/abs/2312.07342) | AAAI 2024 |
| [Deep Image Compression Toward Machine Vision: A Unified Optimization Framework](https://ieeexplore.ieee.org/document/9992257/) | 2023 |
| [FSIC: Frequency-separated Image Compression for Small Object Detection](https://www.sciencedirect.com/science/article/pii/S1051200424004470) | 2025 |
| [Task-Oriented Lossy Compression with Data, Perception, and Classification Constraints](https://ieeexplore.ieee.org/document/10960410/) | 2025 (preprint 2024) |

## 7. Visual Features, Geometry Estimation, and Point Tracking

| Paper | Year / Venue |
| --- | --- |
| [INSID3](https://arxiv.org/abs/2603.28480) | 2026 |
| [Rolling Shutter Relative Pose Estimation Made Practical](https://arxiv.org/abs/2606.26863) | 2026 |
| [VSAC](https://arxiv.org/abs/2106.10240) | 2022 |
| [Track-On-R](https://arxiv.org/abs/2603.12217) | CVPR 2026 |
| [TAPNext++](https://arxiv.org/abs/2604.10582) | 2026 |
| [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) | 2023 |
| [2D Gaussian Splatting](https://arxiv.org/abs/2403.17888) | 2024 |
| [VGGT](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html) | CVPR 2025 |

## 8. Continual and Incremental Learning

| Paper | Year / Venue |
| --- | --- |
| [Learning to Prompt (L2P)](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.html) | CVPR 2022 |
| [CODA-Prompt](https://openaccess.thecvf.com/content/CVPR2023/html/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.html) | CVPR 2023 |
| [InfLoRA](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html) | CVPR 2024 |
| [EASE](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Expandable_Subspace_Ensemble_for_Pre-Trained_Model-Based_Class-Incremental_Learning_CVPR_2024_paper.html) | CVPR 2024 |
| [SEMA](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Self-Expansion_of_Pre-trained_Models_with_Mixture_of_Adapters_for_Continual_CVPR_2025_paper.html) | CVPR 2025 |
| [BiMC](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Enhancing_Few-Shot_Class-Incremental_Learning_via_Training-Free_Bi-Level_Modality_Calibration_CVPR_2025_paper.html) | CVPR 2025 |
| [Computationally Budgeted Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/html/Prabhu_Computationally_Budgeted_Continual_Learning_What_Does_Matter_CVPR_2023_paper.html) | CVPR 2023 |
| [Real-Time Evaluation in Online Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/html/Ghunaim_Real-Time_Evaluation_in_Online_Continual_Learning_A_New_Hope_CVPR_2023_paper.html) | CVPR 2023 |
| [Re-evaluating Continual VQA](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Re-evaluating_Continual_VQA_Toward_Fair_and_Robust_Evaluation_for_Multimodal_CVPR_2026_paper.html) | CVPR 2026 |

## 9. Video Depth, Temporal Consistency, and Video Editing

| Paper | Year / Venue |
| --- | --- |
| [RollingDepth: Video Depth without Video Models](https://arxiv.org/abs/2411.19189) | CVPR 2025 |
| [Video Depth Anything](https://arxiv.org/abs/2501.12375) | CVPR 2025 |
| [Video Depth Propagation (VeloDepth)](https://arxiv.org/abs/2512.10725) | 2025 |
| [DynamiCrafter](https://arxiv.org/abs/2310.12190) | ECCV 2024 |
| [AnyV2V](https://arxiv.org/abs/2403.14468) | 2024 |

## 10. Diffusion Pruning and Adaptive Inference

| Paper | Year / Venue |
| --- | --- |
| [TinyFusion](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_TinyFusion_Diffusion_Transformers_Learned_Shallow_CVPR_2025_paper.html) | CVPR 2025 |
| [Adaptive Time-Stepping Schedules for Diffusion Models](https://proceedings.mlr.press/v244/chen24c.html) | UAI 2024 |
| [AdaDiff](https://ojs.aaai.org/index.php/AAAI/article/view/33075) | AAAI 2025 |
| [Not All Prompts Are Made Equal](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4f1c0a593705124f3061ca7d48b92898-Abstract-Conference.html) | ICLR 2025 |
| [ALTER](https://papers.nips.cc/paper_files/paper/2025/hash/baca5eb9e789dbfc45446715bc9f692e-Abstract-Conference.html) | NeurIPS 2025 |

## Earlier reading list

Earlier reading selections are retained below. Titles without links follow the original notes.

### Complex-Valued Networks and Quantum Theory

- Ruling out real-valued standard formalism of quantum theory
- A Survey of **Complex-Valued** Neural Networks
- [DEEP COMPLEX NETWORKS](https://arxiv.org/abs/1705.09792)

### Scientific Computing and Partial Differential Equations

- DGM: A deep learning algorithm for solving **partial differential**

### Continual Learning, Incremental Learning, and Pruning

- Continual learning in neural networks
- Achieving Forgetting Prevention and Knowledge Transfer in **Continual Learning**
- **Few-Shot** Incremental Learning with Continually Evolved Classifiers
- PackNet Adding Multiple Tasks to a Single Network by Iterative **Pruning**

### Federated Learning and Security

- [Local model poisoning attacks to {Byzantine-Robust} federated learning](https://www.usenix.org/conference/usenixsecurity20/presentation/fang)
