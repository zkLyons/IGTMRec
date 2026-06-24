<p align="center">
  <a href="./README.md"><img alt="README in English" src="https://img.shields.io/badge/English-DFE0E5"></a>
  <a href="./README_zh.md"><img alt="简体中文版自述文件" src="https://img.shields.io/badge/简体中文-DBEDFA"></a>
</p>

 [![Visitors](https://visitor-badge.laobi.icu/badge?page_id=zkLyons.IGTMRec)](https://github.com/zkLyons/IGTMRec)

# Breaking sequential bias: Importance-guided temporal multimodal recommendation via state space models

## 项目结构

```
IGTM-Rec/
├── datasets                 #数据集文件夹
├── main.py                  # 训练与评估入口脚本
├── IGTMRec.py               # 模型核心实现
├── custom_utils.py          # 自定义数据集 & DataLoader
├── custom_trainer.py        # 自定义训练器（混合精度训练、评估逻辑、TensorBoard 日志）
├── config.yaml              # 模型与训练超参数配置
├── environment.yaml         # Conda 环境配置
├── preprocess/
│   ├── readme.md            # 数据下载说明
│   └── preprocess/
│       ├── data_preprocess.ipynb  # 数据预处理（过滤、多模态特征提取）
│       ├── bert-base-uncased/     # BERT 文本特征提取模型
│       ├── clip-vit-base-patch32/ # CLIP 图像特征提取模型
│       └── coreml/                # 辅助资源
├── assets/
│   ├── model_architecture.png   # 模型架构图
│   └── results.png              # 实验结果对比图
└── README.md
```

---

## 模型介绍

### 整体架构

这是我们团队发表在ESWA上的论文：[Breaking sequential bias: Importance-guided temporal multimodal recommendation via state space models](https://www.sciencedirect.com/science/article/pii/S0957417426021639)

![模型架构图](./assets/model_architecture.png)

## 论文简介

**IGTM-Rec** 是一种新颖的多模态序列推荐框架，旨在解决现有序列推荐方法中存在的两个关键挑战：（1）多模态融合缺乏时间动态性，（2）模型对固定序列顺序的过拟合问题。IGTM-Rec 由两个核心组件构成：

**TMSR（Time-aware Multi-Scale Routing）**

TMSR 将特征处理解耦为两个层次：

- **底层**：通过早期融合构建统一的多模态表示，并利用多分支 GTU（门控时序单元）在不同时间尺度下提取模态特征
- **顶层**：以物理时间间隔作为显式先验信号，动态调控不同跨度的门控时序感受野，使模型能够根据实时时间间隔自适应地选择历史信息

与将时间仅作为辅助特征或偏置信号的现有方法不同，TMSR 强调时间作为**路由信号**的角色，实现了从"静态融合"到"时间驱动路由"的范式转变，更精确地捕捉用户兴趣在不同时间粒度上的演化。

**IGTP（Importance-Guided Progressive Perturbation）**

IGTP 是一种针对序列过拟合问题的训练策略，通过评估交互行为的重要性，在训练过程中**渐进式地扰动**不重要的序列位置。与随机 Mask、随机 Shuffle 等无差别增强策略不同，IGTP 在保持核心语义结构的前提下，逐步降低模型对固定序列顺序的依赖，从而增强所学用户兴趣表示的鲁棒性和泛化能力。

## 实验结果

![实验结果](./assets/results.png)

模型在四个 Amazon 公开数据集（Beauty、Games、Toys、Musical Instruments）上与多种基线模型进行对比，评估指标包括 **NDCG@10/20**、**MRR@10/20** 和 **Hit@10/20**。实验结果表明 IGTM-Rec 在所有数据集上均取得了具有竞争力的性能，平均性能提升达 **6.51%**。

---

## 环境配置

通过 Conda 安装：

```bash
git clone https://github.com/zkLyons/IGTM-Rec.git
cd IGTM-Rec
conda env create -f environment.yaml
conda activate your_conda_env
```

所有实验均在 NVIDIA 24GB 3090 GPU 上进行。主要依赖包如下：

- Python 3.10
- PyTorch 2.1.1 + CUDA 11.8
- mamba-ssm 2.2.2
- RecBole 1.2.0
- causal-conv1d 1.4.0

## 数据集获取

### 1.ID数据处理

我们使用了四个 Amazon 公开数据集（Beauty、Games、Toys、Musical Instruments）。设置步骤如下：

1. 创建 `dataset` 文件夹
2. 下载数据集：[Google Drive](https://drive.google.com/drive/folders/1jsF4n1dge4KgdfD8HKxNjOcyHUKdEHka)
3. 将数据文件放入 `dataset/` 目录下

数据集格式遵循 RecBole 的 Atomic File 格式（`.inter` 文件），包含 `user_id`、`item_id`、`timestamp` 三个主要字段。

### 2.文本和图像特征提取

**特征提取模型获取**

使用 `preprocess/preprocess/data_preprocess.ipynb` 进行多模态特征提取：

- **文本特征**：使用 `bert-base-uncased` 模型提取商品描述等文本的语义特征.
- **图像特征**：使用 `clip-vit-base-patch32` 模型提取商品图像的视觉特征.

运行data_preprocess即可自动下载特征提取模型，如果下载有问题，也可以选择手动下载，链接如下：[huggingface:bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)，[huggingface:clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)

提取后的特征分别保存为 `text_feat.pt` 和 `img_feat.pt`，放入对应/dataset目录下的对应数据集下即可。

**文本和图像数据集获取**

可到amazon下载数据所需数据:https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

所需要的数据为两种：5-core和metadata

下载后按照不同数据集整理好放到precess目录下即可。

---

## 快速启动

训练与评估：

```bash
python main.py
```

### 配置说明

所有超参数在 `config.yaml` 中配置，主要包括：

| 参数                   | 说明             | 默认值 |
| ---------------------- | ---------------- | ------ |
| `hidden_size`          | 特征维度         | 256    |
| `d_state`              | SSM 状态扩展维度 | 64     |
| `d_conv`               | 局部卷积宽度     | 4      |
| `expand`               | 块扩展因子       | 2      |
| `headdim`              | 注意力头维度     | 16     |
| `num_layers`           | IGTMRec 层数     | 3      |
| `dropout_prob`         | Dropout 概率     | 0.4    |
| `learning_rate`        | 学习率           | 0.0001 |
| `train_batch_size`     | 训练批次大小     | 1024   |
| `MAX_ITEM_LIST_LENGTH` | 最大序列长度     | 200    |

切换数据集时，修改 `config.yaml` 中对应的数据集配置块，取消注释并注释其他数据集即可。

## 致谢

本项目基于以下优秀开源工作构建，在此表示衷心感谢：

- **Mamba / Mamba2**：[state-spaces/mamba](https://github.com/state-spaces/mamba) — 状态空间模型，为序列建模提供高效骨干网络
- **RecBole**：[RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole) — 推荐系统统一框架，提供数据处理、评估等基础设施
- [SSD4Rec: A Structured State Space Duality Model for Efficient Sequential Recommendation](https://dl.acm.org/doi/10.1145/3773038)
- [M3Rec: Selective State Space Models with Mixture-of-Modality Experts for Multi-Modal Sequential Recommendation](https://github.com/Xu107/M3Rec-main)

---

## 引用

如果您在研究中使用了 IGTM-Rec，请引用我们的论文：

```
@article{ZHANG2026133254,
title = {Breaking sequential bias: Importance-guided temporal multimodal recommendation via state space models},
journal = {Expert Systems with Applications},
volume = {331},
pages = {133254},
year = {2026},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2026.133254},
url = {https://www.sciencedirect.com/science/article/pii/S0957417426021639},
author = {Kang Zhang and Quan Wen and Yujian Huang and Yanmei Hu and Ruixing Huang and Na Dong and Xiaomeng Yang and Shuyi Wang},
}
```
