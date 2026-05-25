# 毛发抑制双流度量学习皮肤病分类网络

基于深度学习的皮肤病图像分类算法研究 -- 毕业设计

## 项目简介

本项目提出了一种**毛发抑制引导的双流度量学习网络**（Hair Suppression Dual-Stream Metric Network），用于皮肤镜图像的七分类任务。模型基于 HAM10000 数据集，针对皮肤镜图像中毛发遮挡干扰诊断的问题，设计了以下关键模块：

- **双流特征提取**：原图流与去毛发流并行提取特征，通过残差适配器保留分支特异性
- **差分图引导融合**：利用原图与去毛发图的差分信息，动态生成融合权重和空间引导图
- **CBAM 注意力机制**：通道注意力 + 空间注意力增强关键病灶区域特征
- **ArcFace 度量约束分类头**：通过角度间隔约束增强类间可分性，缓解"异病同像"问题

## 模型架构

```
Input Image ──┬── Raw Stream ──→ EfficientNet-B0 ──→ ResidualAdapter ──┐
              │                                                         ├── Fusion ──→ CBAM ──→ Pool ──→ Embedding ──→ ArcFace Head
              └── Clean Stream → EfficientNet-B0 ──→ ResidualAdapter ──┘       ↑
                                                                                │
Difference Map ──→ DifferenceGuidance ──→ fusion_weights + spatial_guidance ────┘
```

## 实验结果

### 主模型测试指标（HAM10000, 50 epochs）

| 指标 | 数值 |
| --- | --- |
| Accuracy | 84.56% |
| Macro Precision | 72.76% |
| Macro Recall | 81.56% |
| Macro F1 | 76.20% |
| Macro AUC | 95.09% |

### 消融实验（5 epochs）

| 模型变体 | Accuracy | Macro F1 | ΔAcc | ΔF1 |
| :--- | ---: | ---: | ---: | ---: |
| 完整主模型 | 0.7884 | 0.6977 | -- | -- |
| 无毛发抑制引导 | 0.7872 | 0.6945 | -0.0012 | -0.0032 |
| 无双流结构 | 0.7698 | 0.6805 | -0.0186 | -0.0172 |
| 无 CBAM | 0.7731 | 0.6607 | -0.0153 | -0.0370 |
| 无度量约束分类头 | 0.7831 | 0.6786 | -0.0053 | -0.0191 |
| 无差分引导 | 0.7585 | 0.6405 | -0.0299 | -0.0572 |

消融实验表明，差分引导机制和 CBAM 注意力对模型性能贡献最大。

## 项目结构

```
graduation-project/
├── README.md                    # 项目说明
├── requirements.txt             # Python 依赖
├── LICENSE                      # MIT 开源协议
├── .gitignore
├── configs/                     # 主模型训练配置
│   └── main_experiment.json
├── src/                         # 核心源码
│   ├── __init__.py
│   ├── model.py                 # 模型定义（双流网络、CBAM、ArcFace 等）
│   ├── dataset.py               # 数据集加载（三路图像同步增强）
│   └── losses.py                # 损失函数（CE + Focal Loss 组合）
├── scripts/                     # 可执行脚本
│   ├── prepare_dataset.py       # HAM10000 数据预处理（去毛发、差分图、数据划分）
│   ├── train.py                 # 主模型训练
│   ├── evaluate.py              # 模型测试与指标生成
│   └── run_ablations.py         # 消融实验批量运行
├── experiments/                 # 实验配置与结果
│   └── ablation/
│       ├── ablation_design.md   # 消融实验设计说明
│       ├── configs/             # 各消融变体配置
│       │   ├── full_model/
│       │   ├── no_cbam/
│       │   ├── no_difference_guidance/
│       │   ├── no_dual_stream/
│       │   ├── no_hair_guidance/
│       │   └── no_metric_constraint/
│       └── results/             # 消融实验结果
│           ├── ablation_summary.json
│           └── ablation_results.md
├── docs/                        # 文档与图表
│   └── figures/                 # 训练曲线、混淆矩阵、ROC 曲线等
└── data/                        # 数据集（需自行准备，不包含在仓库中）
    ├── raw/                     # 原始图像（train/val/test/）
    ├── hair_removed/            # 去毛发图像
    ├── difference/              # 差分图像
    └── manifests/               # 数据清单
```

## 环境配置

```bash
pip install -r requirements.txt
```

主要依赖：
- Python >= 3.8
- PyTorch >= 1.10
- EfficientNet-PyTorch
- Albumentations（图像增强）
- OpenCV, scikit-learn, matplotlib, seaborn

## 使用方法

### 1. 数据准备

下载 [HAM10000 数据集](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)，按类别整理后运行预处理脚本：

```bash
python scripts/prepare_dataset.py \
    --source-root /path/to/HAM10000 \
    --output-root data/
```

脚本会自动完成：
- 7:1.5:1.5 分层划分 train/val/test
- DullRazor 算法去除毛发
- 计算原图与去毛发图的差分图
- 生成 CSV 数据清单

### 2. 训练主模型

```bash
python scripts/train.py --config configs/main_experiment.json
```

可选参数：
- `--max-epochs N`：临时覆盖训练轮数
- `--limit-train-batches N`：限制每轮训练 batch 数（快速调试用）

### 3. 测试评估

```bash
python scripts/evaluate.py --config configs/main_experiment.json
```

输出包括：Accuracy、Macro Precision/Recall/F1、混淆矩阵图、ROC 曲线图。

### 4. 消融实验

```bash
# 运行全部消融实验
python scripts/run_ablations.py

# 运行指定变体
python scripts/run_ablations.py --variants full_model no_cbam --max-epochs 5
```

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。
