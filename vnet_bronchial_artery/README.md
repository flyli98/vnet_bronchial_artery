# V-Net 支气管动脉分割 — 实现方案说明

## 1. 研究概述

本项目实现了论文 *"Deep learning-based automatic segmentation of bronchial arteries and identification of culprit bleeding vessels in hemoptysis: a clinical feasibility study"* 中描述的 **3D V-Net** 分割网络。

网络采用 **V-Net** 架构（残差卷积块 + 编解码器结构），在跳连路径中集成 **注意力机制**（Attention Gate），并采用 **两阶段级联学习策略**（Coarse → Fine）以提升细小支气管动脉的分割精度。

**核心设计原则：**
- 网络结构参数（阶段数、特征通道数、卷积核大小、步长等）硬编码于 `config/plan_config.py`
- 数据预处理参数（强度归一化的均值/标准差、百分位裁剪范围）硬编码于 `config/plan_config.py`
- V-Net 的残差块、注意力门、级联策略、损失函数等依据论文原文实现

---

## 2. 网络架构

### 2.1 V-Net 整体结构

V-Net 采用经典的编码器-解码器（Encoder-Decoder）结构，与 PlainConvUNet 的关键区别在于 **每个阶段使用残差连接**：

```
Input [1, D, H, W]
  │
  ├─ Encoder Stage 0 (stride=1, 残差块)  ─── skip 0 ──→ Attention Gate 0 ──┐
  │                                                                         │
  ├─ Encoder Stage 1 (stride=2, 下采样+残差) ── skip 1 ──→ Attention Gate 1 ─┤
  │                                                                         │
  ├─ Encoder Stage 2 (stride=2, 下采样+残差) ── skip 2 ──→ Attention Gate 2 ─┤
  │                                                                         │
  ├─ ...                                                                    │
  │                                                                         │
  ├─ Encoder Stage 5 (stride=2, 下采样+残差)                                │
  │                                                                         │
  ├─ Bottleneck (残差块)                                                    │
  │                                                                         │
  ├─ Decoder Stage 0 (转置卷积上采样 + cat(skip) + 残差) ←──────────────────┘
  │
  ├─ Decoder Stage 1 (转置卷积上采样 + cat(skip) + 残差)
  │
  ├─ ...
  │
  ├─ Decoder Stage 4 (转置卷积上采样 + cat(skip) + 残差)
  │
  └─ Output Conv 1×1 → logits [2, D, H, W]
```

### 2.2 V-Net 结构参数

网络的所有结构超参数硬编码于 `config/plan_config.py`。以 `3d_fullres` 配置为例：

| 参数 | 值 | 说明 |
|------|----------------------|------|
| `n_stages` | 6 | 编码器阶段数（含第一阶段） |
| `features_per_stage` | [32, 64, 128, 256, 320, 320] | 各阶段特征通道数 |
| `kernel_sizes` | [[3,3,3]] × 6 | 各阶段卷积核大小 |
| `strides` | [[1,1,1], [2,2,2]×5] | 各阶段下采样步长 |
| `n_conv_per_stage` | [2, 2, 2, 2, 2, 2] | 各编码器阶段卷积数 |
| `n_conv_per_stage_decoder` | [2, 2, 2, 2, 2] | 各解码器阶段卷积数 |
| `conv_op` | Conv3d | 3D 卷积 |
| `norm_op` | InstanceNorm3d | 实例归一化 |
| `nonlin` | LeakyReLU (inplace) | 激活函数 |
| `conv_bias` | True | 卷积偏置 |
| `patch_size` | [128, 128, 128] | 训练 patch 大小 |
| `batch_size` | 2 | 批大小 |
| `spacing` | [0.7, 0.694, 0.694] | 目标体素间距 (mm) |

### 2.3 残差块（V-Net 核心特征）

每个编码器/解码器阶段使用残差连接，这是 V-Net 区别于 U-Net 的核心特征：

```python
# 残差块前向传播
residual = projection(x)       # 1×1 卷积调整通道数（若需要）
out = conv_block_n(...conv_block_2(conv_block_1(x))...)
output = out + residual        # 残差相加
```

- **输入投影**：当输入/输出通道数不一致时，使用 1×1 卷积进行通道匹配
- **下采样**：通过步长为 2 的卷积实现（非最大池化），保留更多信息
- **上采样**：通过转置卷积实现，步长来自配置参数

### 2.4 注意力门（Attention Gate）

依据论文描述，在每条跳连路径上集成注意力门：

> *"an attention mechanism was integrated into the skip connection pathways.
> This mechanism adaptively amplifies the feature weights corresponding to
> vascular regions while suppressing background tissue signals."*

注意力门计算流程：

```
gate (解码器特征, 低分辨率, 语义信息)
  │
  ├─ W_gate: 1×1×1 Conv → Norm  ──→ g
  │                                    │
skip (编码器特征, 高分辨率, 细节信息)   │
  │                                    │
  ├─ W_skip: 1×1×1 Conv → Norm  ──→ x  │
  │                                    │
  ├─ 上采样 g 至 x 的尺寸              │
  │                                    │
  ├─ q = ReLU(g + x)                   │
  ├─ alpha = sigmoid(W_psi(q))         │  注意力系数
  │                                    │
  └─ output = skip * alpha             │  加权后的跳连特征
```

注意力系数 `alpha ∈ [0, 1]` 对每个空间位置进行自适应加权，使网络聚焦于血管区域，抑制背景噪声，提升对细小支气管动脉的敏感度。

### 2.5 深度监督（Deep Supervision）

训练阶段启用深度监督：在解码器的中间层添加辅助输出头（1×1 卷积），产生低分辨率的分割预测。总损失为主输出与各辅助输出的加权求和，权重按指数衰减（0.5^i）。

---

## 3. 数据预处理

### 3.1 预处理流程

预处理参数硬编码于 `config/plan_config.py`，流程如下：

```
原始 CTA 图像
    │
    ├─ 1. 读取 (SimpleITKIO)
    │
    ├─ 2. 重采样至目标间距 (来自配置)
    │      - 数据: 三次样条插值 (order=3)
    │      - 标签: 最近邻插值 (order=1)
    │
    ├─ 3. 强度裁剪 (基于前景强度统计)
    │      - 百分位裁剪: [percentile_00_5, percentile_99_5] = [-207, 511]
    │      - CT 窗宽窗位: [WW=400 HU, WL=40 HU] → 裁剪至 [-160, 240]
    │
    ├─ 4. Z-Score 归一化 (基于前景强度统计)
    │      - mean = 145.38, std = 107.59
    │      - normalized = (x - mean) / std
    │
    └─ 5. 裁剪/填充至 patch 大小 (来自配置)
           - 训练: 随机裁剪
           - 推理: 中心裁剪 + 滑窗
```

### 3.2 前景强度统计参数

| 参数 | 值 | 用途 |
|------|------|------|
| `mean` | 145.38 | Z-Score 归一化均值 |
| `std` | 107.59 | Z-Score 归一化标准差 |
| `percentile_00_5` | -207.0 | 强度下界裁剪 |
| `percentile_99_5` | 511.0 | 强度上界裁剪 |
| `median_relative_size_after_cropping` | 1.0 | 裁剪后相对大小 |

### 3.3 数据增强

依据论文描述，训练时随机应用以下变换：

| 增强方式 | 参数 | 概率 |
|----------|------|------|
| 水平翻转 (左右镜像) | — | 0.5 |
| 垂直翻转 (冠状面镜像) | — | 0.5 |
| 图像缩放 | 缩放因子 [0.8, 1.2] | 0.3 |
| 随机裁剪 | 裁剪至 patch_size | 1.0 |
| Gamma 校正 | gamma [0.7, 1.5] | 0.15 |
| 高斯噪声 | std=0.1 | 0.15 |
| 亮度调整 | offset [-0.1, 0.1] | 0.15 |

---

## 4. 级联学习策略

### 4.1 两阶段级联架构

依据论文描述，采用两阶段级联学习：

| 阶段 | 配置 | 体素间距 | Patch 大小 | 采样策略 |
|------|------|----------|------------|----------|
| 粗分割 (Coarse) | 3d_lowres | [3.0, 3.0, 3.0] mm | [128, 128, 128] | 全局随机采样 |
| 精分割 (Fine) | 3d_fullres | [1.0, 1.0, 1.0] mm | [128, 128, 128] | 限定于粗分割 bbox (扩展20体素) |

### 4.2 训练流程

```
1. 训练粗分割模型 (3d_lowres)
   ├── 全局随机采样整个 CT 体积
   ├── 快速定位支气管动脉区域
   └── 产生粗分割掩码 (ROI 定位)

2. 运行粗分割推理
   ├── 对训练集每个样本生成粗预测
   └── 提取 3D 边界框并扩展 20 体素

3. 训练精分割模型 (3d_fullres)
   ├── 采样限定在扩展后的边界框内
   ├── 聚焦计算资源于目标血管区域
   └── 精细化血管边界细节
```

### 4.3 推理流程

```
1. 粗分割模型生成定位掩码
2. 提取边界框并扩展 20 体素 (所有方向)
3. 精分割模型处理裁剪区域
4. 结果放回原始图像空间
5. 后处理 (连通域去除 + 形态学闭运算)
```

---

## 5. 损失函数

### 5.1 组合损失

依据论文描述，损失函数为 Tversky Loss 与 ClDice Loss 的等权组合：

```
Loss = Tversky Loss + ClDice Loss
```

### 5.2 Tversky Loss

```
Tversky Loss = 1 - TP / (TP + α × FP + β × FN)
```

- `α = 0.3`（假阳性权重）
- `β = 0.7`（假阴性权重）

非对称加权使网络更严重惩罚假阴性（漏检），优先检测细小支气管动脉。

### 5.3 ClDice Loss

```
ClDice = 2 × |Centerline(Y) ∩ Centerline(Ŷ)| / (|Centerline(Y)| + |Centerline(Ŷ)|)
Loss_cl = 1 - ClDice
```

基于血管中心线（骨架化）的拓扑一致性损失，通过可微的软骨架化（soft skeletonization）实现端到端训练，减少分割结果中的拓扑断裂。

---

## 6. 训练配置

| 参数 | 值 | 来源 |
|------|------|------|
| 优化器 | AdamW | 论文 |
| 初始学习率 | 1×10⁻⁴ | 论文 |
| 学习率调度 | Cosine Annealing | 论文 |
| 权重衰减 | 1×10⁻⁵ | 论文 |
| 批大小 | 2 | 配置参数 |
| 最大训练轮数 | 100 | 论文 |
| 早停耐心值 | 15 | 论文 |
| 梯度裁剪 | max_norm=12.0 | 实践经验 |
| 交叉验证 | 5 折 | 论文 |
| 深度监督 | 启用 | 实践经验 |

---

## 7. 后处理

依据论文描述：

1. **连通域去除**：仅保留体素数 ≥ 10 的 3D 连通组件
2. **形态学闭运算**：使用半径为 1 体素的球形结构元素填充小的不连续区域

---

## 8. 项目结构

```
vnet_bronchial_artery/
├── config/
│   ├── __init__.py
│   └── plan_config.py          # 结构与预处理参数 (硬编码)
├── models/
│   ├── __init__.py
│   ├── blocks.py               # 残差卷积块 (V-Net 核心)
│   ├── attention.py            # 注意力门 (Attention Gate)
│   └── vnet.py                 # V-Net 主网络
├── data/
│   ├── __init__.py
│   ├── preprocessing.py        # 预处理 (基于前景强度统计)
│   ├── dataset.py              # PyTorch Dataset
│   └── augmentation.py         # 数据增强
├── losses/
│   ├── __init__.py
│   ├── tversky.py              # Tversky Loss (α=0.3, β=0.7)
│   ├── cldice.py               # ClDice Loss (拓扑一致性)
│   └── combined.py             # 组合损失 + 深度监督
├── trainers/
│   ├── __init__.py
│   ├── trainer.py              # 单阶段训练器
│   └── cascade.py              # 级联学习流水线
├── inference/
│   ├── __init__.py
│   ├── predictor.py            # 滑窗推理
│   └── postprocess.py          # 后处理
├── utils/
│   ├── __init__.py
│   └── metrics.py              # DSC, HD95 等指标
├── train.py                    # 训练入口
├── predict.py                  # 推理入口
├── requirements.txt
└── README.md                   # 本文件
```

---

## 9. 使用方法

### 9.1 安装依赖

```bash
pip install -r requirements.txt
```

### 9.2 训练

```bash
# 单阶段训练 (3d_fullres)
python train.py \
    --data_dir /path/to/data \
    --stage fullres \
    --output_dir ./output

# 级联训练 (粗 + 精)
python train.py \
    --data_dir /path/to/data \
    --stage cascade \
    --output_dir ./output

# 5 折交叉验证
python train.py \
    --data_dir /path/to/data \
    --stage cascade \
    --kfold 5 \
    --output_dir ./output
```

### 9.3 推理

```bash
# 单阶段推理
python predict.py \
    --image /path/to/image.nii.gz \
    --checkpoint /path/to/best_model.pth \
    --stage fullres \
    --output prediction.nii.gz

# 级联推理
python predict.py \
    --image /path/to/image.nii.gz \
    --cascade \
    --fold_dir /path/to/fold_0 \
    --output prediction.nii.gz
```

### 9.4 数据目录结构

```
data/
├── imagesTr/
│   ├── case_0000.nii.gz
│   ├── case_0001.nii.gz
│   └── ...
└── labelsTr/
    ├── case_0000.nii.gz
    ├── case_0001.nii.gz
    └── ...
```

> **注意**：所有网络结构和预处理参数已硬编码至 `config/plan_config.py`，项目运行时无需任何外部配置文件。

---
