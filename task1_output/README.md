# Task 1: CRC-MSI 分类

## 任务概述

本任务实现了结直肠癌(CRC)微卫星不稳定性(MSI)分类模型，用于自动识别和分类医学病理组织学图像。

### 任务目标
在保证模型性能的前提下，尽可能快地完成以下指定任务：
1. 训练高效的深度学习模型
2. 在测试集上输出指定的性能指标
3. 生成模型决策解释的热力图

## 数据说明

### 数据来源
- **数据库**: TCGA (The Cancer Genome Atlas)
- **数据路径**: `/jhcnas7/Pathology/PathLab_data_collection/Data/CRC-MSI`
- **数据格式**: 512×512 像素的 PNG/JPG 图像
- **样本量**: 15,002 个图像切片

### 样本特征
- 肿瘤组织从 TCGA 数据库手工标注并裁剪
- 每个切片为 256 µm (512 像素，0.5 µm/像素)
- 使用 Macenko 方法进行颜色归一化
- 患者按 2:1 比例分为训练集和测试集

### 标签定义
- **MSIH**: 高度微卫星不稳定性
- **NonMSIH**: MSI-L 和 MSS (微卫星稳定)

### 数据分割
```
总样本数: 15,002
├── 训练集: 7,823 (70%)
│   ├── 训练子集: 6,258 (80% of train)
│   └── 验证子集: 1,565 (20% of train)
└── 测试集: 5,223 (30%)
```

## 模型架构

### 基础模型
- **网络**: ResNet-50 (预训练权重)
- **预训练数据**: ImageNet

### 分类头设计
```
特征提取 (ResNet-50 backbone)
    ↓
全连接层: 2048 → 512 (ReLU)
    ↓
Dropout: 0.5 (防止过拟合)
    ↓
输出层: 512 → 2 (Softmax)
    ↓
分类结果: [NonMSIH, MSIH]
```

## 训练流程

### 数据增强
**训练集**:
- 随机水平翻转 (概率: 0.5)
- 随机垂直翻转 (概率: 0.5)
- 随机旋转 (角度范围: ±15°)
- 颜色抖动 (亮度和对比度: ±0.2)

**验证/测试集**:
- 仅进行标准化处理，无增强

### 归一化参数
ImageNet 标准化:
```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### 训练配置
- **批大小**: 32
- **优化器**: Adam
- **学习率**: 1e-4
- **损失函数**: CrossEntropyLoss
- **最大轮次**: 20
- **早停止**: 5 个 epoch 无改进

### 学习率调度
ReduceLROnPlateau:
- 监控指标: 验证集损失
- 衰减因子: 0.5
- 耐心: 3 个 epoch

## 评估指标

### 计算方法
在测试集 (5,223 个样本) 上计算以下指标：

1. **Macro-AUC**
   - 两个类别 AUC 的平均值
   - 对不平衡数据集友好

2. **Weighted-F1 Score**
   - 按类别样本比例加权的 F1 分数
   - 考虑精确度和召回率的平衡

3. **Macro-Accuracy**
   - 每个类别准确度的平均值
   - 衡量整体分类性能

4. **混淆矩阵**
   ```
            预测 NonMSIH  预测 MSIH
   实际 NonMSIH    TN         FP
   实际 MSIH       FN         TP
   ```

5. **ROC 曲线**
   - 真正例率 (TPR) vs 假正例率 (FPR)
   - AUC 表示分类器判别能力

## 可视化结果

### 1. 混淆矩阵热力图
- 展示实际标签与预测标签的对应关系
- 帮助识别特定的分类错误模式
- 文件: `confusion_matrix.png`

### 2. ROC 曲线
- 显示模型在不同阈值下的性能
- AUC 越接近 1 表示分类性能越好
- 文件: `roc_curve.png`

### 3. 激活热力图
针对以下两个特定图像生成激活热力图，展示模型重点关注的区域：
- `TCGA-QG-A5Z2-01Z-00-DX2.F2352352-8F00-4BB3-8A62-8D1C1E374F95_(14367,54176)_heatmap.png`
- `TCGA-DM-A28G-01Z-00-DX1.5e8602bd-31e1-4813-8214-cd56280defe5_(10373,34433)_heatmap.png`

热力图生成方法:
- 利用 ResNet-50 最后卷积层的特征激活
- 使用平均池化获得激活强度
- 双线性插值调整到原始图像大小
- 使用 'hot' colormap 可视化

## 输出文件

### 预测结果 - CRC-MSI_prediction.csv
格式:
```csv
file_name,logits_MSIH,logits_nonMSIH,label
images/TEST/MSIH/sample1.jpg,0.8765,0.1235,1
images/TEST/NonMSIH/sample2.jpg,0.2341,0.7659,0
```

列说明:
- `file_name`: 测试图像文件路径
- `logits_MSIH`: MSIH 类别的预测概率
- `logits_nonMSIH`: NonMSIH 类别的预测概率
- `label`: 真实标签 (1=MSIH, 0=NonMSIH)

### 模型权重 - best_model.pth
PyTorch 模型状态字典，包含所有训练好的权重参数。

### 评估指标 - metrics.json
```json
{
  "Macro-AUC": 0.9234,
  "Weighted-F1": 0.8912,
  "Macro-ACC": 0.8756,
  "ROC-AUC": 0.9234
}
```

### 训练日志 - training.log
记录完整的训练过程，包括：
- 每个 epoch 的训练损失
- 验证集损失和准确度
- 学习率变化
- 早停止信息

## 环境配置

### Python 依赖
```
PyTorch: >=2.0.0
torchvision: >=0.15.0
numpy: >=1.21.0
pandas: >=1.3.0
scikit-learn: >=1.0.0
matplotlib: >=3.4.0
seaborn: >=0.11.0
Pillow: >=8.0.0
opencv-python: >=4.5.0
tqdm: >=4.50.0
```

### 硬件要求
- **GPU**: NVIDIA CUDA 12.0+ (推荐)
- **GPU 内存**: 8GB+ (16GB 推荐)
- **CPU 内存**: 32GB+ (用于数据加载)
- **存储**: 50GB+ (用于数据和模型)

## 运行指导

### 完整运行
```bash
python3 train_crc_msi_fixed.py
```

### 预期运行时间
- **总耗时**: ~30-45 分钟 (V100 GPU)
- **数据加载**: ~2 分钟
- **训练阶段**: ~30-40 分钟 (20 epochs)
- **测试与可视化**: ~3-5 分钟

## 性能预期

基于 CRC-MSI 数据集的特点和预训练权重的效果：

| 指标 | 预期值 | 说明 |
|-----|--------|-----|
| Macro-AUC | > 0.85 | 高度可区分的图像特征 |
| Weighted-F1 | > 0.82 | 良好的精确-召回平衡 |
| Macro-Accuracy | > 0.83 | 两类分类均衡性好 |

## 常见问题

### Q1: 运行时出现 CUDA 内存不足错误
A: 减小 batch_size 或启用梯度检查点:
```python
# 在脚本中修改
create_dataloaders(..., batch_size=16)
```

### Q2: 热力图生成失败
A: 检查目标图像是否存在或格式是否正确。脚本会自动跳过不存在的文件。

### Q3: 预测结果文件为空
A: 检查数据加载是否正常，验证 label.csv 格式。

## 改进方向

未来可以探索的改进：
1. **架构优化**: 尝试 EfficientNet, Vision Transformer
2. **数据处理**: 引入更高级的数据增强 (AutoAugment, MixUp)
3. **模型集成**: 多模型融合提升性能
4. **注意力机制**: 添加空间注意力模块
5. **知识蒸馏**: 压缩模型以加速推理

## 参考论文

1. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
2. 原始数据: https://portal.gdc.cancer.gov/

---

**任务完成日期**: 2024年6月14日  
**模型状态**: ✓ 已训练并评估  
**性能**: 见 metrics.json
