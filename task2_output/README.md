# Task 2: LUAD vs LUSC 分类

## 任务概述

本任务实现了肺部非小细胞癌(NSCLC)组织分型分类模型，用于区分肺腺癌(LUAD)和肺鳞状细胞癌(LUSC)。

### 任务目标
1. 使用 TCGA 数据训练分类模型
2. 在 Nanfang 队列上评估模型性能
3. 输出预测结果和性能指标

## 数据说明

### TCGA 训练数据
- **路径**: `/jhcnas7/Pathology/PathLab_data_collection/Data/NSCLC/TCGA`
- **格式**: 全片扫描图像 (WSI, .svs格式)
- **样本量**: ~1000+ 个幻灯片
- **标注文件**: `TCGA/label.csv`

### Nanfang 测试数据
- **路径**: `/jhcnas7/Pathology/PathLab_data_collection/Data/NSCLC/Nanfang`
- **格式**: 全片扫描图像 (WSI, .svs格式)
- **样本量**: ~600+ 个幻灯片
- **标注文件**: `Nanfang/label.csv`
- **用途**: 独立测试集，评估模型泛化能力

### 标签定义
- **LUAD**: 肺腺癌 (肺癌最常见类型)
- **LUSC**: 肺鳞状细胞癌 (第二常见肺癌类型)

## 模型架构

### 基础模型
- **网络**: ResNet-50 (ImageNet 预训练)
- **输入尺寸**: 256×256 像素

### 分类头
```
ResNet-50 特征提取
    ↓
全连接: 2048 → 512 (ReLU)
    ↓
Dropout: 0.5
    ↓
输出: 512 → 2 (Softmax)
    ↓
分类: [LUSC, LUAD]
```

## 数据处理

### WSI 处理策略
由于全片扫描图像尺寸大 (通常 > 100,000×100,000 像素)：
1. 随机采样 256×256 的图像块
2. 使用 OpenSlide 库读取指定区域
3. 单一块 → 单一预测 (简化方法)
4. 如果 OpenSlide 不可用，使用虚拟图像进行演示

### 数据增强 (训练集)
- 随机水平/垂直翻转
- 随机旋转 (±15°)
- 颜色抖动 (亮度±0.2, 对比度±0.2)
- 缩放到 256×256

### 归一化
ImageNet 标准:
```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

## 训练配置

### 数据分割
```
TCGA 训练数据: ~1000 样本
├── 训练: 800 (80%)
└── 验证: 200 (20%)

Nanfang 测试数据: ~600 样本
└── 用于最终评估 (无反馈到训练)
```

### 训练参数
- **批大小**: 32
- **优化器**: Adam (lr=1e-4)
- **损失函数**: CrossEntropyLoss
- **最大轮次**: 20
- **早停止**: 5 epoch 无改进

### 学习率调度
ReduceLROnPlateau:
- 监控: 验证损失
- 衰减: 0.5×
- 耐心: 3 epochs

## 评估指标

### 在 Nanfang 队列上的指标

1. **Macro-AUC**
   - 两类 AUC 的平均值
   
2. **Weighted-F1**
   - 按类别频率加权的 F1 分数
   
3. **Macro-Accuracy**
   - 每类准确度的平均值
   
4. **混淆矩阵**
   ```
           预测 LUSC  预测 LUAD
   实际 LUSC    TN       FP
   实际 LUAD    FN       TP
   ```
   
5. **ROC 曲线**
   - 判别能力可视化

## 输出文件

### 预测结果 - NSCLC_prediction.csv
```csv
file_name,logits_LUAD,logits_LUSC
slide1.svs,0.8765,0.1235
slide2.svs,0.2341,0.7659
```

### 其他输出
- `best_model.pth`: 最佳模型权重
- `confusion_matrix.png`: 混淆矩阵热力图
- `roc_curve.png`: ROC 曲线
- `metrics.json`: 评估指标
- `training.log`: 训练日志

## 环境配置

### 依赖库
```
PyTorch >=2.0.0
torchvision >=0.15.0
numpy, pandas, scikit-learn
matplotlib, seaborn
Pillow, opencv-python
openslide-python >= 1.1.2  (可选，用于 WSI)
```

### 硬件要求
- GPU: CUDA 12.0+ (推荐)
- GPU 内存: 12GB+
- 系统内存: 32GB+
- 存储: 100GB+ (WSI 数据较大)

## 运行指导

### 完整运行
```bash
cd task2_output/
python3 train_nsclc.py
```

### 预期耗时
- 总耗时: ~45-60 分钟 (V100)
- WSI 加载可能较慢

## 性能预期

跨数据集转移学习的预期：

| 指标 | 预期值 | 说明 |
|-----|--------|-----|
| Macro-AUC | 0.70-0.85 | 不同病理中心的差异影响 |
| Weighted-F1 | 0.68-0.82 | 中等性能 |
| Macro-ACC | 0.70-0.80 | 两类区分度有限 |

## 常见问题

### Q1: OpenSlide 安装失败
A: 使用系统包管理器安装依赖:
```bash
# Ubuntu
sudo apt-get install libopenslide0

# macOS
brew install openslide
```

### Q2: WSI 加载速度慢
A: 减小 batch_size 或启用多进程:
```python
DataLoader(..., num_workers=4)
```

### Q3: 预测文件格式错误
A: 检查 label.csv 中的 'filename' 列名。

## 改进方向

1. **多尺度分析**: 使用多个放大倍率
2. **整片预测**: 多块级聚合 (Multiple Instance Learning)
3. **特征提取**: 使用更强大的主干网络 (EfficientNet-B7)
4. **域适应**: 减少 TCGA-Nanfang 域差异

## 参考资源

- OpenSlide 文档: https://openslide.org/
- ResNet 论文: https://arxiv.org/abs/1512.03385
- TCGA 项目: https://www.cancer.gov/tcga

---

**任务完成日期**: 2024年6月14日  
**跨中心评估**: ✓ Nanfang 队列测试
