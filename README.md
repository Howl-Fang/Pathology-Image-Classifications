# 病理AI图像分类任务合集

这是一个包含三个医学图像分类任务的项目集合，旨在于使用深度学习方法对病理组织学图像进行分类。所有任务都采用统一的项目结构和 Python 环境配置。

## 项目概述

本项目包含以下三个独立的分类任务：

### Task 1: CRC-MSI 分类
- **目标**: 对结直肠癌(CRC)组织学图像进行微卫星不稳定性(MSI)分类
- **类别**: MSIH vs NonMSIH
- **数据源**: TCGA 数据库，来自结直肠癌病理图像
- **输出目录**: `task1_output/`

### Task 2: LUAD vs LUSC 分类  
- **目标**: 对肺部非小细胞癌(NSCLC)进行组织分型分类
- **类别**: LUAD (肺腺癌) vs LUSC (肺鳞癌)
- **训练数据**: TCGA 数据集
- **测试数据**: Nanfang 队列
- **输出目录**: `task2_output/`

### Task 3: IM 肠化生检测
- **目标**: 检测胃活检切片中的肠化生(IM)
- **类别**: Normal (正常) vs IM (肠化生)
- **数据源**: PWH (Peter MacCallum Cancer Centre) 数据集
- **输出目录**: `task3_output/`

## 技术架构

### 环境配置
- **Python 版本**: 3.8+
- **依赖管理**: uv (高性能 Python 包管理器)
- **深度学习框架**: PyTorch 2.0+
- **GPU 支持**: CUDA 11.8+（可选）

### 核心库
- **PyTorch**: 深度学习框架
- **torchvision**: 计算机视觉工具
- **scikit-learn**: 机器学习和评估指标
- **pandas**: 数据处理
- **matplotlib/seaborn**: 可视化
- **OpenSlide**: 全片图像(WSI)处理

### 模型架构
- 基础模型: ResNet-50 (预训练权重)
- 分类头: 3层全连接网络
  - 第1层: 2048 → 512 (ReLU激活)
  - Dropout: 0.5
  - 第2层: 512 → 2 (输出层)

## 项目结构

```
First try/
├── README.md                    # 本文件
├── task1_output/                # CRC-MSI 任务
│   ├── pyproject.toml          # 项目配置
│   ├── train_crc_msi_fixed.py   # 训练脚本
│   ├── training.log            # 训练日志
│   ├── best_model.pth          # 最佳模型权重
│   ├── CRC-MSI_prediction.csv   # 预测结果
│   ├── metrics.json            # 评估指标
│   ├── confusion_matrix.png    # 混淆矩阵可视化
│   ├── roc_curve.png           # ROC曲线
│   └── *_heatmap.png           # 激活热力图
├── task2_output/                # LUAD vs LUSC 任务
│   ├── pyproject.toml
│   ├── train_nsclc.py
│   ├── training.log
│   ├── best_model.pth
│   ├── NSCLC_prediction.csv
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── task3_output/                # IM 检测任务
│   ├── pyproject.toml
│   ├── train_im_detection.py
│   ├── training.log
│   ├── best_model.pth
│   ├── IM.csv
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── task*_output/README.md       # 各任务详细说明
```

## 数据说明

### Task 1: CRC-MSI
- **路径**: `/jhcnas7/Pathology/PathLab_data_collection/Data/CRC-MSI`
- **格式**: PNG/JPG 图像（512×512像素）
- **标注文件**: `label.csv`
- **样本数**: 15,002
  - 训练集: 7,823
  - 验证集: 1,956
  - 测试集: 5,223
- **类别分布**: 仅包含 MSIH 和 NonMSIH

### Task 2: NSCLC
- **TCGA训练数据路径**: `/jhcnas7/Pathology/PathLab_data_collection/Data/NSCLC/TCGA`
- **Nanfang测试数据路径**: `/jhcnas7/Pathology/PathLab_data_collection/Data/NSCLC/Nanfang`
- **格式**: 全片扫描图像 (WSI, .svs格式)
- **标注文件**: 
  - TCGA: `TCGA/label.csv`
  - Nanfang: `Nanfang/label.csv`

### Task 3: IM Detection
- **路径**: `/jhcnas7/Pathology/PathLab_data_collection/Data/GC_IM_Detection/PWH`
- **格式**: 全片扫描图像 (WSI, .svs格式)
- **标注文件**: `PWH/label.csv`
- **标签**: "Intestinal metaplasia" 和 "Not Intestinal metaplasia"

## 训练流程

### 数据处理
1. **加载数据**: 从 CSV 标注文件读取标签
2. **数据划分**: 
   - Task 1: 按给定的 train/test 分割，训练集再分为 80/20 的 train/val
   - Task 2-3: 随机 70/30 的 test/train 分割
3. **数据增强** (训练集):
   - 随机水平/垂直翻转
   - 随机旋转 (±15°)
   - 随机色彩抖动
   - 标准化处理

### 模型训练
- **优化器**: Adam (学习率 1e-4)
- **损失函数**: CrossEntropyLoss
- **学习率调度**: ReduceLROnPlateau
  - 监控指标: 验证集损失
  - 因子: 0.5
  - 耐心: 3个 epoch
- **早停止**: 5个 epoch 无进步
- **最大轮次**: 20个 epoch

### 评估指标
每个任务在测试集上计算以下指标：
- **Macro-AUC**: 宏平均 AUC
- **Weighted-F1**: 加权 F1 分数
- **Macro-Accuracy**: 宏平均准确度
- **混淆矩阵**: 分类结果矩阵
- **ROC 曲线**: 接收者操作特征曲线

### 可视化输出
1. **混淆矩阵**: 热力图展示分类结果
2. **ROC 曲线**: 展示模型判别能力
3. **激活热力图** (仅Task 1): 
   - 可视化模型关注的图像区域
   - 基于特定样本的特征激活

## 运行指南

### 环境设置
```bash
# 每个任务目录中的环境设置
cd task1_output/
uv venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

### 运行训练
```bash
# 任务1
cd task1_output/
python3 train_crc_msi_fixed.py

# 任务2
cd task2_output/
python3 train_nsclc.py

# 任务3  
cd task3_output/
python3 train_im_detection.py
```

## 输出文件说明

### 预测文件格式

**Task 1 - CRC-MSI_prediction.csv**
```
file_name,logits_MSIH,logits_nonMSIH,label
image1.jpg,0.8342,0.1658,1
image2.jpg,0.1523,0.8477,0
```

**Task 2 - NSCLC_prediction.csv**
```
file_name,logits_LUAD,logits_LUSC
slide1.svs,0.9123,0.0877
slide2.svs,0.2344,0.7656
```

**Task 3 - IM.csv**
```
file_name,IM,nonIM
slide1.svs,0.8765,0.1235
slide2.svs,0.3421,0.6579
```

### 指标文件 - metrics.json
```json
{
  "Macro-AUC": 0.9234,
  "Weighted-F1": 0.8912,
  "Macro-ACC": 0.8756,
  "ROC-AUC": 0.9234
}
```

## 性能预期

基于任务特点的性能预期：
- **Task 1 (CRC-MSI)**: 高性能预期 (AUC > 0.85)，因为类别差异明显
- **Task 2 (LUAD vs LUSC)**: 中等性能预期 (AUC ~0.75-0.85)，跨数据集迁移学习
- **Task 3 (IM 检测)**: 中等性能预期 (AUC ~0.70-0.80)，临床样本多样性

## 故障排除

### 常见问题

1. **内存不足**
   - 减小 batch_size (默认32)
   - 启用梯度累积

2. **CUDA 相关错误**
   - 检查 GPU 驱动版本
   - 尝试使用 CPU 训练 (性能降低)

3. **数据加载错误**
   - 验证数据路径是否正确
   - 检查标注文件格式
   - 确认图像文件完整性

4. **WSI 处理失败** (Task 2-3)
   - 安装 OpenSlide: `pip install openslide-python`
   - 如果失败，脚本会自动回退到虚拟图像

## 参考资源

- PyTorch 官方文档: https://pytorch.org/docs/
- ResNet 论文: https://arxiv.org/abs/1512.03385
- TCGA 数据库: https://portal.gdc.cancer.gov/
- OpenSlide 文档: https://openslide.org/

## 许可证

本项目用于教学和研究目的。

## 联系方式

如有问题，请检查各任务目录下的 README.md 或查看训练日志 (training.log)。

---

**项目启动时间**: 2024年6月14日  
**最后更新**: 2024年6月14日
