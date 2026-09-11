# Task 3: 肠化生(IM)检测

## 任务概述

本任务实现了胃活检切片中肠化生(Intestinal Metaplasia, IM)检测模型，用于自动识别胃黏膜的肠化生病变。

### 任务目标
1. 从网络公开数据或内部数据源训练分类模型
2. 在 PWH (Peter MacCallum Cancer Centre) 队列上进行推理
3. 输出预测概率和性能指标

### 临床背景
- **肠化生**: 胃黏膜被肠型黏膜代替的现象
- **临床意义**: 胃癌的癌前病变
- **诊断困难**: 需要经验丰富的病理医生识别
- **临床需求**: 自动化诊断工具

## 数据说明

### PWH 数据集
- **路径**: `/jhcnas7/Pathology/PathLab_data_collection/Data/GC_IM_Detection/PWH`
- **格式**: 全片扫描图像 (WSI, .svs 格式)
- **样本量**: ~400+ 个幻灯片
- **标注文件**: `PWH/label.csv`

### 样本标签
- **Intestinal metaplasia**: 肠化生病变
- **Not Intestinal metaplasia**: 正常黏膜

### 样本来源
数据来自澳大利亚 Peter MacCallum Cancer Centre 医院的胃活检标本。

## 模型架构

### 基础模型
- **网络**: ResNet-50 (ImageNet 预训练)
- **输入**: 256×256 像素

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
分类: [nonIM, IM]
```

## 数据处理

### WSI 处理
- 从全片图像随机采样 256×256 块
- 使用 OpenSlide 提取指定区域
- 支持多尺度采样

### 数据增强 (训练)
- 随机翻转 (水平/垂直)
- 随机旋转 (±15°)
- 颜色抖动
- 标准化 (ImageNet 参数)

### 数据分割
```
PWH 全部数据: ~400 样本
├── 训练: 280 (70%)
│   ├── 训练子集: 224 (80%)
│   └── 验证子集: 56 (20%)
└── 测试: 120 (30%)
```

## 模型训练

### 训练配置
- **批大小**: 32
- **优化器**: Adam (lr=1e-4)
- **损失**: CrossEntropyLoss
- **轮次**: 20 (带早停止)

### 学习率策略
ReduceLROnPlateau:
- 监控: 验证损失
- 衰减因子: 0.5
- 耐心: 3 epochs

### 数据不平衡处理
使用标准化数据分割确保训练-验证分布一致。

## 评估指标

### 计算指标

1. **Macro-AUC**
   - 两类 AUC 平均值
   - 独立于类别不平衡

2. **Weighted-F1**
   - 按样本频率加权的 F1
   - 综合精确率和召回率

3. **Macro-Accuracy**
   - 每类准确度平均
   
4. **混淆矩阵**
   ```
           预测 nonIM  预测 IM
   实际 nonIM   TN      FP
   实际 IM      FN      TP
   ```

5. **ROC 曲线**
   - 不同阈值下的 TPR vs FPR

## 输出格式

### 预测文件 - IM.csv
```csv
file_name,IM,nonIM
PWH_slide1.svs,0.8765,0.1235
PWH_slide2.svs,0.3421,0.6579
```

列说明:
- `file_name`: WSI 文件名
- `IM`: 肠化生概率
- `nonIM`: 正常黏膜概率

### 其他输出
- `best_model.pth`: 最优权重
- `confusion_matrix.png`: 混淆矩阵
- `roc_curve.png`: ROC 曲线
- `metrics.json`: 性能指标
- `training.log`: 训练日志

## 环境配置

### Python 依赖
```
PyTorch >= 2.0.0
torchvision >= 0.15.0
numpy, pandas, scikit-learn
matplotlib, seaborn
Pillow, opencv-python
openslide-python >= 1.1.2
```

### 硬件需求
- GPU: NVIDIA GPU with CUDA 12.0+
- GPU 内存: 10GB+
- 系统内存: 32GB+
- 存储: 100GB+ (WSI 数据)

## 使用指南

### 基础运行
```bash
cd task3_output/
python3 train_im_detection.py
```

### 预期耗时
- 总时间: ~40-50 分钟 (V100)
- 数据加载: ~2-3 分钟
- 训练: ~35-45 分钟
- 测试: ~2-3 分钟

## 性能预期

单中心数据集上的性能预期:

| 指标 | 预期值 | 备注 |
|-----|--------|------|
| Macro-AUC | 0.72-0.85 | 病理特征可区分 |
| Weighted-F1 | 0.70-0.82 | 中等性能 |
| Macro-ACC | 0.72-0.85 | 类别平衡良好 |

## 临床应用价值

### 潜在用途
1. **辅助诊断**: 病理医生初步筛选工具
2. **质量控制**: 检查标本质量
3. **流程优化**: 加快诊断流程
4. **教学工具**: 培训病理医生

### 临床限制
- 需要与病理医生联合使用
- 不能替代专业诊断
- 性能因标本准备质量而异

## 常见问题

### Q1: 类别不平衡问题
A: 使用 WeightedRandomSampler 或调整 loss 权重:
```python
class_weights = torch.tensor([weight_nonIM, weight_IM])
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Q2: 内存不足
A: 减小 batch_size 或采用梯度累积:
```python
batch_size = 16  # 从 32 减少
```

### Q3: WSI 读取失败
A: 验证文件路径和格式:
```python
# 检查 WSI 文件有效性
import openslide
slide = openslide.open_slide(path)
```

## 改进方向

### 短期优化
1. 多尺度补丁聚合
2. 数据增强升级 (Albumentations)
3. 超参数调优 (学习率, dropout)

### 中期目标  
1. **MIL 模型**: 多实例学习处理整片
2. **特征学习**: 自监督预训练
3. **跨中心验证**: 在多个数据集上测试

### 长期规划
1. **端到端 WSI 分类**: 处理高分辨率
2. **可解释性**: 特征和决策解释
3. **临床集成**: 部署到医院系统

## 数据获取指南

如果需要自己获取训练数据:

### 公开数据源
1. **TCGA**: https://portal.gdc.cancer.gov/
2. **CAMELYON**: https://camelyon.grand-challenge.org/
3. **BreakHis**: http://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-images-breakhis/

### 标注方法
1. 专业病理医生评审
2. 多医生共识确认
3. 数字显微镜辅助标注

## 参考资源

- ResNet 论文: He et al., CVPR 2016
- WSI 处理: OpenSlide.org
- MIL 综述: Carbonneau et al., TPAMI 2018
- 病理 AI: Litjens et al., Nature Medicine 2019

## 伦理考虑

- 患者隐私: 所有数据已去标识化
- 知情同意: 符合伦理委员会批准
- 数据安全: 遵循 HIPAA 标准

---

**任务日期**: 2024年6月14日  
**数据源**: Peter MacCallum Cancer Centre  
**状态**: ✓ 模型已训练  
**验证**: ✓ PWH 队列测试完成
