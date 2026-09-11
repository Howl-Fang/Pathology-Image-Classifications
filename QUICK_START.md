# 快速开始指南

## 🚀 项目概览

本项目包含 **3 个并行的医学图像分类任务**，已全部启动并正在执行中。

| 任务 | 类型 | 数据量 | 状态 |
|-----|------|--------|------|
| Task 1 | CRC-MSI 分类 | 15,002 图像 | 🟡 训练中 |
| Task 2 | LUAD vs LUSC | 1,600+ WSI | 🟡 训练中 |
| Task 3 | IM 肠化生检测 | 400+ WSI | 🟡 训练中 |

---

## 📂 关键文件位置

```
/storage/jmabq/Users/howl/First try/

📄 文档
├── README.md                  # 项目总述
├── FINAL_REPORT.md           # 最终报告
├── PROJECT_SUMMARY.md        # 技术细节
├── EXECUTION_LOG.md          # 执行日志
└── QUICK_START.md           # 本文件

📁 Task 1
├── task1_output/README.md    # Task 1 文档
├── train_crc_msi_fixed.py    # 训练脚本
└── training.log             # 日志文件

📁 Task 2
├── task2_output/README.md    # Task 2 文档
├── train_nsclc.py            # 训练脚本
└── training.log             # 日志文件

📁 Task 3
├── task3_output/README.md    # Task 3 文档
├── train_im_detection.py     # 训练脚本
└── training.log             # 日志文件
```

---

## ⏱️ 预期时间表

```
当前时间: ~22:50
Task 1 完成: 约 23:20 (30分钟)
Task 2 完成: 约 23:50 (60分钟)
Task 3 完成: 约 23:40 (50分钟)

全部完成: 约 00:00 (总计 ~70分钟)
```

---

## 🔍 监控进度

### 方式 1: 自动脚本
```bash
python3 "/storage/jmabq/Users/howl/First try/monitor.py"
```

### 方式 2: 实时日志
```bash
# Task 1
tail -f "/storage/jmabq/Users/howl/First try/task1_output/training.log"

# Task 2
tail -f "/storage/jmabq/Users/howl/First try/task2_output/training.log"

# Task 3
tail -f "/storage/jmabq/Users/howl/First try/task3_output/training.log"
```

### 方式 3: 检查输出文件
```bash
ls -lh "/storage/jmabq/Users/howl/First try/task1_output/" | grep metrics
ls -lh "/storage/jmabq/Users/howl/First try/task2_output/" | grep metrics
ls -lh "/storage/jmabq/Users/howl/First try/task3_output/" | grep metrics
```

---

## 📊 预期输出

### Task 1: CRC-MSI_prediction.csv
```
file_name,logits_MSIH,logits_nonMSIH,label
image1.jpg,0.8765,0.1235,1
image2.jpg,0.2341,0.7659,0
```

### Task 2: NSCLC_prediction.csv
```
file_name,logits_LUAD,logits_LUSC
slide1.svs,0.9234,0.0766
slide2.svs,0.1523,0.8477
```

### Task 3: IM.csv
```
file_name,IM,nonIM
PWH_slide1.svs,0.8765,0.1235
PWH_slide2.svs,0.3421,0.6579
```

### 性能指标: metrics.json
```json
{
  "Macro-AUC": 0.9234,
  "Weighted-F1": 0.8912,
  "Macro-ACC": 0.8756,
  "ROC-AUC": 0.9234
}
```

---

## 🎯 关键信息

### 数据源
- **Task 1**: `/jhcnas7/Pathology/.../Data/CRC-MSI` (15K 图像)
- **Task 2**: `/jhcnas7/Pathology/.../Data/NSCLC` (TCGA + Nanfang)
- **Task 3**: `/jhcnas7/Pathology/.../Data/GC_IM_Detection` (PWH)

### 模型配置
- 所有任务使用 **ResNet-50**（预训练 ImageNet）
- 批大小: **32**
- 最大轮次: **20**（带早停止）
- 优化器: **Adam** (lr=1e-4)

### 硬件需求
- GPU: NVIDIA CUDA 12.0+
- GPU 内存: 12-16GB
- 系统内存: 32GB+

---

## ✅ 完成检查清单

打印以下命令查看完成情况：

```bash
# 显示所有输出文件
echo "=== Task 1 ===" && \
ls "/storage/jmabq/Users/howl/First try/task1_output/" | grep -E "csv|png|json" && \
echo "=== Task 2 ===" && \
ls "/storage/jmabq/Users/howl/First try/task2_output/" | grep -E "csv|png|json" && \
echo "=== Task 3 ===" && \
ls "/storage/jmabq/Users/howl/First try/task3_output/" | grep -E "csv|png|json"
```

预期输出:
```
=== Task 1 ===
CRC-MSI_prediction.csv ✓
confusion_matrix.png   ✓
metrics.json          ✓
roc_curve.png        ✓

=== Task 2 ===
confusion_matrix.png   ✓
metrics.json          ✓
NSCLC_prediction.csv  ✓
roc_curve.png        ✓

=== Task 3 ===
confusion_matrix.png   ✓
IM.csv                ✓
metrics.json          ✓
roc_curve.png        ✓
```

---

## 📚 文档导航

| 需要了解 | 查看文件 |
|---------|---------|
| 项目总体 | README.md |
| Task 1 详情 | task1_output/README.md |
| Task 2 详情 | task2_output/README.md |
| Task 3 详情 | task3_output/README.md |
| 技术细节 | PROJECT_SUMMARY.md |
| 执行进度 | EXECUTION_LOG.md |
| 完整报告 | FINAL_REPORT.md |

---

## 🐛 常见问题

### Q: 如何知道训练是否完成？
A: 方法1 - 查看 metrics.json 是否存在
```bash
test -f "/path/to/metrics.json" && echo "DONE" || echo "PENDING"
```
方法2 - 运行监控脚本
```bash
python3 "/storage/jmabq/Users/howl/First try/monitor.py"
```

### Q: 训练需要多久？
A: 约 60-90 分钟（V100 GPU），具体取决于：
- GPU 型号
- 网络速度
- 磁盘 I/O

### Q: 如何查看实时日志？
A: 
```bash
tail -f "/storage/jmabq/Users/howl/First try/task1_output/training.log"
```

### Q: 模型大小是多少？
A: 每个任务约 94MB (ResNet-50 权重)

### Q: 可以中途停止训练吗？
A: 可以，按 Ctrl+C，但会丢失进度。不建议。

---

## 📞 快速命令

```bash
# 监控进度
python3 "/storage/jmabq/Users/howl/First try/monitor.py"

# 查看 Task 1 日志
tail -50 "/storage/jmabq/Users/howl/First try/task1_output/training.log"

# 查看 Task 1 指标
cat "/storage/jmabq/Users/howl/First try/task1_output/metrics.json"

# 查看预测结果
head "/storage/jmabq/Users/howl/First try/task1_output/CRC-MSI_prediction.csv"

# 检查所有任务进程
ps aux | grep "train_" | grep -v grep

# 统计文件
find "/storage/jmabq/Users/howl/First try" -name "*.csv" -o -name "*.json" | wc -l
```

---

## 🎉 完成后

所有任务完成后，您将获得：

✅ 3 个训练好的模型 (best_model.pth)  
✅ 3 个预测结果文件 (CSV)  
✅ 3 个混淆矩阵可视化  
✅ 3 个 ROC 曲线  
✅ 3 个性能指标 JSON  
✅ 完整的文档和日志  

---

## 状态指示

```
🟢 已完成 (✓)
🟡 进行中 (⏳)
🔴 失败 (✗)
```

当前:
- 🟡 基础设施: 100% 完成
- 🟡 代码实现: 100% 完成
- 🟡 文档编写: 100% 完成
- 🟡 模型训练: 60-70% 完成
- 🟢 预期完成: ~60分钟后

---

**创建时间**: 2024年6月14日 22:45  
**预期完成**: 2024年6月15日 00:15  
**文件位置**: `/storage/jmabq/Users/howl/First try/`

