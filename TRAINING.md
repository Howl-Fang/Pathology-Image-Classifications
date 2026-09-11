# 病理AI图像分类 — 训练指南

> 最后更新: 2026-07-07 | 代码版本: v2 (修复后)

---

## 1. 数据状态

| 任务 | 标签文件 | 格式 | 类别 | 状态 |
|------|----------|------|------|------|
| Task 1 CRC-MSI | 目录结构 (TRAIN/TEST) | PNG/JPG 512×512 | MSIH + nonMSIH | ✅ |
| Task 2 NSCLC | TCGA_NSCLC.csv / Nanfang_lung_NSCLC_VALID.csv | WSI .svs | LUAD + LUSC | ✅ |
| Task 3 IM Detection | PWH/label.csv | WSI .svs/.mrxs | IM + nonIM | ✅ |

---

## 2. 环境准备

```bash
# Python 依赖
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow opencv-python tqdm

# OpenSlide (Task 2 & 3 必须！不装=白训练)
sudo apt-get install libopenslide0
pip install openslide-python

# 验证
python3 -c "import torch, openslide; print('OK')"
```

---

## 3. 训练

```bash
# Task 1: CRC-MSI (~30-45 min)
cd "/home/student/First try/task1_output"
python3 train_crc_msi_fixed.py

# Task 2: NSCLC (~45-60 min)
cd "/home/student/First try/task2_output"
python3 train_nsclc.py

# Task 3: IM Detection (~40-50 min)
cd "/home/student/First try/task3_output"
python3 train_im_detection.py
```

### 输出文件（每个任务）
- `best_model.pth` — 模型权重
- `metrics.json` — Macro-AUC, Weighted-F1, Macro-ACC
- `confusion_matrix.png` / `roc_curve.png`
- 预测 CSV（CRC-MSI_prediction.csv / NSCLC_prediction.csv / IM.csv）

---

## 4. 修复记录 (v2)

| Bug | 根因 | 修复 |
|-----|------|------|
| Task 1 单类无诊断 | label.csv 仅 MSIH | 目录回退扫描 + nonMSIH 大小写处理 |
| Task 2 CSV 列名 | `class`/`slide` vs `label`/`filename` | 自动列名映射 |
| Task 3 同上 + 不平衡 | 同上 + 9:1 类别比 | 列名映射 + 类别权重 + 分层采样 |
| 所有 NaN 指标 | 单类时 roc_auc=NaN | NaN→null, 安全跳过 AUC/ROC |
| WSI 白图回退 | OpenSlide 不可用时静默降级 | 启动预检 + 明确告警 |

---

## 5. 常见问题

**Q: Val Acc 一直是 ~0.5？**
A: 检查 `python3 -c "import openslide"`，没装 OpenSlide 时模型收到空白图。

**Q: metrics.json 有 null 值？**
A: 正常。测试集单类时 AUC 无法计算。

**Q: CUDA out of memory？**
A: 改脚本中 `batch_size=32` 为 16 或 8。
