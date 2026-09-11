"""
Task 3: Intestinal Metaplasia (IM) Detection (Improved Version)
================================================================
Architecture: ResNet50 feature extractor + Gated Attention MIL
Key improvements over v1:
  1. Multiple Instance Learning (MIL) with attention pooling
  2. Tissue-aware grid patch sampling (skip white background)
  3. Feature pre-extraction for efficient training
  4. Proper WSI handling with OpenSlide
  5. Class-balanced training (handles 90/10 imbalance)
"""

import os
import json
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              confusion_matrix, roc_curve, auc)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'base_path': '/jhcnas7/Pathology/PathLab_data_collection/Data/GC_IM_Detection',
    'patch_size': 256,
    'resize_size': 224,
    'n_total_patches': 128,
    'n_bag_size': 96,
    'tissue_threshold': 220,
    'feature_dim': 2048,
    'hidden_dim': 256,
    'att_dropout': 0.3,
    'inst_dropout': 0.2,
    'batch_size': 4,
    'epochs': 60,
    'lr': 2e-4,
    'weight_decay': 2e-4,
    'patience': 20,
    'num_workers': 4,
    'feature_cache_dir': '/home/student/First try/task3_output/features_cache',
    'color_jitter': {'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.2, 'hue': 0.05},
}


# ============================================================================
# Fast Grid-based Patch Sampling (same as task2 v2)
# ============================================================================
def grid_patch_coords(slide, n_patches):
    """Fast grid-based patch coordinate generation with tissue filtering."""
    n_levels = slide.level_count
    thumb_level = n_levels - 1

    thumb_w, thumb_h = slide.level_dimensions[thumb_level]
    thumb = np.array(slide.read_region((0, 0), thumb_level, (thumb_w, thumb_h)).convert('L'))
    tissue_mask = thumb < CONFIG['tissue_threshold']

    patch_level = 1 if n_levels >= 3 else 0
    pw, ph = slide.level_dimensions[patch_level]
    patch_size = CONFIG['patch_size']
    stride = patch_size

    grid = []
    for y in range(0, ph - patch_size + 1, stride):
        for x in range(0, pw - patch_size + 1, stride):
            tx = int(x * thumb_w / pw)
            ty = int(y * thumb_h / ph)
            if tissue_mask[ty, tx]:
                grid.append((x, y, patch_level))

    if len(grid) < n_patches and patch_level > 0:
        px, py = slide.level_dimensions[0]
        grid = []
        for y in range(0, py - patch_size + 1, stride * 2):
            for x in range(0, px - patch_size + 1, stride * 2):
                tx = int(x * thumb_w / px)
                ty = int(y * thumb_h / py)
                if tissue_mask[ty, tx]:
                    grid.append((x, y, 0))

    if len(grid) > n_patches:
        grid = random.sample(grid, n_patches)

    return grid


# ============================================================================
# Feature Extractor
# ============================================================================
def build_feature_extractor(device):
    model = resnet50(pretrained=True)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


# ============================================================================
# Gated Attention MIL (with LayerNorm + dropout regularization)
# ============================================================================
class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, num_classes=2, att_dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.attention_V = nn.Linear(input_dim, hidden_dim)
        self.attention_U = nn.Linear(input_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.att_dropout = nn.Dropout(att_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, features, inst_dropout=0.0):
        B, N, D = features.shape
        if inst_dropout > 0 and self.training:
            mask = torch.rand(B, N, 1, device=features.device) > inst_dropout
            features = features * mask

        features_norm = self.norm(features)
        V = torch.tanh(self.attention_V(features_norm))
        U = torch.sigmoid(self.attention_U(features_norm))
        A = self.attention_w(V * U)
        A = self.att_dropout(A)
        A = torch.softmax(A, dim=1)
        aggregated = torch.sum(A * features_norm, dim=1)
        logits = self.classifier(aggregated)
        return logits, A


# ============================================================================
# Stage 1: Feature Pre-extraction (Parallel I/O + Batched GPU)
# ============================================================================
def _read_slide_patches(slide_path, n_patches, patch_size):
    """Worker: read all patches from one WSI, return uint8 numpy array."""
    slide = openslide.open_slide(slide_path)
    coords = grid_patch_coords(slide, n_patches)
    patches = []
    for x, y, level in coords:
        try:
            patch = slide.read_region((x, y), level, (patch_size, patch_size))
            patches.append(np.array(patch.convert('RGB'), dtype=np.uint8))
        except Exception:
            continue
    slide.close()
    if len(patches) >= n_patches // 2:
        return np.stack(patches)
    return None


def pre_extract_all_features(df, data_path, feature_extractor, device, cache_dir):
    """Parallel WSI reading + batched GPU feature extraction."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    os.makedirs(cache_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((CONFIG['resize_size'], CONFIG['resize_size'])),
        transforms.ColorJitter(**CONFIG['color_jitter']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    wsi_dir = os.path.join(data_path, 'WSIs')
    feature_files = {}

    to_extract = []
    for idx, row in df.iterrows():
        slide_name = row['filename']
        cache_path = os.path.join(cache_dir, os.path.basename(slide_name) + '.pt')
        if os.path.exists(cache_path):
            feature_files[slide_name] = cache_path
        else:
            to_extract.append((slide_name, os.path.join(wsi_dir, slide_name), cache_path))

    if not to_extract:
        print(f"All {len(feature_files)} slides already cached, skipping extraction.")
        return feature_files

    n_workers = min(8, len(to_extract))
    print(f"\nPre-extracting features: {len(to_extract)} slides ({len(feature_files)} cached)")
    print(f"  Workers: {n_workers} threads  |  GPU batch: 256 patches")

    pending_names = []
    pending_patches = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_map = {
            executor.submit(_read_slide_patches, path, CONFIG['n_total_patches'], CONFIG['patch_size']): (name, cache)
            for name, path, cache in to_extract
        }

        for future in tqdm(as_completed(future_map), total=len(to_extract), desc="Reading WSIs"):
            name, cache_path = future_map[future]
            patches_np = future.result()

            if patches_np is None:
                print(f"  WARNING: {name} too few tissue patches, using random")
                torch.save(torch.randn(CONFIG['n_total_patches'], CONFIG['feature_dim']), cache_path)
                feature_files[name] = cache_path
                continue

            tensors = []
            for p in patches_np:
                tensors.append(transform(Image.fromarray(p)))
            bag = torch.stack(tensors)

            pending_names.append((name, cache_path, bag.shape[0]))
            pending_patches.append(bag)

            if sum(p.shape[0] for p in pending_patches) >= 256:
                _gpu_infer_and_save(pending_patches, pending_names,
                                    feature_extractor, device, feature_files)
                pending_patches = []
                pending_names = []

    if pending_patches:
        _gpu_infer_and_save(pending_patches, pending_names,
                            feature_extractor, device, feature_files)

    print(f"Features extracted: {len(feature_files)} slides saved")
    return feature_files


def _gpu_infer_and_save(pending_patches, pending_names, feature_extractor, device, feature_files):
    """Run GPU inference on accumulated patches and save per-slide features."""
    all_patches = torch.cat(pending_patches, dim=0).to(device)
    micro = 128
    all_feats = []
    for i in range(0, all_patches.shape[0], micro):
        batch = all_patches[i:i + micro]
        with torch.no_grad():
            feats = feature_extractor(batch)
        all_feats.append(feats.cpu())
    all_feats = torch.cat(all_feats, dim=0)

    offset = 0
    for name, cache_path, n_patches in pending_names:
        feats = all_feats[offset:offset + n_patches]
        offset += n_patches
        torch.save(feats, cache_path)
        feature_files[name] = cache_path


# ============================================================================
# MIL Dataset (loads pre-extracted features)
# ============================================================================
class MILDataset(Dataset):
    def __init__(self, df, feature_files, label_map, bag_size=None, augment=False):
        self.df = df.reset_index(drop=True)
        self.feature_files = feature_files
        self.label_map = label_map
        self.bag_size = bag_size or CONFIG['n_bag_size']
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_name = row['filename']
        label = self.label_map.get(row['label'], 0)

        cache_path = self.feature_files.get(slide_name)
        if cache_path is None or not os.path.exists(cache_path):
            features = torch.zeros(CONFIG['n_bag_size'], CONFIG['feature_dim'])
        else:
            features = torch.load(cache_path, weights_only=True)

        n_available = features.shape[0]
        if n_available >= self.bag_size:
            indices = torch.randperm(n_available)[:self.bag_size]
        else:
            indices = torch.randint(0, n_available, (self.bag_size,))

        bag = features[indices]
        if self.augment:
            bag = bag + torch.randn_like(bag) * 0.01
        return bag, label, slide_name


# ============================================================================
# Training & Evaluation
# ============================================================================
def train_epoch_mil(model, dataloader, optimizer, criterion, device, inst_dropout=0.0):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for features, labels, _ in tqdm(dataloader, desc='Training', leave=False):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(features, inst_dropout=inst_dropout)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


@torch.no_grad()
def validate_mil(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_logits = []
    for features, labels, _ in tqdm(dataloader, desc='Validating', leave=False):
        features, labels = features.to(device), labels.to(device)
        logits, _ = model(features)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(probs.cpu().numpy()[:, 1])
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    metrics = {}
    if len(np.unique(all_labels)) >= 2:
        metrics['val_auc'] = roc_auc_score(all_labels, all_logits)
    else:
        metrics['val_auc'] = float('nan')
    metrics['val_acc'] = accuracy_score(all_labels, all_preds)
    metrics['val_loss'] = total_loss / len(dataloader)
    return metrics


@torch.no_grad()
def test_mil(model, dataloader, device):
    model.eval()
    all_preds = []
    all_logits_full = []
    all_labels = []
    all_paths = []
    for features, labels, slide_names in tqdm(dataloader, desc='Testing', leave=False):
        features, labels = features.to(device), labels.to(device)
        logits, _ = model(features)
        probs = torch.softmax(logits, dim=1)
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_logits_full.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_paths.extend(slide_names)

    all_preds = np.array(all_preds)
    all_logits_full = np.array(all_logits_full)
    all_labels = np.array(all_labels)

    n_unique = len(np.unique(all_labels))
    if n_unique >= 2:
        macro_auc = roc_auc_score(all_labels, all_logits_full[:, 1])
        fpr, tpr, _ = roc_curve(all_labels, all_logits_full[:, 1])
        roc_auc_val = auc(fpr, tpr)
    else:
        macro_auc = float('nan')
        fpr = tpr = None
        roc_auc_val = float('nan')

    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        'Macro-AUC': macro_auc,
        'Weighted-F1': weighted_f1,
        'Macro-ACC': macro_acc,
        'ROC-AUC': roc_auc_val,
    }

    print("\n" + "=" * 50)
    print("=== PWH Test Set Performance ===")
    print("=" * 50)
    for m, v in metrics.items():
        print(f"  {m}: {v:.4f}" if not math.isnan(v) else f"  {m}: N/A")

    # Save predictions
    pred_df = pd.DataFrame({
        'file_name': all_paths,
        'IM': all_logits_full[:, 1],
        'nonIM': all_logits_full[:, 0],
    })
    pred_df.to_csv('IM.csv', index=False)
    print("\nPredictions saved to IM.csv")

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NonIM', 'IM'], yticklabels=['NonIM', 'IM'])
    plt.title('Confusion Matrix - IM Detection (MIL)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Confusion matrix saved to confusion_matrix.png")

    # ROC Curve
    if fpr is not None and tpr is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc_val:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - IM Detection (MIL)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300)
        print("ROC curve saved to roc_curve.png")

    return metrics, all_logits_full, all_labels, all_paths, cm, fpr, tpr


# ============================================================================
# Data Loading
# ============================================================================
def normalize_text(value):
    if pd.isna(value):
        return value
    return str(value).strip()


def load_pwh_data(data_path):
    csv_path = os.path.join(data_path, 'PWH', 'label.csv')
    df = pd.read_csv(csv_path)
    print(f"CSV columns: {list(df.columns)}")

    # Auto-detect label column
    for candidate in ['class', 'Class', 'label', 'Label', 'category']:
        if candidate in df.columns:
            df['label'] = df[candidate].map(normalize_text)
            print(f"  Mapped '{candidate}' -> 'label'")
            break

    # Auto-detect filename column
    for candidate in ['slide', 'Slide', 'filename', 'file_name', 'id', 'name']:
        if candidate in df.columns:
            df['filename'] = df[candidate]
            print(f"  Mapped '{candidate}' -> 'filename'")
            break

    valid_labels = ['Intestinal metaplasia', 'Not Intestinal metaplasia']
    df = df[df['label'].isin(valid_labels)].reset_index(drop=True)

    print(f"Total PWH samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    return df


# ============================================================================
# Main
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")

    if not HAS_OPENSLIDE:
        print("\n" + "!" * 60)
        print("ERROR: OpenSlide is not installed!")
        print("!" * 60)
        return

    base_path = CONFIG['base_path']
    cache_dir = CONFIG['feature_cache_dir']
    label_map = {'Intestinal metaplasia': 1, 'Not Intestinal metaplasia': 0}

    # Load data
    print("\nLoading PWH dataset...")
    df = load_pwh_data(base_path)

    # Split: 56% train / 14% val / 30% test (stratified)
    train_val_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['label']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label']
    )
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    # Stage 1: Pre-extract features
    print("\n" + "=" * 50)
    print("STAGE 1: Feature Pre-extraction")
    print("=" * 50)
    feature_extractor = build_feature_extractor(device)

    pwh_data_path = os.path.join(base_path, 'PWH')
    all_feature_files = {}
    for name, sub_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        sub_cache = os.path.join(cache_dir, name)
        all_feature_files[name] = pre_extract_all_features(
            sub_df, pwh_data_path, feature_extractor, device, sub_cache
        )

    del feature_extractor
    torch.cuda.empty_cache()

    # Stage 2: Train MIL
    print("\n" + "=" * 50)
    print("STAGE 2: Attention MIL Training")
    print("=" * 50)

    train_dataset = MILDataset(train_df, all_feature_files['train'],
                                label_map, augment=True)
    val_dataset = MILDataset(val_df, all_feature_files['val'],
                              label_map, augment=False)
    test_dataset = MILDataset(test_df, all_feature_files['test'],
                               label_map, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                               shuffle=True, num_workers=CONFIG['num_workers'],
                               drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'] * 2,
                             shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'] * 2,
                              shuffle=False, num_workers=CONFIG['num_workers'])

    mil_model = GatedAttentionMIL(
        input_dim=CONFIG['feature_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_classes=2,
        att_dropout=CONFIG['att_dropout'],
    ).to(device)
    print(f"MIL parameters: {sum(p.numel() for p in mil_model.parameters()):,}")

    # Class-balanced loss (handles extreme 90/10 imbalance)
    im_count = (train_df['label'] == 'Intestinal metaplasia').sum()
    non_im_count = (train_df['label'] == 'Not Intestinal metaplasia').sum()
    weight_im = len(train_df) / (2.0 * im_count)
    weight_non_im = len(train_df) / (2.0 * non_im_count)
    class_weights = torch.tensor([weight_non_im, weight_im], device=device, dtype=torch.float32)
    print(f"Class weights: nonIM={weight_non_im:.3f}, IM={weight_im:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(mil_model.parameters(), lr=CONFIG['lr'],
                             weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_auc = 0
    patience_counter = 0
    history = []

    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch_mil(mil_model, train_loader,
                                                  optimizer, criterion, device,
                                                  inst_dropout=CONFIG['inst_dropout'])
        val_metrics = validate_mil(mil_model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc']:.4f} | "
              f"Val AUC: {val_metrics['val_auc']:.4f} | LR: {current_lr:.2e}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['val_loss'],
            'val_acc': val_metrics['val_acc'],
            'val_auc': val_metrics['val_auc'],
        })

        if val_metrics['val_auc'] > best_val_auc:
            best_val_auc = val_metrics['val_auc']
            patience_counter = 0
            torch.save({
                'model_state_dict': mil_model.state_dict(),
                'config': CONFIG,
                'history': history,
            }, 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best
    checkpoint = torch.load('best_model.pth', weights_only=True)
    mil_model.load_state_dict(checkpoint['model_state_dict'])

    # Stage 3: Test
    print("\n" + "=" * 50)
    print("STAGE 3: PWH Test Set Evaluation")
    print("=" * 50)
    metrics, all_logits, all_labels, all_paths, cm, fpr, tpr = test_mil(
        mil_model, test_loader, device
    )

    safe_metrics = {}
    for k, v in metrics.items():
        val = float(v)
        safe_metrics[k] = None if math.isnan(val) else val
    with open('metrics.json', 'w') as f:
        json.dump(safe_metrics, f, indent=2)
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 50)
    print("Task 3 (v2 MIL) completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
