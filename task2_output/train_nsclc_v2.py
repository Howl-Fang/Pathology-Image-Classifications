"""
Task 2: LUAD vs LUSC Classification (Improved Version)
======================================================
Architecture: ResNet50 feature extractor + Gated Attention MIL
Key improvements over v1:
  1. Multiple Instance Learning (MIL) with attention pooling
  2. Tissue-aware patch sampling (skip white background)
  3. Feature pre-extraction for efficient training
  4. Proper WSI handling with OpenSlide
"""

import os
import json
import pickle
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
    'base_path': '/jhcnas7/Pathology/PathLab_data_collection/Data/NSCLC',
    'patch_size': 256,
    'resize_size': 224,           # ResNet standard input
    'n_total_patches': 128,       # Total patches to pre-extract per slide
    'n_bag_size': 64,             # Patches per bag during MIL training
    'tissue_threshold': 220,      # Grayscale threshold for tissue detection
    'feature_dim': 2048,          # ResNet50 feature dimension
    'hidden_dim': 256,            # MIL attention hidden dim
    'batch_size': 4,              # Slides per batch
    'epochs': 50,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'patience': 10,
    'num_workers': 4,
    'feature_cache_dir': '/home/student/First try/task2_output/features_cache',
}


# ============================================================================
# Tissue Detection & Patch Sampling (Fast Grid-based)
# ============================================================================
def grid_patch_coords(slide, n_patches):
    """Fast grid-based patch coordinate generation with tissue filtering.

    Strategy: read tissue mask from thumbnail, then grid-tile at level 1 (or 0),
    filter by tissue, and randomly sample N positions. No per-pixel retries.
    """
    n_levels = slide.level_count
    thumb_level = n_levels - 1   # lowest-res for tissue detection

    # Read thumbnail for tissue mask
    thumb_w, thumb_h = slide.level_dimensions[thumb_level]
    thumb = np.array(slide.read_region((0, 0), thumb_level, (thumb_w, thumb_h)).convert('L'))
    tissue_mask = thumb < CONFIG['tissue_threshold']

    # Use level 1 (medium res) for patches, fallback to level 0
    patch_level = 1 if n_levels >= 3 else 0
    pw, ph = slide.level_dimensions[patch_level]
    patch_size = CONFIG['patch_size']
    stride = patch_size  # non-overlapping grid

    # Build grid of tissue-containing positions
    grid = []
    for y in range(0, ph - patch_size + 1, stride):
        for x in range(0, pw - patch_size + 1, stride):
            # Map to thumbnail coords
            tx = int(x * thumb_w / pw)
            ty = int(y * thumb_h / ph)
            if tissue_mask[ty, tx]:
                grid.append((x, y, patch_level))

    # If too few tissue patches at level 1, try level 0
    if len(grid) < n_patches and patch_level > 0:
        px, py = slide.level_dimensions[0]
        grid = []
        for y in range(0, py - patch_size + 1, stride * 2):
            for x in range(0, px - patch_size + 1, stride * 2):
                tx = int(x * thumb_w / px)
                ty = int(y * thumb_h / py)
                if tissue_mask[ty, tx]:
                    grid.append((x, y, 0))

    # Randomly sample N positions
    if len(grid) > n_patches:
        grid = random.sample(grid, n_patches)

    return grid


# ============================================================================
# Feature Extractor
# ============================================================================
def build_feature_extractor(device):
    """Build a frozen ResNet50 feature extractor (output: 2048-dim)."""
    model = resnet50(pretrained=True)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


# ============================================================================
# Gated Attention MIL Model
# ============================================================================
class GatedAttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning with gating mechanism.
    Reference: Ilse et al. (2018), "Attention-based Deep Multiple Instance Learning"
    """
    def __init__(self, input_dim=2048, hidden_dim=256, num_classes=2):
        super().__init__()
        self.attention_V = nn.Linear(input_dim, hidden_dim)
        self.attention_U = nn.Linear(input_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )

    def forward(self, features):
        """
        Args:
            features: (B, N, feature_dim) bag of patch features
        Returns:
            logits: (B, num_classes)
            attention: (B, N, 1) attention weights
        """
        # Gated attention
        V = torch.tanh(self.attention_V(features))      # (B, N, H)
        U = torch.sigmoid(self.attention_U(features))    # (B, N, H)
        A = self.attention_w(V * U)                      # (B, N, 1)
        A = torch.softmax(A, dim=1)                      # (B, N, 1)

        # Weighted aggregation
        aggregated = torch.sum(A * features, dim=1)      # (B, input_dim)
        logits = self.classifier(aggregated)              # (B, num_classes)

        return logits, A


# ============================================================================
# Stage 1: Feature Pre-extraction
# ============================================================================
def extract_features_for_slide(slide_path, feature_extractor, device, transform):
    """Extract features from N grid-sampled tissue patches of a WSI (fast)."""
    if not HAS_OPENSLIDE:
        return None

    try:
        slide = openslide.open_slide(slide_path)
    except Exception:
        return None

    n_patches = CONFIG['n_total_patches']
    coords = grid_patch_coords(slide, n_patches)

    features_list = []

    # Process patches in micro-batches (GPU-efficient)
    micro_batch = 64
    for i in range(0, len(coords), micro_batch):
        batch_coords = coords[i:i + micro_batch]
        patches = []

        for x, y, level in batch_coords:
            try:
                patch = slide.read_region((x, y), level, (CONFIG['patch_size'], CONFIG['patch_size']))
                patch = patch.convert('RGB')
                patches.append(patch)
            except Exception:
                continue

        if not patches:
            continue

        # Transform and extract features
        patch_tensors = torch.stack([transform(p) for p in patches]).to(device)
        with torch.no_grad():
            feats = feature_extractor(patch_tensors)
        features_list.append(feats.cpu())

    slide.close()

    if features_list:
        features = torch.cat(features_list, dim=0)
        return features
    return None


def pre_extract_all_features(df, data_path, feature_extractor, device, cache_dir):
    """Pre-extract features for all slides in the dataframe."""
    os.makedirs(cache_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((CONFIG['resize_size'], CONFIG['resize_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    wsi_dir = os.path.join(data_path, 'WSIs')
    feature_files = {}

    print(f"\nPre-extracting features to {cache_dir}...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        slide_name = row['filename']
        slide_path = os.path.join(wsi_dir, slide_name)
        cache_path = os.path.join(cache_dir, slide_name.replace('.svs', '.pt'))

        if os.path.exists(cache_path):
            feature_files[slide_name] = cache_path
            continue

        result = extract_features_for_slide(slide_path, feature_extractor, device, transform)
        if result is None or result.shape[0] < CONFIG['n_bag_size']:
            # Fallback: use random features so training doesn't crash
            actual = result.shape[0] if result is not None else 0
            if result is None:
                print(f"  WARNING: Could not extract features for {slide_name}, using random features")
            else:
                print(f"  WARNING: Only {actual} patches for {slide_name}, padding with random")
            features = torch.randn(CONFIG['n_total_patches'], CONFIG['feature_dim'])
        else:
            features = result

        torch.save(features, cache_path)
        feature_files[slide_name] = cache_path

    print(f"Features extracted: {len(feature_files)} slides saved to {cache_dir}")
    return feature_files


# ============================================================================
# Stage 2: MIL Dataset (loads pre-extracted features)
# ============================================================================
class MILDataset(Dataset):
    """Dataset that loads pre-extracted patch features for MIL training."""
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

        # Randomly sample a bag of patches
        n_available = features.shape[0]
        if n_available <= self.bag_size:
            indices = torch.arange(n_available)
        else:
            indices = torch.randperm(n_available)[:self.bag_size]

        bag = features[indices]  # (bag_size, feature_dim)

        # Simple feature augmentation: add small Gaussian noise
        if self.augment:
            bag = bag + torch.randn_like(bag) * 0.01

        return bag, label, slide_name


# ============================================================================
# Training & Evaluation Functions
# ============================================================================
def train_epoch_mil(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for features, labels, _ in tqdm(dataloader, desc='Training', leave=False):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(features)
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
        all_logits.extend(probs.cpu().numpy()[:, 1])  # LUAD probability

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
    """Comprehensive test evaluation."""
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

    # Metrics
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
    print("=== Nanfang Test Set Performance ===")
    print("=" * 50)
    for m, v in metrics.items():
        print(f"  {m}: {v:.4f}" if not math.isnan(v) else f"  {m}: N/A")

    # Save predictions CSV
    pred_df = pd.DataFrame({
        'file_name': all_paths,
        'logits_LUAD': all_logits_full[:, 1],
        'logits_LUSC': all_logits_full[:, 0],
    })
    pred_df.to_csv('NSCLC_prediction.csv', index=False)
    print("\nPredictions saved to NSCLC_prediction.csv")

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['LUSC', 'LUAD'], yticklabels=['LUSC', 'LUAD'])
    plt.title('Confusion Matrix - NSCLC LUAD vs LUSC (MIL)')
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
        plt.title('ROC Curve - NSCLC LUAD vs LUSC (MIL)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300)
        print("ROC curve saved to roc_curve.png")

    return metrics, all_logits_full, all_labels, all_paths, cm, fpr, tpr


# ============================================================================
# Data Loading Helpers (adapted from v1)
# ============================================================================
def normalize_text(value):
    if pd.isna(value):
        return value
    return str(value).strip()


def _normalize_nsclc_csv(df, csv_path):
    """Auto-detect and normalize NSCLC CSV columns."""
    print(f"CSV columns: {list(df.columns)}")

    for candidate in ['class', 'Class', 'label', 'Label', 'category']:
        if candidate in df.columns:
            if candidate != 'label':
                df['label'] = df[candidate].map(normalize_text)
                print(f"  Mapped '{candidate}' -> 'label'")
            else:
                df['label'] = df['label'].map(normalize_text)
            break
    else:
        raise ValueError(f"Cannot find label column in {csv_path}. Columns: {list(df.columns)}")

    for candidate in ['slide', 'Slide', 'filename', 'file_name', 'id', 'name']:
        if candidate in df.columns:
            if candidate != 'filename':
                df['filename'] = df[candidate]
                print(f"  Mapped '{candidate}' -> 'filename'")
            break

    return df


def load_data(data_path, dataset_name):
    """Load NSCLC dataset."""
    if dataset_name == 'TCGA':
        csv_path = os.path.join(data_path, 'TCGA', 'TCGA_NSCLC.csv')
    else:
        csv_path = os.path.join(data_path, 'Nanfang', 'Nanfang_lung_NSCLC_VALID.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _normalize_nsclc_csv(df, csv_path)
    df = df[df['label'].isin(['LUAD', 'LUSC'])].reset_index(drop=True)

    print(f"Total {dataset_name} samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    return df


# ============================================================================
# Main Training Pipeline
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")

    if not HAS_OPENSLIDE:
        print("\n" + "!" * 60)
        print("ERROR: OpenSlide is not installed!")
        print("  Install with: pip install openslide-python")
        print("!" * 60)
        return

    base_path = CONFIG['base_path']
    cache_dir = CONFIG['feature_cache_dir']
    label_map = {'LUAD': 1, 'LUSC': 0}

    # ---- Load dataframes ----
    print("\nLoading data...")
    tcga_df = load_data(base_path, 'TCGA')
    nanfang_df = load_data(base_path, 'Nanfang')

    # ---- Split TCGA into train/val ----
    train_df, val_df = train_test_split(
        tcga_df, test_size=0.2, random_state=42, stratify=tcga_df['label']
    )
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test (Nanfang): {len(nanfang_df)}")

    # ---- Stage 1: Pre-extract features ----
    print("\n" + "=" * 50)
    print("STAGE 1: Feature Pre-extraction")
    print("=" * 50)

    feature_extractor = build_feature_extractor(device)

    # Extract for all datasets
    all_dfs = {
        'tcga': (train_df, os.path.join(base_path, 'TCGA')),
        'val': (val_df, os.path.join(base_path, 'TCGA')),
        'nanfang': (nanfang_df, os.path.join(base_path, 'Nanfang')),
    }

    all_feature_files = {}
    for name, (df, data_path) in all_dfs.items():
        sub_cache = os.path.join(cache_dir, name)
        all_feature_files[name] = pre_extract_all_features(
            df, data_path, feature_extractor, device, sub_cache
        )

    # Free GPU memory
    del feature_extractor
    torch.cuda.empty_cache()

    # ---- Stage 2: Train MIL ----
    print("\n" + "=" * 50)
    print("STAGE 2: Attention MIL Training")
    print("=" * 50)

    train_dataset = MILDataset(train_df, all_feature_files['tcga'],
                                label_map, bag_size=CONFIG['n_bag_size'], augment=True)
    val_dataset = MILDataset(val_df, all_feature_files['val'],
                              label_map, bag_size=CONFIG['n_bag_size'], augment=False)
    test_dataset = MILDataset(nanfang_df, all_feature_files['nanfang'],
                               label_map, bag_size=CONFIG['n_bag_size'], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                               shuffle=True, num_workers=CONFIG['num_workers'],
                               drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'] * 2,
                             shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'] * 2,
                              shuffle=False, num_workers=CONFIG['num_workers'])

    # Build MIL model
    mil_model = GatedAttentionMIL(
        input_dim=CONFIG['feature_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_classes=2,
    ).to(device)
    print(f"MIL parameters: {sum(p.numel() for p in mil_model.parameters()):,}")

    # Class-balanced loss
    luad_count = (train_df['label'] == 'LUAD').sum()
    lusc_count = (train_df['label'] == 'LUSC').sum()
    class_weights = torch.tensor([1.0 / lusc_count, 1.0 / luad_count], device=device, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 2
    print(f"Class weights: LUAD={class_weights[1]:.3f}, LUSC={class_weights[0]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(mil_model.parameters(), lr=CONFIG['lr'],
                             weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_auc = 0
    patience_counter = 0
    history = []

    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch_mil(mil_model, train_loader,
                                                  optimizer, criterion, device)
        val_metrics = validate_mil(mil_model, val_loader, criterion, device)
        scheduler.step(val_metrics['val_auc'])

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

    # Load best model
    checkpoint = torch.load('best_model.pth', weights_only=True)
    mil_model.load_state_dict(checkpoint['model_state_dict'])

    # ---- Stage 3: Test on Nanfang ----
    print("\n" + "=" * 50)
    print("STAGE 3: Nanfang Test Set Evaluation")
    print("=" * 50)

    metrics, all_logits, all_labels, all_paths, cm, fpr, tpr = test_mil(
        mil_model, test_loader, device
    )

    # Save metrics
    safe_metrics = {}
    for k, v in metrics.items():
        val = float(v)
        safe_metrics[k] = None if math.isnan(val) else val
    with open('metrics.json', 'w') as f:
        json.dump(safe_metrics, f, indent=2)

    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 50)
    print("Task 2 (v2 MIL) completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
