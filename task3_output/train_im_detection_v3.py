"""
Task 3: IM Detection (v3 — Virchow2 features)
==============================================
Uses pre-extracted Virchow2 pathology foundation model features.
"""

import os
import json
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              confusion_matrix, roc_curve, auc)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'features_dir': '/jhcnas7/Pathology/PathLab_data_collection/tmp_test_data/PWH-GC-IM/features/virchow2',
    'data_path': '/jhcnas7/Pathology/PathLab_data_collection/Data/GC_IM_Detection',
    'feature_dim': 2560,
    'hidden_dim': 384,
    'bag_size': 256,
    'att_dropout': 0.3,
    'inst_dropout': 0.2,
    'batch_size': 4,
    'epochs': 40,
    'lr': 1e-4,
    'weight_decay': 5e-4,
    'patience': 15,
    'num_workers': 4,
}


# ============================================================================
# Gated Attention MIL
# ============================================================================
class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=384, num_classes=2, att_dropout=0.3):
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
# Virchow2 Dataset
# ============================================================================
class Virchow2Dataset(Dataset):
    def __init__(self, df, feature_dir, label_map, bag_size=256, augment=False):
        self.df = df.reset_index(drop=True)
        self.feature_dir = feature_dir
        self.label_map = label_map
        self.bag_size = bag_size
        self.augment = augment
        self._build_index()

    def _build_index(self):
        self.file_map = {}
        missing = 0
        for idx, row in self.df.iterrows():
            fpath = _find_feature_file(row['filename'], self.feature_dir)
            if fpath:
                self.file_map[idx] = fpath
            else:
                missing += 1
        if missing:
            print(f"  WARNING: {missing} slides have no Virchow2 features")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.label_map.get(row['label'], 0)
        fpath = self.file_map.get(idx)

        if fpath is None:
            features = torch.zeros(self.bag_size, CONFIG['feature_dim'])
        else:
            features = torch.load(fpath, weights_only=True)
            n_available = features.shape[0]
            if n_available >= self.bag_size:
                indices = torch.randperm(n_available)[:self.bag_size]
            else:
                indices = torch.randint(0, n_available, (self.bag_size,))
            features = features[indices]

        if self.augment:
            features = features + torch.randn_like(features) * 0.01

        return features, label, row['filename']


def _find_feature_file(slide_name, feature_dir):
    stem = os.path.splitext(slide_name)[0]
    # Try exact match
    fpath = os.path.join(feature_dir, stem + '.pt')
    if os.path.exists(fpath):
        return fpath
    # Try with / replaced (some files use different naming)
    fpath = os.path.join(feature_dir, os.path.basename(stem) + '.pt')
    if os.path.exists(fpath):
        return fpath
    return None


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

    for candidate in ['class', 'Class', 'label', 'Label', 'category']:
        if candidate in df.columns:
            df['label'] = df[candidate].map(normalize_text)
            print(f"  Mapped '{candidate}' → 'label'")
            break
    for candidate in ['slide', 'Slide', 'filename', 'file_name', 'id', 'name']:
        if candidate in df.columns:
            df['filename'] = df[candidate]
            print(f"  Mapped '{candidate}' → 'filename'")
            break

    df = df[df['label'].isin(['Intestinal metaplasia', 'Not Intestinal metaplasia'])].reset_index(drop=True)
    print(f"Total PWH samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    return df


# ============================================================================
# Training & Evaluation
# ============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device, inst_dropout=0.0):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
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
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_logits = [], [], []
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
    all_preds, all_labels, all_logits = map(np.array, [all_preds, all_labels, all_logits])
    return {
        'val_loss': total_loss / len(dataloader),
        'val_acc': accuracy_score(all_labels, all_preds),
        'val_auc': roc_auc_score(all_labels, all_logits) if len(np.unique(all_labels)) >= 2 else float('nan'),
    }


@torch.no_grad()
def test_model(model, dataloader, device):
    model.eval()
    all_preds, all_logits, all_labels, all_paths = [], [], [], []
    for features, labels, slide_names in tqdm(dataloader, desc='Testing', leave=False):
        features, labels = features.to(device), labels.to(device)
        logits, _ = model(features)
        probs = torch.softmax(logits, dim=1)
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_logits.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_paths.extend(slide_names)

    all_preds = np.array(all_preds)
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)

    n_unique = len(np.unique(all_labels))
    if n_unique >= 2:
        macro_auc = roc_auc_score(all_labels, all_logits[:, 1])
        fpr, tpr, _ = roc_curve(all_labels, all_logits[:, 1])
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

    pred_df = pd.DataFrame({
        'file_name': all_paths,
        'IM': all_logits[:, 1],
        'nonIM': all_logits[:, 0],
    })
    pred_df.to_csv('IM.csv', index=False)
    print("\nPredictions saved to IM.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NonIM', 'IM'], yticklabels=['NonIM', 'IM'])
    plt.title('Confusion Matrix - IM Detection (Virchow2)')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Confusion matrix saved")

    if fpr is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC (AUC = {roc_auc_val:.4f})')
        plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - IM Detection (Virchow2)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300)
        print("ROC curve saved")

    return metrics, all_logits, all_labels, all_paths, cm, fpr, tpr


# ============================================================================
# Main
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")

    label_map = {'Intestinal metaplasia': 1, 'Not Intestinal metaplasia': 0}

    print("\nLoading PWH dataset...")
    df = load_pwh_data(CONFIG['data_path'])

    train_val_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label'])
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    print("\nLoading Virchow2 features...")
    train_dataset = Virchow2Dataset(train_df, CONFIG['features_dir'], label_map,
                                     bag_size=CONFIG['bag_size'], augment=True)
    val_dataset = Virchow2Dataset(val_df, CONFIG['features_dir'], label_map,
                                   bag_size=CONFIG['bag_size'], augment=False)
    test_dataset = Virchow2Dataset(test_df, CONFIG['features_dir'], label_map,
                                    bag_size=CONFIG['bag_size'], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                               shuffle=True, num_workers=CONFIG['num_workers'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'] * 2,
                             shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'] * 2,
                              shuffle=False, num_workers=CONFIG['num_workers'])

    model = GatedAttentionMIL(
        input_dim=CONFIG['feature_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_classes=2,
        att_dropout=CONFIG['att_dropout'],
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Heavy class imbalance (90/10)
    im_count = (train_df['label'] == 'Intestinal metaplasia').sum()
    non_im_count = (train_df['label'] == 'Not Intestinal metaplasia').sum()
    weight_im = len(train_df) / (2.0 * im_count)
    weight_non_im = len(train_df) / (2.0 * non_im_count)
    class_weights = torch.tensor([weight_non_im, weight_im], device=device, dtype=torch.float32)
    print(f"Class weights: nonIM={weight_non_im:.3f}, IM={weight_im:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_auc = 0
    patience_counter = 0
    history = []

    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion,
                                              device, inst_dropout=CONFIG['inst_dropout'])
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc']:.4f} | "
              f"Val AUC: {val_metrics['val_auc']:.4f} | LR: {lr:.2e}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_metrics['val_loss'], 'val_acc': val_metrics['val_acc'],
            'val_auc': val_metrics['val_auc'],
        })

        if val_metrics['val_auc'] > best_val_auc:
            best_val_auc = val_metrics['val_auc']
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'config': CONFIG, 'history': history},
                       'best_model.pth')
        else:
            patience_counter += 1
        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    ckpt = torch.load('best_model.pth', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])

    print("\n" + "=" * 50)
    print("STAGE: PWH Test Evaluation")
    print("=" * 50)
    metrics, _, _, _, _, _, _ = test_model(model, test_loader, device)

    safe_metrics = {k: None if math.isnan(float(v)) else float(v) for k, v in metrics.items()}
    with open('metrics.json', 'w') as f:
        json.dump(safe_metrics, f, indent=2)
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\nTask 3 (v3 Virchow2) completed!")


if __name__ == '__main__':
    main()
