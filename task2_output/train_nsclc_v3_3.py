"""
Task 2: LUAD vs LUSC Classification (v3.3 — z-score + random-bag test)
=======================================================================
v3.1 baseline + per-slide z-score feature normalization + random-bag test.
Eliminates slide-level staining/scanner bias before MIL attention.
"""

import os, json, random, math
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
    'tcga_features':    '/jhcnas7/Pathology/PathLab_data_collection/tmp_test_data/TCGA-NSCLC/features/virchow2',
    'nanfang_features': '/jhcnas7/Pathology/PathLab_data_collection/tmp_test_data/Nanfang-NSCLC/features/virchow2',
    'data_path':        '/jhcnas7/Pathology/PathLab_data_collection/Data/NSCLC',

    'feature_dim':   2560,
    'head_hidden':   256,
    'num_heads':     4,
    'bag_size':      1024,
    'att_dropout':   0.3,
    'inst_dropout':  0.15,

    'batch_size':       1,
    'accum_steps':      4,
    'epochs':           30,
    'lr':               1e-4,
    'weight_decay':     5e-4,
    'patience':         12,
    'num_workers':      4,

    'test_bags':        16,    # random bags for test averaging
    'zscore_eps':       1e-6,  # epsilon for z-score normalization
}


# ========================================================================
# Multi-Head Gated Attention MIL
# ========================================================================
class MultiHeadAttentionMIL(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=256, num_heads=4,
                 num_classes=2, att_dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(input_dim)
        self.V = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_heads)])
        self.U = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_heads)])
        self.w = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])
        self.att_dropout = nn.Dropout(att_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, features, inst_dropout=0.0):
        B, N, D = features.shape
        if inst_dropout > 0 and self.training:
            mask = torch.rand(B, N, 1, device=features.device) > inst_dropout
            features = features * mask
        features_norm = self.norm(features)
        head_outs = []
        for i in range(self.num_heads):
            V = torch.tanh(self.V[i](features_norm))
            U = torch.sigmoid(self.U[i](features_norm))
            A = self.w[i](V * U)
            A = self.att_dropout(A)
            A = torch.softmax(A, dim=1)
            head_outs.append(torch.sum(A * features_norm, dim=1))
        aggregated = torch.stack(head_outs, dim=0).mean(dim=0)
        return self.classifier(aggregated), head_outs[-1].detach()


# ========================================================================
# Virchow2 Dataset with per-slide z-score normalization
# ========================================================================
class Virchow2Dataset(Dataset):
    """Loads Virchow2 features with per-slide z-score normalization.
    return_all=True → returns all patches (for multi-bag test)."""

    def __init__(self, df, feature_dir, label_map, bag_size=1024,
                 augment=False, return_all=False):
        self.df = df.reset_index(drop=True)
        self.feature_dir = feature_dir
        self.label_map = label_map
        self.bag_size = bag_size
        self.augment = augment
        self.return_all = return_all
        self.eps = CONFIG['zscore_eps']
        self._build_index()

    def _build_index(self):
        self.file_map = {}
        missing = 0
        for idx, row in self.df.iterrows():
            fpath = _find_feature_file(row['filename'], self.feature_dir)
            if fpath:
                self.file_map[idx] = fpath
            else:
                found = False
                stem = os.path.splitext(row['filename'])[0]
                for fname in os.listdir(self.feature_dir):
                    if fname.endswith('.pt') and stem in fname:
                        self.file_map[idx] = os.path.join(self.feature_dir, fname)
                        found = True; break
                if not found:
                    missing += 1
        if missing:
            print(f"  WARNING: {missing}/{len(self.df)} slides missing features")

    def __len__(self):
        return len(self.df)

    def _normalize(self, features):
        """Per-slide z-score normalization."""
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True) + self.eps
        return (features - mean) / std

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.label_map.get(row['label'], 0)
        fpath = self.file_map.get(idx)

        if fpath is None:
            features = torch.zeros(self.bag_size, CONFIG['feature_dim'])
        else:
            features = torch.load(fpath, weights_only=True)
            features = self._normalize(features)

        if self.return_all:
            return features, label, row['filename']

        n = features.shape[0]
        if n >= self.bag_size:
            indices = torch.randperm(n)[:self.bag_size]
        else:
            indices = torch.randint(0, n, (self.bag_size,))
        features = features[indices]
        if self.augment:
            features = features + torch.randn_like(features) * 0.01
        return features, label, row['filename']


def _find_feature_file(slide_name, feature_dir):
    stem = os.path.splitext(slide_name)[0]
    for prefix in ['LUAD__', 'LUSC__']:
        fpath = os.path.join(feature_dir, prefix + stem + '.pt')
        if os.path.exists(fpath):
            return fpath
    fpath = os.path.join(feature_dir, stem + '.pt')
    if os.path.exists(fpath):
        return fpath
    return None


# ========================================================================
# CSV loading
# ========================================================================
def normalize_text(v):
    return str(v).strip() if pd.notna(v) else v

def _norm_csv(df):
    for c in ['class', 'Class', 'label', 'Label', 'category']:
        if c in df.columns:
            df['label'] = df[c].map(normalize_text); break
    for c in ['slide', 'Slide', 'filename', 'file_name', 'id', 'name']:
        if c in df.columns:
            df['filename'] = df[c]; break
    return df

def load_data(base, name):
    csv_name = f'{name}_NSCLC.csv' if name == 'TCGA' else 'Nanfang_lung_NSCLC_VALID.csv'
    df = pd.read_csv(os.path.join(base, name, csv_name))
    print(f"CSV: {list(df.columns)}")
    df = _norm_csv(df)
    df = df[df['label'].isin(['LUAD', 'LUSC'])].reset_index(drop=True)
    print(f"{name}: {len(df)}  |  {dict(df['label'].value_counts())}")
    return df


# ========================================================================
# Training with gradient accumulation
# ========================================================================
def train_epoch(model, dataloader, optimizer, criterion, device,
                inst_dropout=0.0, accum_steps=4):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    optimizer.zero_grad()
    step = 0

    for step, (features, labels, _) in enumerate(tqdm(dataloader, desc='Train', leave=False)):
        features, labels = features.to(device), labels.to(device)
        logits, _ = model(features, inst_dropout=inst_dropout)
        loss = criterion(logits, labels) / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        if (step + 1) % accum_steps == 0:
            optimizer.step(); optimizer.zero_grad()
    if step > 0 and (step + 1) % accum_steps != 0:
        optimizer.step(); optimizer.zero_grad()

    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_logits = [], [], []
    for features, labels, _ in tqdm(dataloader, desc='Val', leave=False):
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
def test_model(model, dataloader, device, test_bags=16):
    """Test with random bag sampling & averaging (matches training distribution)."""
    model.eval()
    all_preds, all_logits, all_labels, all_paths = [], [], [], []
    bag_size = CONFIG['bag_size']

    for full_features, labels, slide_names in tqdm(dataloader, desc='Test', leave=False):
        full = full_features.to(device)
        labels = labels.to(device)
        B, N, D = full.shape

        bag_logits = []
        for _ in range(test_bags):
            if N >= bag_size:
                idx = torch.randperm(N, device=device)[:bag_size]
            else:
                idx = torch.randint(0, N, (bag_size,), device=device)
            bag = full[:, idx, :]  # use fancy indexing
            logits, _ = model(bag)
            bag_logits.append(logits)
        logits = torch.stack(bag_logits).mean(dim=0)

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
        macro_auc = roc_auc_val = float('nan')
        fpr = tpr = None

    metrics = {'Macro-AUC': macro_auc, 'Weighted-F1':
               f1_score(all_labels, all_preds, average='weighted'),
               'Macro-ACC': accuracy_score(all_labels, all_preds),
               'ROC-AUC': roc_auc_val}

    print(f"\n{'='*50}\nNanfang Test (z-score norm, {test_bags}-bag avg)\n{'='*50}")
    for m, v in metrics.items():
        print(f"  {m}: {v:.4f}" if not math.isnan(v) else f"  {m}: N/A")

    pd.DataFrame({
        'file_name': all_paths,
        'logits_LUAD': all_logits[:, 1],
        'logits_LUSC': all_logits[:, 0],
    }).to_csv('NSCLC_prediction.csv', index=False)
    print("→ NSCLC_prediction.csv")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['LUSC', 'LUAD'], yticklabels=['LUSC', 'LUAD'])
    plt.title('Confusion Matrix - NSCLC (v3.3 z-score)')
    plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=300)

    if fpr is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'ROC (AUC={roc_auc_val:.4f})')
        plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.legend(loc="lower right")
        plt.tight_layout(); plt.savefig('roc_curve.png', dpi=300)

    return metrics


# ========================================================================
# Main
# ========================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")

    label_map = {'LUAD': 1, 'LUSC': 0}

    print("\n=== Loading data ===")
    tcga_df = load_data(CONFIG['data_path'], 'TCGA')
    nanfang_df = load_data(CONFIG['data_path'], 'Nanfang')
    train_df, val_df = train_test_split(tcga_df, test_size=0.2, random_state=42, stratify=tcga_df['label'])
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(nanfang_df)}")

    print("\n=== Loading Virchow2 features (with z-score norm) ===")
    train_ds = Virchow2Dataset(train_df, CONFIG['tcga_features'], label_map,
                                bag_size=CONFIG['bag_size'], augment=True)
    val_ds   = Virchow2Dataset(val_df,  CONFIG['tcga_features'], label_map,
                                bag_size=CONFIG['bag_size'], augment=False)
    test_ds  = Virchow2Dataset(nanfang_df, CONFIG['nanfang_features'], label_map,
                                return_all=True)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                               shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader   = DataLoader(val_ds,  batch_size=1, shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = MultiHeadAttentionMIL(
        input_dim=CONFIG['feature_dim'], hidden_dim=CONFIG['head_hidden'],
        num_heads=CONFIG['num_heads'], num_classes=2, att_dropout=CONFIG['att_dropout'],
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    luad_n = (train_df['label'] == 'LUAD').sum()
    lusc_n = (train_df['label'] == 'LUSC').sum()
    w = torch.tensor([1.0/lusc_n, 1.0/luad_n], device=device, dtype=torch.float32)
    w = w / w.sum() * 2
    print(f"Class weights: LUAD={w[1]:.3f}, LUSC={w[0]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-6)

    best_auc = 0; patience = 0; history = []

    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            inst_dropout=CONFIG['inst_dropout'], accum_steps=CONFIG['accum_steps'])
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch+1:2d} | Loss {train_loss:.4f} Acc {train_acc:.3f} | "
              f"V-Loss {val_m['val_loss']:.4f} V-Acc {val_m['val_acc']:.3f} "
              f"V-AUC {val_m['val_auc']:.4f} | LR {lr:.2e}")

        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc,
                        'val_loss': val_m['val_loss'], 'val_acc': val_m['val_acc'],
                        'val_auc': val_m['val_auc']})

        if val_m['val_auc'] > best_auc:
            best_auc = val_m['val_auc']; patience = 0
            torch.save({'model': model.state_dict(), 'history': history}, 'best_model.pth')
        else:
            patience += 1
        if patience >= CONFIG['patience']:
            print(f"Early stop epoch {epoch+1}"); break

    ckpt = torch.load('best_model.pth', weights_only=True)
    model.load_state_dict(ckpt['model'])

    print(f"\n{'='*50}\nNanfang Test ({CONFIG['test_bags']}-bag avg)\n{'='*50}")
    metrics = test_model(model, test_loader, device, test_bags=CONFIG['test_bags'])

    safe = {k: None if math.isnan(float(v)) else float(v) for k, v in metrics.items()}
    json.dump(safe, open('metrics.json', 'w'), indent=2)
    json.dump(history, open('training_history.json', 'w'), indent=2)
    print("\nTask2 v3.3 (z-score) done!")


if __name__ == '__main__':
    main()
