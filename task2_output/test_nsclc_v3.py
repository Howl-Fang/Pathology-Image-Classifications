"""
Task 2: Inference-only script — loads trained model, evaluates on Nanfang test set.
Usage: uv run python test_nsclc_v3.py
"""

import os, json, math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              confusion_matrix, roc_curve, auc)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

FEATURE_DIM = 2560
HEAD_HIDDEN = 256
NUM_HEADS = 4
BAG_SIZE = 1024
FEATURES_DIR = '/jhcnas7/Pathology/PathLab_data_collection/tmp_test_data/Nanfang-NSCLC/features/virchow2'
DATA_PATH = '/jhcnas7/Pathology/PathLab_data_collection/Data/NSCLC'
CKPT_PATH = 'best_model.pth'


# ========================================================================
# Model (must match training architecture exactly)
# ========================================================================
class MultiHeadAttentionMIL(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=256, num_heads=4, num_classes=2):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(input_dim)
        self.V = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_heads)])
        self.U = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_heads)])
        self.w = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(256, num_classes),
        )

    def forward(self, features):
        B, N, D = features.shape
        features_norm = self.norm(features)
        head_outs = []
        for i in range(self.num_heads):
            V = torch.tanh(self.V[i](features_norm))
            U = torch.sigmoid(self.U[i](features_norm))
            A = self.w[i](V * U)
            A = torch.softmax(A, dim=1)
            head_outs.append(torch.sum(A * features_norm, dim=1))
        aggregated = torch.stack(head_outs, dim=0).mean(dim=0)
        return self.classifier(aggregated)


# ========================================================================
# Dataset & loader
# ========================================================================
class TestDataset(Dataset):
    """Loads all patches per slide (no sampling)."""

    def __init__(self, df, feature_dir):
        self.df = df
        self.feature_dir = feature_dir
        self._build_index()

    def _build_index(self):
        self.paths = []
        missing = 0
        for _, row in self.df.iterrows():
            stem = os.path.splitext(row['filename'])[0]
            found = None
            for prefix in ['LUAD__', 'LUSC__', '']:
                fpath = os.path.join(self.feature_dir, prefix + stem + '.pt')
                if os.path.exists(fpath):
                    found = fpath; break
            if not found:
                # partial match
                for fname in os.listdir(self.feature_dir):
                    if fname.endswith('.pt') and stem in fname:
                        found = os.path.join(self.feature_dir, fname); break
            if found:
                self.paths.append(found)
            else:
                self.paths.append(None)
                missing += 1
        if missing:
            print(f"WARNING: {missing} slides have no feature file")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fpath = self.paths[idx]
        if fpath is None:
            return torch.zeros(1, FEATURE_DIM), -1, self.df.iloc[idx]['filename']
        features = torch.load(fpath, weights_only=True)
        return features, -1, self.df.iloc[idx]['filename']


def load_nanfang_data():
    df = pd.read_csv(os.path.join(DATA_PATH, 'Nanfang', 'Nanfang_lung_NSCLC_VALID.csv'))
    for c in ['class', 'Class', 'label', 'Label']:
        if c in df.columns:
            df['label'] = df[c].astype(str).str.strip(); break
    for c in ['slide', 'Slide', 'filename', 'file_name']:
        if c in df.columns:
            df['filename'] = df[c]; break
    df = df[df['label'].isin(['LUAD', 'LUSC'])].reset_index(drop=True)
    print(f"Nanfang: {len(df)} slides  |  {dict(df['label'].value_counts())}")
    return df


# ========================================================================
# Inference
# ========================================================================
@torch.no_grad()
def infer(model, dataloader, device):
    model.eval()
    all_preds, all_logits, all_labels, all_paths = [], [], [], []

    for full_features, labels, slide_names in tqdm(dataloader, desc='Inference'):
        full = full_features.to(device)
        labels = labels.to(device)
        N = full.shape[1]

        chunk_logits = []
        for start in range(0, N, BAG_SIZE):
            chunk = full[:, start:start + BAG_SIZE, :]
            if chunk.shape[1] < BAG_SIZE:
                extra_idx = torch.randint(0, N, (BAG_SIZE - chunk.shape[1],), device=device)
                extra = full[:, extra_idx, :]
                chunk = torch.cat([chunk, extra], dim=1)
            chunk_logits.append(model(chunk))

        logits = torch.stack(chunk_logits).mean(dim=0)
        probs = torch.softmax(logits, dim=1)
        _, preds = torch.max(logits, 1)

        all_preds.extend(preds.cpu().numpy())
        all_logits.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_paths.extend(slide_names)

    return (np.array(all_preds), np.array(all_logits),
            np.array(all_labels), np.array(all_paths))


# ========================================================================
# Main
# ========================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(CKPT_PATH, weights_only=True, map_location=device)
    model = MultiHeadAttentionMIL(
        input_dim=FEATURE_DIM, hidden_dim=HEAD_HIDDEN,
        num_heads=NUM_HEADS, num_classes=2,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded from {CKPT_PATH}")
    if 'history' in ckpt:
        best_auc = max(e['val_auc'] for e in ckpt['history'])
        print(f"  Best Val AUC during training: {best_auc:.4f}")

    # Load data
    df = load_nanfang_data()
    ds = TestDataset(df, FEATURES_DIR)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Infer
    preds, logits, labels, paths = infer(model, dl, device)

    # Metrics
    label_map = {'LUAD': 1, 'LUSC': 0}
    true_labels = np.array([label_map.get(df.iloc[i]['label'], 0) for i in range(len(df))])

    n_unique = len(np.unique(true_labels))
    macro_auc = roc_auc_score(true_labels, logits[:, 1]) if n_unique >= 2 else float('nan')
    fpr, tpr, _ = (roc_curve(true_labels, logits[:, 1]) if n_unique >= 2 else (None, None, None))
    roc_auc_val = auc(fpr, tpr) if fpr is not None else float('nan')
    weighted_f1 = f1_score(true_labels, preds, average='weighted')
    macro_acc = accuracy_score(true_labels, preds)
    cm = confusion_matrix(true_labels, preds)

    metrics = {'Macro-AUC': macro_auc, 'Weighted-F1': weighted_f1,
               'Macro-ACC': macro_acc, 'ROC-AUC': roc_auc_val}

    print("\n" + "=" * 50)
    print("Nanfang Test Results (full-patch inference)")
    print("=" * 50)
    for m, v in metrics.items():
        print(f"  {m}: {v:.4f}" if not (isinstance(v, float) and math.isnan(v)) else f"  {m}: N/A")

    # Save predictions
    pd.DataFrame({
        'file_name': paths,
        'logits_LUAD': logits[:, 1],
        'logits_LUSC': logits[:, 0],
    }).to_csv('NSCLC_prediction.csv', index=False)
    print("\n→ NSCLC_prediction.csv")

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['LUSC', 'LUAD'], yticklabels=['LUSC', 'LUAD'])
    plt.title('Confusion Matrix - NSCLC (Virchow2 MIL)')
    plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=300)

    # ROC
    if fpr is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'ROC (AUC={roc_auc_val:.4f})')
        plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.title('ROC - NSCLC')
        plt.legend(loc="lower right")
        plt.tight_layout(); plt.savefig('roc_curve.png', dpi=300)

    json.dump({k: None if (isinstance(v, float) and math.isnan(v)) else v
               for k, v in metrics.items()}, open('metrics.json', 'w'), indent=2)
    print("Done.")


if __name__ == '__main__':
    main()
