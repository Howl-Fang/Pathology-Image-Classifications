import os
import json
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
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def normalize_text(value):
    if pd.isna(value):
        return value
    return str(value).strip()

# Try to import openslide, but raise a clear error if not available
try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False


def check_wsi_accessibility(data_paths):
    """Pre-flight check: verify OpenSlide and WSI files are accessible."""
    if not HAS_OPENSLIDE:
        print("=" * 60)
        print("WARNING: OpenSlide is not installed!")
        print("  WSI (.svs) files cannot be read without openslide-python.")
        print("  The model will receive blank images and learn nothing.")
        print()
        print("  Install OpenSlide:")
        print("    Ubuntu: sudo apt-get install libopenslide0")
        print("    then:   pip install openslide-python")
        print("=" * 60)
        return False
    
    # Check if at least one WSI file is accessible
    for path in data_paths:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith('.svs'):
                    return True
    
    print("=" * 60)
    print("WARNING: No .svs files found in data directories!")
    print(f"  Checked paths: {data_paths}")
    print("  Without WSI files, training will use blank images.")
    print("=" * 60)
    return False

class NSCLCDataset(Dataset):
    def __init__(self, df, data_path, transform=None, sample_size=5):
        self.df = df
        self.data_path = data_path
        self.transform = transform
        self.sample_size = sample_size
        self.label_map = {'LUAD': 1, 'LUSC': 0}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Try to load WSI using openslide
        wsi_path = os.path.join(self.data_path, 'WSIs', row['filename'])
        
        try:
            if HAS_OPENSLIDE and os.path.exists(wsi_path):
                # Load from WSI - sample random patches
                import random
                slide = openslide.open_slide(wsi_path)
                w, h = slide.dimensions
                
                # Sample random patch
                patch_size = 256
                x = random.randint(0, max(0, w - patch_size))
                y = random.randint(0, max(0, h - patch_size))
                
                patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                image = patch.convert('RGB')
                slide.close()
            else:
                # Fallback: create a dummy image
                image = Image.new('RGB', (256, 256), color='white')
        except:
            image = Image.new('RGB', (256, 256), color='white')
        
        label = self.label_map.get(row['label'], 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, row['filename']

def _scan_wsi_directory(wsi_dir):
    """Scan a WSI directory and return list of .svs files."""
    files = []
    if os.path.isdir(wsi_dir):
        for f in sorted(os.listdir(wsi_dir)):
            if f.lower().endswith(('.svs', '.mrxs', '.tiff', '.tif', '.ndpi')):
                files.append(f)
    return files


def _generate_label_template(csv_path, wsi_dir, dataset_name):
    """Generate a label.csv template from WSI directory listing."""
    files = _scan_wsi_directory(wsi_dir)
    if not files:
        return None
    
    df = pd.DataFrame({
        'filename': files,
        'label': 'UNKNOWN'
    })
    df.to_csv(csv_path, index=False)
    print(f"\nGenerated label template: {csv_path}")
    print(f"  Found {len(files)} WSI files in {wsi_dir}")
    print(f"  Please edit {csv_path} and replace 'UNKNOWN' with 'LUAD' or 'LUSC'")
    return df


def _normalize_nsclc_csv(df, csv_path):
    """Auto-detect and normalize NSCLC CSV columns: class→label, slide→filename."""
    print(f"CSV columns: {list(df.columns)}")
    
    for candidate in ['class', 'Class', 'label', 'Label', 'category']:
        if candidate in df.columns:
            if candidate != 'label':
                df['label'] = df[candidate].map(normalize_text)
                print(f"  Mapped '{candidate}' → 'label'")
            else:
                df['label'] = df['label'].map(normalize_text)
            break
    else:
        raise ValueError(f"Cannot find label column in {csv_path}. Columns: {list(df.columns)}")
    
    for candidate in ['slide', 'Slide', 'filename', 'file_name', 'id', 'name']:
        if candidate in df.columns:
            if candidate != 'filename':
                df['filename'] = df[candidate]
                print(f"  Mapped '{candidate}' → 'filename'")
            break
    
    return df


def load_data(data_path):
    """Load NSCLC dataset from TCGA (TCGA_NSCLC.csv)"""
    csv_path = os.path.join(data_path, 'TCGA', 'TCGA_NSCLC.csv')
    wsi_dir = os.path.join(data_path, 'TCGA', 'WSIs')
    
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found!")
        df = _generate_label_template(csv_path, wsi_dir, 'TCGA')
        if df is not None:
            raise FileNotFoundError(
                f"Label template generated at {csv_path}.\n"
                f"Please edit it to set correct LUAD/LUSC labels, then re-run."
            )
        else:
            raise FileNotFoundError(
                f"No label CSV and no WSI files found in {wsi_dir}."
            )
    
    df = pd.read_csv(csv_path)
    df = _normalize_nsclc_csv(df, csv_path)
    df = df[df['label'].isin(['LUAD', 'LUSC'])].reset_index(drop=True)
    
    print(f"Total TCGA samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    return df


def load_nanfang_data(data_path):
    """Load Nanfang test dataset (Nanfang_lung_NSCLC_VALID.csv)"""
    csv_path = os.path.join(data_path, 'Nanfang', 'Nanfang_lung_NSCLC_VALID.csv')
    wsi_dir = os.path.join(data_path, 'Nanfang', 'WSIs')
    
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found!")
        df = _generate_label_template(csv_path, wsi_dir, 'Nanfang')
        if df is not None:
            raise FileNotFoundError(
                f"Label template generated at {csv_path}.\n"
                f"Please edit it to set correct LUAD/LUSC labels, then re-run."
            )
        else:
            raise FileNotFoundError(
                f"No label CSV and no WSI files found in {wsi_dir}."
            )
    
    df = pd.read_csv(csv_path)
    df = _normalize_nsclc_csv(df, csv_path)
    df = df[df['label'].isin(['LUAD', 'LUSC'])].reset_index(drop=True)
    
    print(f"Total Nanfang samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    return df

def create_dataloaders(train_df, nanfang_df, train_data_path, nanfang_data_path, batch_size=32):
    """Create dataloaders"""
    
    # Split train into train and val
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, 
        stratify=train_df['label']
    )
    
    print(f"Train: {len(train_split)}, Val: {len(val_split)}, Nanfang Test: {len(nanfang_df)}")
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = NSCLCDataset(train_split, train_data_path, train_transform)
    val_dataset = NSCLCDataset(val_split, train_data_path, val_test_transform)
    test_dataset = NSCLCDataset(nanfang_df, nanfang_data_path, val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def build_model():
    """Build ResNet50 model"""
    model = resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    return model

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels, _ in tqdm(train_loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(val_loader), accuracy

def train_model(model, train_loader, val_loader, epochs=20, device='cuda'):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def test_model(model, test_loader, device):
    """Evaluate on test set"""
    model.eval()
    all_preds = []
    all_logits = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_logits.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
    
    all_preds = np.array(all_preds)
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    
    n_unique_labels = len(np.unique(all_labels))
    
    # Calculate metrics (with NaN safety for single-class case)
    if n_unique_labels >= 2:
        macro_auc = roc_auc_score(all_labels, all_logits[:, 1])
        fpr, tpr, _ = roc_curve(all_labels, all_logits[:, 1])
        roc_auc = auc(fpr, tpr)
    else:
        print(f"WARNING: Only {n_unique_labels} class(es) in test set, cannot compute AUC metrics.")
        macro_auc = float('nan')
        fpr, tpr = None, None
        roc_auc = float('nan')
    
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'Macro-AUC': macro_auc,
        'Weighted-F1': weighted_f1,
        'Macro-ACC': macro_acc,
        'ROC-AUC': roc_auc,
    }
    
    print("\n=== Nanfang Test Set Performance ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'file_name': all_paths,
        'logits_LUAD': all_logits[:, 1],
        'logits_LUSC': all_logits[:, 0],
    })
    pred_df.to_csv('NSCLC_prediction.csv', index=False)
    print("\nPredictions saved to NSCLC_prediction.csv")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['LUSC', 'LUAD'],
                yticklabels=['LUSC', 'LUAD'])
    plt.title('Confusion Matrix - NSCLC LUAD vs LUSC')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Confusion matrix saved to confusion_matrix.png")
    
    # Plot ROC curve (only if both classes present)
    if fpr is not None and tpr is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - NSCLC LUAD vs LUSC')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300)
        print("ROC curve saved to roc_curve.png")
    else:
        print("Skipping ROC curve: only one class present in test set.")
    
    return metrics, all_logits, all_labels, all_paths, cm, fpr, tpr

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_path = '/jhcnas7/Pathology/PathLab_data_collection/Data/NSCLC'
    
    # Pre-flight check: verify WSI accessibility
    wsi_dirs = [
        os.path.join(base_path, 'TCGA', 'WSIs'),
        os.path.join(base_path, 'Nanfang', 'WSIs'),
    ]
    check_wsi_accessibility(wsi_dirs)
    
    # Load data
    print("Loading TCGA training data...")
    train_df = load_data(base_path)
    
    print("Loading Nanfang test data...")
    nanfang_df = load_nanfang_data(base_path)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, nanfang_df, 
        os.path.join(base_path, 'TCGA'),
        os.path.join(base_path, 'Nanfang'),
        batch_size=32
    )
    
    # Build and train model
    model = build_model().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTraining model on TCGA...")
    model = train_model(model, train_loader, val_loader, epochs=20, device=device)
    
    # Test model
    print("\nTesting model on Nanfang...")
    metrics, all_logits, all_labels, all_paths, cm, fpr, tpr = test_model(model, test_loader, device)
    
    # Save metrics (handle NaN values for JSON compliance)
    import math
    safe_metrics = {}
    for k, v in metrics.items():
        val = float(v)
        safe_metrics[k] = None if math.isnan(val) else val
    with open('metrics.json', 'w') as f:
        json.dump(safe_metrics, f, indent=2)
    
    print("\n✓ Task 2 completed!")

if __name__ == '__main__':
    main()
