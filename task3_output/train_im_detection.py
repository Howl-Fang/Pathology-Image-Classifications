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

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False


def check_wsi_accessibility(data_path):
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
    
    wsi_dir = os.path.join(data_path, 'PWH', 'WSIs')
    if os.path.isdir(wsi_dir):
        for f in os.listdir(wsi_dir):
            if f.endswith('.svs'):
                return True
    
    print("=" * 60)
    print(f"WARNING: No .svs files found in {wsi_dir}")
    print("  Without WSI files, training will use blank images.")
    print("=" * 60)
    return False

class IMDetectionDataset(Dataset):
    def __init__(self, df, data_path, transform=None):
        self.df = df
        self.data_path = data_path
        self.transform = transform
        # Map labels
        self.label_map = {
            'Intestinal metaplasia': 1,
            'Not Intestinal metaplasia': 0,
            'IM': 1,
            'nonIM': 0
        }
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Handle different label column names
        if 'label' in row:
            label_str = row['label']
        elif 'Label' in row:
            label_str = row['Label']
        else:
            label_str = 'nonIM'
        
        # Get filename - try multiple common column names
        filename = ''
        for col in ['filename', 'file_name', 'slide_id', 'slide', 'id', 'name']:
            if col in row.index and pd.notna(row[col]):
                filename = str(row[col])
                break
        
        if not filename:
            # Last resort: use index as identifier
            filename = f'sample_{idx}'
        
        wsi_path = os.path.join(self.data_path, 'WSIs', filename)
        
        try:
            if HAS_OPENSLIDE and os.path.exists(wsi_path):
                import random
                slide = openslide.open_slide(wsi_path)
                w, h = slide.dimensions
                
                patch_size = 256
                x = random.randint(0, max(0, w - patch_size))
                y = random.randint(0, max(0, h - patch_size))
                
                patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                image = patch.convert('RGB')
                slide.close()
            else:
                image = Image.new('RGB', (256, 256), color='white')
        except:
            image = Image.new('RGB', (256, 256), color='white')
        
        label = self.label_map.get(label_str, 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, filename

def load_pwh_data(data_path):
    """Load PWH dataset with automatic column detection"""
    csv_path = os.path.join(data_path, 'PWH', 'label.csv')
    df = pd.read_csv(csv_path)
    
    # Print column names for diagnostics
    print(f"CSV columns: {list(df.columns)}")
    
    # Auto-detect and normalize label column
    label_col = None
    for candidate in ['label', 'Label', 'class', 'Class', 'category']:
        if candidate in df.columns:
            label_col = candidate
            break
    
    if label_col is None:
        raise ValueError(f"Cannot find label column in CSV. Columns: {list(df.columns)}")
    
    # Rename to 'label' for consistency
    if label_col != 'label':
        df['label'] = df[label_col].map(normalize_text)
        print(f"Detected label column: '{label_col}' → mapped to 'label'")
    else:
        df['label'] = df['label'].map(normalize_text)
    
    # Filter valid classes
    df = df[df['label'].isin(['Intestinal metaplasia', 'Not Intestinal metaplasia'])].reset_index(drop=True)
    
    # Auto-detect and normalize filename/slide column
    file_col = None
    for candidate in ['filename', 'file_name', 'slide', 'Slide', 'slide_id', 'id', 'name']:
        if candidate in df.columns:
            file_col = candidate
            break
    
    if file_col and file_col != 'filename':
        df['filename'] = df[file_col]
        print(f"Detected filename column: '{file_col}' → mapped to 'filename'")
    
    print(f"Total PWH samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    return df

def create_dataloaders(df, data_path, batch_size=32):
    """Create dataloaders with stratified sampling"""
    
    # Detect label column for stratification
    label_col = 'label' if 'label' in df.columns else None
    
    # Split into train, val, test with stratification
    train_split, test_split = train_test_split(
        df, test_size=0.3, random_state=42,
        stratify=df[label_col] if label_col else None
    )
    
    train_split, val_split = train_test_split(
        train_split, test_size=0.2, random_state=42,
        stratify=train_split[label_col] if label_col else None
    )
    
    print(f"Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")
    if label_col:
        print(f"Train class dist:\n{train_split[label_col].value_counts()}")
        print(f"Val class dist:\n{val_split[label_col].value_counts()}")
        print(f"Test class dist:\n{test_split[label_col].value_counts()}")
    
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
    
    train_dataset = IMDetectionDataset(train_split, data_path, train_transform)
    val_dataset = IMDetectionDataset(val_split, data_path, val_test_transform)
    test_dataset = IMDetectionDataset(test_split, data_path, val_test_transform)
    
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

def train_model(model, train_loader, val_loader, epochs=20, device='cuda', class_weights=None):
    """Train the model with optional class weights for imbalance"""
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"Using class weights: {class_weights.tolist()}")
    else:
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
    
    print("\n=== PWH Test Set Performance ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'file_name': all_paths,
        'IM': all_logits[:, 1],
        'nonIM': all_logits[:, 0],
    })
    pred_df.to_csv('IM.csv', index=False)
    print("\nPredictions saved to IM.csv")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NonIM', 'IM'],
                yticklabels=['NonIM', 'IM'])
    plt.title('Confusion Matrix - Intestinal Metaplasia Detection')
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
        plt.title('ROC Curve - Intestinal Metaplasia Detection')
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
    
    data_path = '/jhcnas7/Pathology/PathLab_data_collection/Data/GC_IM_Detection'
    
    # Pre-flight: check WSI accessibility
    check_wsi_accessibility(data_path)
    
    # Load data
    print("Loading PWH dataset...")
    df = load_pwh_data(data_path)
    
    if len(df) == 0:
        print("Error: No valid samples found!")
        return
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(df, data_path, batch_size=32)
    
    # Compute class weights for imbalanced dataset
    label_col = 'label' if 'label' in df.columns else None
    if label_col:
        class_counts = df[label_col].value_counts().sort_index()
        # Inverse frequency weighting
        weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float32)
        weights = weights / weights.sum() * len(class_counts)  # normalize
        print(f"Computed class weights: {weights.tolist()}")
    else:
        weights = None
    
    # Build and train model
    model = build_model().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTraining model on PWH...")
    model = train_model(model, train_loader, val_loader, epochs=20, device=device, class_weights=weights)
    
    # Test model
    print("\nTesting model...")
    metrics, all_logits, all_labels, all_paths, cm, fpr, tpr = test_model(model, test_loader, device)
    
    # Save metrics (handle NaN values for JSON compliance)
    import math
    safe_metrics = {}
    for k, v in metrics.items():
        val = float(v)
        safe_metrics[k] = None if math.isnan(val) else val
    with open('metrics.json', 'w') as f:
        json.dump(safe_metrics, f, indent=2)
    
    print("\n✓ Task 3 completed!")

if __name__ == '__main__':
    main()
