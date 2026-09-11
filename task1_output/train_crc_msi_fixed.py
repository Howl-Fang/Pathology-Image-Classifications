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

class CRCMSIDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {'MSIH': 1, 'NonMSIH': 0}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        label = self.label_map[row['class']]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, row['image_path']

def load_data(data_path):
    """Load CRC-MSI dataset"""
    csv_path = os.path.join(data_path, 'label.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'class' in df.columns:
            df['class'] = df['class'].map(normalize_text)
        if 'split' in df.columns:
            df['split'] = df['split'].map(normalize_text).str.lower()
    else:
        records = []
        split_candidates = {
            'TRAIN': 'train',
            'train': 'train',
            'TEST': 'test',
            'test': 'test',
        }
        class_map = {
            'msih': 'MSIH',
            'nonmsih': 'NonMSIH',
            'non-msih': 'NonMSIH',
            'non_msih': 'NonMSIH',
            'nonmsi': 'NonMSIH',
        }

        for split_dir, split_value in split_candidates.items():
            split_path = os.path.join(data_path, split_dir)
            if not os.path.isdir(split_path):
                continue
            for class_dir in os.listdir(split_path):
                class_key = normalize_text(class_dir).lower()
                class_value = class_map.get(class_key)
                if class_value is None:
                    continue
                class_path = os.path.join(split_path, class_dir)
                if not os.path.isdir(class_path):
                    continue
                for file_name in os.listdir(class_path):
                    lower_name = file_name.lower()
                    if not (lower_name.endswith('.jpg') or lower_name.endswith('.jpeg') or lower_name.endswith('.png')):
                        continue
                    rel_path = os.path.join(split_dir, class_dir, file_name)
                    records.append({
                        'image_path': rel_path,
                        'class': class_value,
                        'split': split_value,
                    })

        if not records:
            raise FileNotFoundError(
                f"No label.csv and no discoverable image layout under {data_path}."
            )
        df = pd.DataFrame(records)
    
    # Diagnostic: print unique class values BEFORE filtering
    unique_classes_before = sorted(df['class'].dropna().unique())
    print(f"Unique classes found in data (before filter): {unique_classes_before}")
    
    # Filter to only MSIH and NonMSIH
    df = df[df['class'].isin(['MSIH', 'NonMSIH'])].reset_index(drop=True)
    
    print(f"Total samples after filtering: {len(df)}")
    print(f"Class distribution:\n{df['class'].value_counts()}")

    if df['class'].nunique() < 2:
        print("\n" + "=" * 60)
        print("ERROR: Only one class found in the dataset!")
        print(f"  Found classes: {list(df['class'].unique())}")
        print(f"  Classes in label.csv (before filter): {unique_classes_before}")
        print("  Expected classes: MSIH, NonMSIH")
        print()
        print("  Possible causes:")
        print("  1. label.csv may use different naming (e.g., 'MSS', 'MSI-L', 'non-msih')")
        print("  2. The data directory may be missing NonMSIH samples")
        print("  3. The CSV 'class' column may have formatting issues")
        print()
        print("  Suggested fix: check the label.csv file and ensure both")
        print("  'MSIH' and 'NonMSIH' labels are present.")
        print("=" * 60)
        raise ValueError(
            f"CRC-MSI labels collapsed to a single class: {list(df['class'].unique())}. "
            f"CSV contains: {unique_classes_before}"
        )
    
    return df, data_path

def create_dataloaders(df, data_path, batch_size=32, val_split=0.2):
    """Create train and validation dataloaders"""
    
    # Split into train and val (from training split only)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    train_split, val_split = train_test_split(
        train_df, test_size=val_split, random_state=42, 
        stratify=train_df['class']
    )
    
    print(f"Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_df)}")
    
    # Data augmentation
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
    
    train_dataset = CRCMSIDataset(train_split, data_path, train_transform)
    val_dataset = CRCMSIDataset(val_split, data_path, val_test_transform)
    test_dataset = CRCMSIDataset(test_df, data_path, val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader, test_df

def build_model():
    """Build ResNet50 model"""
    model = resnet50(pretrained=True)
    # Modify final layer for binary classification
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
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def test_model(model, test_loader, test_df, device):
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
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    
    n_unique_labels = len(np.unique(all_labels))
    
    # Macro-AUC (requires both classes to be present)
    if n_unique_labels >= 2:
        macro_auc = roc_auc_score(all_labels, all_logits[:, 1])
        fpr, tpr, _ = roc_curve(all_labels, all_logits[:, 1])
        roc_auc = auc(fpr, tpr)
    else:
        print(f"WARNING: Only {n_unique_labels} class(es) in test set, cannot compute AUC metrics.")
        macro_auc = float('nan')
        fpr, tpr = None, None
        roc_auc = float('nan')
    
    # Weighted F1
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Macro Accuracy
    macro_acc = accuracy_score(all_labels, all_preds)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'Macro-AUC': macro_auc,
        'Weighted-F1': weighted_f1,
        'Macro-ACC': macro_acc,
        'ROC-AUC': roc_auc,
    }
    
    print("\n=== Test Set Performance ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'file_name': all_paths,
        'logits_MSIH': all_logits[:, 1],
        'logits_nonMSIH': all_logits[:, 0],
        'label': all_labels
    })
    pred_df.to_csv('CRC-MSI_prediction.csv', index=False)
    print("\nPredictions saved to CRC-MSI_prediction.csv")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NonMSIH', 'MSIH'],
                yticklabels=['NonMSIH', 'MSIH'])
    plt.title('Confusion Matrix - CRC-MSI Classification')
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
        plt.title('ROC Curve - CRC-MSI Classification')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300)
        print("ROC curve saved to roc_curve.png")
    else:
        print("Skipping ROC curve: only one class present in test set.")
    
    return metrics, all_logits, all_labels, all_paths, cm, fpr, tpr

def generate_heatmaps(model, image_paths, data_path, device):
    """Generate activation heatmaps for specific images"""
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    model.layer4.register_forward_hook(hook_fn('layer4'))
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    for target_img in image_paths:
        csv_path = os.path.join(data_path, 'label.csv')
        df = pd.read_csv(csv_path)
        
        # Try to find matching image
        for idx, row in df.iterrows():
            if target_img.split('/')[-1].split('_')[0] in row['image_path']:
                img_path = os.path.join(data_path, row['image_path'])
                
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(image_tensor)
                    
                    acts = activations['layer4']
                    acts = acts.mean(dim=1).squeeze(0)
                    acts = torch.nn.functional.interpolate(
                        acts.unsqueeze(0).unsqueeze(0), 
                        size=(256, 256), 
                        mode='bilinear'
                    ).squeeze()
                    
                    acts = (acts - acts.min()) / (acts.max() - acts.min() + 1e-8)
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    axes[0].imshow(np.array(image))
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')
                    
                    heatmap = axes[1].imshow(acts.cpu().numpy(), cmap='hot')
                    axes[1].set_title('Activation Heatmap')
                    axes[1].axis('off')
                    plt.colorbar(heatmap, ax=axes[1])
                    
                    filename = target_img.split('/')[-1].replace('.jpg', '_heatmap.png')
                    plt.tight_layout()
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Heatmap saved: {filename}")
                    break

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = '/jhcnas7/Pathology/PathLab_data_collection/Data/CRC-MSI/CRC-MSI'
    
    # Load data
    df, data_path = load_data(data_path)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, test_df = create_dataloaders(df, data_path, batch_size=32)
    
    # Build and train model
    model = build_model().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTraining model...")
    model = train_model(model, train_loader, val_loader, epochs=20, device=device)
    
    # Test model
    print("\nTesting model...")
    metrics, all_logits, all_labels, all_paths, cm, fpr, tpr = test_model(model, test_loader, test_df, device)
    
    # Generate heatmaps for specific images
    target_images = [
        'TCGA-QG-A5Z2-01Z-00-DX2.F2352352-8F00-4BB3-8A62-8D1C1E374F95_(14367,54176).jpg',
        'TCGA-DM-A28G-01Z-00-DX1.5e8602bd-31e1-4813-8214-cd56280defe5_(10373,34433).jpg'
    ]
    
    print("\nGenerating heatmaps...")
    generate_heatmaps(model, target_images, '/jhcnas7/Pathology/PathLab_data_collection/Data/CRC-MSI/CRC-MSI', device)
    
    # Save metrics (handle NaN values for JSON compliance)
    import math
    safe_metrics = {}
    for k, v in metrics.items():
        val = float(v)
        safe_metrics[k] = None if math.isnan(val) else val
    with open('metrics.json', 'w') as f:
        json.dump(safe_metrics, f, indent=2)
    
    print("\n✓ Task 1 completed!")

if __name__ == '__main__':
    main()
