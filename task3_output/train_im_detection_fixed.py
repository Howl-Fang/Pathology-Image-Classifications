import os, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt, seaborn as sns
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import openslide
    HAS_OPENSLIDE = True
except:
    HAS_OPENSLIDE = False

class IMDetectionDataset(Dataset):
    def __init__(self, df, data_path, transform=None):
        self.df, self.data_path, self.transform = df, data_path, transform
        self.label_map = {'Intestinal metaplasia': 1, 'Not Intestinal metaplasia': 0, 'IM': 1, 'nonIM': 0}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label_str = row.get('label', row.get('Label', 'nonIM'))
        filename = row.get('filename', row.get('file_name', ''))
        wsi_path = os.path.join(self.data_path, 'WSIs', filename)
        
        try:
            if HAS_OPENSLIDE and os.path.exists(wsi_path):
                import random
                slide = openslide.open_slide(wsi_path)
                w, h = slide.dimensions
                x = random.randint(0, max(0, w - 256))
                y = random.randint(0, max(0, h - 256))
                patch = slide.read_region((x, y), 0, (256, 256))
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
    df = pd.read_csv(os.path.join(data_path, 'PWH', 'label.csv'))
    if 'label' in df.columns:
        df = df[df['label'].isin(['Intestinal metaplasia', 'Not Intestinal metaplasia'])]
    print(f"Total PWH samples: {len(df)}")
    if len(df) > 0 and 'label' in df.columns:
        print(f"Class distribution:\n{df['label'].value_counts()}")
    return df

def create_dataloaders(df, data_path, batch_size=16):
    train_split, test_split = train_test_split(df, test_size=0.3, random_state=42)
    train_split, val_split = train_test_split(train_split, test_size=0.2, random_state=42)
    print(f"Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = IMDetectionDataset(train_split, data_path, train_transform)
    val_dataset = IMDetectionDataset(val_split, data_path, val_test_transform)
    test_dataset = IMDetectionDataset(test_split, data_path, val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

def build_model():
    model = resnet50(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 2))
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
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(val_loader), accuracy_score(all_labels, all_preds)

def train_model(model, train_loader, val_loader, epochs=20, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    best_val_acc, patience_counter = 0, 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc, patience_counter = val_acc, 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= 5:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def test_model(model, test_loader, device):
    model.eval()
    all_preds, all_logits, all_labels, all_paths = [], [], [], []
    
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
    
    all_preds, all_logits, all_labels = np.array(all_preds), np.array(all_logits), np.array(all_labels)
    
    macro_auc = roc_auc_score(all_labels, all_logits[:, 1], average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_logits[:, 1])
    roc_auc = auc(fpr, tpr)
    
    metrics = {'Macro-AUC': macro_auc, 'Weighted-F1': weighted_f1, 'Macro-ACC': macro_acc, 'ROC-AUC': roc_auc}
    
    print("\n=== PWH Test Set Performance ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    pd.DataFrame({'file_name': all_paths, 'IM': all_logits[:, 1], 'nonIM': all_logits[:, 0]}).to_csv('IM.csv', index=False)
    print("\nPredictions saved to IM.csv")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NonIM', 'IM'], yticklabels=['NonIM', 'IM'])
    plt.title('Confusion Matrix - Intestinal Metaplasia Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    
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
    
    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    data_path = '/jhcnas7/Pathology/PathLab_data_collection/Data/GC_IM_Detection'
    df = load_pwh_data(data_path)
    if len(df) == 0:
        return
    train_loader, val_loader, test_loader = create_dataloaders(df, data_path, batch_size=16)
    model = build_model().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nTraining model on PWH...")
    model = train_model(model, train_loader, val_loader, epochs=20, device=device)
    print("\nTesting model...")
    metrics = test_model(model, test_loader, device)
    with open('metrics.json', 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print("\n✓ Task 3 completed!")

if __name__ == '__main__':
    main()
