"""
训练潜在空间分类器
使用VAE encoder提取的latent vectors + 视觉标签
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
# from tqdm import tqdm  # Optional, fallback to simple loop

# 导入VAE和分类器
from vae_recon.vae_model_64x64 import SimpleVAE64x64
from lane_classifier.latent_classifier import get_latent_classifier
from lane_classifier.dataset_visual import LaneDatasetVisual


class LatentDataset(Dataset):
    """
    从图像数据集提取latent vectors作为训练数据
    """
    def __init__(self, image_dataset, vae_model, device):
        """
        Args:
            image_dataset: LaneDatasetVisual实例
            vae_model: 训练好的VAE模型
            device: 计算设备
        """
        self.device = device
        print(f"Extracting latent vectors from {len(image_dataset)} images...")
        
        vae_model.eval()
        self.latents = []
        self.labels = []
        
        # 批量提取latent vectors
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(image_dataset), batch_size):
                if i % 500 == 0:
                    print(f"  Encoding {i}/{len(image_dataset)}...")
                end_idx = min(i + batch_size, len(image_dataset))
                batch_images = []
                batch_labels = []
                
                for j in range(i, end_idx):
                    img, label = image_dataset[j]
                    batch_images.append(img)
                    batch_labels.append(label)
                
                batch_images = torch.stack(batch_images).to(device)
                
                # Encode to latent space
                z, _, _ = vae_model.encode(batch_images)
                
                self.latents.append(z.cpu())
                self.labels.extend(batch_labels)
        
        self.latents = torch.cat(self.latents, dim=0)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        print(f"Extracted {len(self.latents)} latent vectors")
        print(f"Latent shape: {self.latents[0].shape}")
        print(f"Label distribution: Left={sum(self.labels==0)}, Right={sum(self.labels==1)}")
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for latents, labels in train_loader:
        latents = latents.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(latents)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for latents, labels in val_loader:
            latents = latents.to(device)
            labels = labels.to(device)
            
            outputs = model(latents)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(val_loader), 100.0 * correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train Latent Space Classifier')
    parser.add_argument('--vae_model_path', type=str, default='vae_recon/best_model.pt',
                        help='Path to trained VAE model')
    parser.add_argument('--npz_files', nargs='+', default=['npz_data/traj1_64x64.npz', 'npz_data/traj2_64x64.npz'],
                        help='NPZ data files')
    parser.add_argument('--model_type', type=str, default='conv', choices=['mlp', 'conv'],
                        help='Classifier architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--save_dir', type=str, default='lane_classifier/latent_classifier_checkpoints',
                        help='Save directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*70)
    print("Training Latent Space Classifier")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    
    # 1. 加载VAE
    print(f"\nLoading VAE from {args.vae_model_path}...")
    vae = SimpleVAE64x64(latent_dim=64)
    vae_checkpoint = torch.load(args.vae_model_path, map_location=device, weights_only=False)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    
    # 2. 加载图像数据集（使用视觉标签）
    print(f"\nLoading image dataset...")
    image_dataset = LaneDatasetVisual(
        npz_files=args.npz_files,
        balance=True,  # 平衡类别
        debug=False
    )
    
    # 3. 提取latent vectors
    latent_dataset = LatentDataset(image_dataset, vae, device)
    
    # 4. 划分训练/验证集
    val_size = int(len(latent_dataset) * args.val_split)
    train_size = len(latent_dataset) - val_size
    train_dataset, val_dataset = random_split(
        latent_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    
    # 5. 创建分类器
    model = get_latent_classifier(
        model_type=args.model_type,
        latent_dim=64,
        latent_spatial_size=4,
        dropout_rate=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model_type}")
    print(f"Total parameters: {total_params:,}")
    
    # 6. 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))
    
    # 7. 训练
    print(f"\n{'='*70}")
    print("Starting Training...")
    print("="*70)
    
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Scheduler
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%", end='')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'args': vars(args)
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(" [*] Best model saved!")
        else:
            print()
    
    # 8. 保存最终模型和训练曲线
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'args': vars(args)
    }, os.path.join(args.save_dir, 'final_model.pt'))
    
    # 绘制训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, args.epochs + 1)
    ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs_range, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs_range, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    # 混淆矩阵
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Left', 'Right'],
                yticklabels=['Left', 'Right'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Best Val Acc: {best_val_acc:.2f}%')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {args.save_dir}")
    print(f"\nFinal Classification Report:")
    print(classification_report(val_labels, val_preds, target_names=['Left', 'Right'], digits=4))
    
    writer.close()


if __name__ == '__main__':
    main()
