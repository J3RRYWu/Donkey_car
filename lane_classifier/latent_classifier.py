"""
Latent Space Classifier
直接在潜在空间进行左右分类，跳过VAE解码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentClassifier(nn.Module):
    """
    在VAE潜在空间进行分类
    Input: latent_z (B, latent_dim, H, W) - 来自VAE encoder或LSTM predictor
    Output: logits (B, 2) - Left (0) or Right (1)
    """
    
    def __init__(self, latent_dim=64, latent_spatial_size=4, dropout_rate=0.3):
        """
        Args:
            latent_dim: 潜在向量的通道数
            latent_spatial_size: 潜在向量的空间大小 (H=W)
            dropout_rate: Dropout概率
        """
        super(LatentClassifier, self).__init__()
        
        self.latent_dim = latent_dim
        self.latent_spatial_size = latent_spatial_size
        
        # 从潜在空间 (64, 4, 4) = 1024维 到分类
        input_size = latent_dim * latent_spatial_size * latent_spatial_size
        
        # 简单的MLP分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 2)  # Binary classification
        )
    
    def forward(self, z):
        """
        Args:
            z: Latent tensor (B, latent_dim, H, W)
        Returns:
            logits: (B, 2)
        """
        return self.classifier(z)


class ConvLatentClassifier(nn.Module):
    """
    使用卷积的潜在空间分类器
    可能比纯MLP更好，因为保留了空间结构
    """
    
    def __init__(self, latent_dim=64, latent_spatial_size=4, dropout_rate=0.3):
        super(ConvLatentClassifier, self).__init__()
        
        self.latent_dim = latent_dim
        self.latent_spatial_size = latent_spatial_size
        
        # 卷积层处理潜在空间
        # (64, 4, 4) -> (128, 2, 2) -> (256, 1, 1)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(latent_dim, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),
        )
        
        # 全局平均池化 + 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 2)
        )
    
    def forward(self, z):
        """
        Args:
            z: Latent tensor (B, latent_dim, H, W)
        Returns:
            logits: (B, 2)
        """
        features = self.conv_layers(z)
        logits = self.classifier(features)
        return logits


def get_latent_classifier(model_type='mlp', latent_dim=64, latent_spatial_size=4, dropout_rate=0.3):
    """
    工厂函数
    Args:
        model_type: 'mlp' or 'conv'
        latent_dim: 潜在向量维度
        latent_spatial_size: 空间大小
        dropout_rate: Dropout率
    Returns:
        classifier model
    """
    if model_type == 'mlp':
        return LatentClassifier(latent_dim, latent_spatial_size, dropout_rate)
    elif model_type == 'conv':
        return ConvLatentClassifier(latent_dim, latent_spatial_size, dropout_rate)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
