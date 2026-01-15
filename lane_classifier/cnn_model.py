"""
CNN Binary Classifier for Lane Position (Left/Right)
Based on Cross Track Error (CTE) from NPZ data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneCNN(nn.Module):
    """
    CNN Binary Classifier for lane position detection
    Input: (B, 3, 64, 64) RGB images
    Output: (B, 2) logits for [left, right] classification
    """
    
    def __init__(self, dropout_rate=0.5):
        super(LaneCNN, self).__init__()
        
        # Convolutional layers
        # Input: (B, 3, 64, 64)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)  # -> (B, 32, 32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # -> (B, 64, 16, 16)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # -> (B, 128, 8, 8)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # -> (B, 256, 4, 4)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, 2)  # Binary classification: left (0) or right (1)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (B, 3, 64, 64) input images
        Returns:
            logits: (B, 2) classification logits
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        logits = self.fc3(x)
        
        return logits


class LaneCNNLightweight(nn.Module):
    """
    Lightweight CNN for faster training and inference
    Input: (B, 3, 64, 64) RGB images
    Output: (B, 2) logits for [left, right] classification
    """
    
    def __init__(self, dropout_rate=0.3):
        super(LaneCNNLightweight, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)  # -> (B, 16, 32, 32)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # -> (B, 32, 16, 16)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # -> (B, 64, 8, 8)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 2)  # Binary classification
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (B, 3, 64, 64) input images
        Returns:
            logits: (B, 2) classification logits
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        logits = self.fc2(x)
        
        return logits


def get_model(model_type='standard', dropout_rate=0.5):
    """
    Factory function to create model
    Args:
        model_type: 'standard' or 'lightweight'
        dropout_rate: dropout probability
    Returns:
        model: LaneCNN instance
    """
    if model_type == 'lightweight':
        return LaneCNNLightweight(dropout_rate=dropout_rate)
    else:
        return LaneCNN(dropout_rate=dropout_rate)


if __name__ == '__main__':
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test standard model
    model = LaneCNN().to(device)
    print("Standard Model:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 64, 64).to(device)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output}")
    
    # Test lightweight model
    print("\n" + "="*50)
    model_light = LaneCNNLightweight().to(device)
    print("\nLightweight Model:")
    print(model_light)
    print(f"\nTotal parameters: {sum(p.numel() for p in model_light.parameters()):,}")
    
    output_light = model_light(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output_light.shape}")
    print(f"Output logits: {output_light}")
