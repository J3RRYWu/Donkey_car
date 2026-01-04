"""
Simplified VAE model for 64x64 images
Lighter architecture compared to 224x224 version
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import os
from typing import Tuple
from PIL import Image


class SimpleVAE64x64(nn.Module):
    """
    Simplified VAE for 64x64 images
    - Lighter architecture
    - Faster training
    - Lower memory usage
    """
    
    def __init__(self, latent_dim: int = 64, channels: int = 3, use_skip_connections: bool = False):
        super(SimpleVAE64x64, self).__init__()
        self.latent_dim = latent_dim  # Latent channels (increased from 32 to 128)
        self.channels = channels
        self.use_skip_connections = use_skip_connections
        
        # Encoder: 64 -> 32 -> 16 -> 8 -> 4
        # Input: (B, 3, 64, 64)
        # Use LayerNorm instead of BatchNorm to avoid collapse
        self.encoder_conv1 = nn.Sequential(  # 64 -> 32
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),  # GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True),
        )
        self.encoder_conv2 = nn.Sequential(  # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv3 = nn.Sequential(  # 16 -> 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv4 = nn.Sequential(  # 8 -> 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.ReLU(inplace=True),
        )
        
        # Latent space: 4x4 feature map
        self.to_mu = nn.Conv2d(512, latent_dim, kernel_size=1)  # (B, latent_dim, 4, 4)
        self.to_logvar = nn.Conv2d(512, latent_dim, kernel_size=1)  # (B, latent_dim, 4, 4)
        
        # Decoder: 4 -> 8 -> 16 -> 32 -> 64
        # Increased capacity and removed BatchNorm
        self.decoder_start = nn.Sequential(
            nn.Conv2d(latent_dim, 512, kernel_size=1),  # (B, 512, 4, 4)
            nn.ReLU(inplace=True),  # No normalization in decoder
        )
        
        self.decoder_block1 = nn.Sequential(  # 4 -> 8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),  # No BatchNorm
        )
        self.decoder_block2 = nn.Sequential(  # 8 -> 16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder_block3 = nn.Sequential(  # 16 -> 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder_block4 = nn.Sequential(  # 32 -> 64
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # Increased from 32 to 64
            nn.ReLU(inplace=True),
        )
        
        # Final output with more capacity
        self.decoder_output = nn.Sequential(
            nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0, 1] - better for bright floor + dark background
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Encode image to latent space"""
        h1 = self.encoder_conv1(x)  # (B, 64, 32, 32)
        h2 = self.encoder_conv2(h1)  # (B, 128, 16, 16)
        h3 = self.encoder_conv3(h2)  # (B, 256, 8, 8)
        h4 = self.encoder_conv4(h3)  # (B, 512, 4, 4)
        
        mu = self.to_mu(h4)  # (B, latent_dim, 4, 4)
        logvar = self.to_logvar(h4)  # (B, latent_dim, 4, 4)
        
        return mu, logvar, []
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, skip_features: list = None) -> torch.Tensor:
        """Decode latent to image"""
        h = self.decoder_start(z)  # (B, 512, 4, 4)
        h = self.decoder_block1(h)  # (B, 256, 8, 8)
        h = self.decoder_block2(h)  # (B, 128, 16, 16)
        h = self.decoder_block3(h)  # (B, 64, 32, 32)
        h = self.decoder_block4(h)  # (B, 32, 64, 64)
        x_recon = self.decoder_output(h)  # (B, channels, 64, 64)
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar, skip_features = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, skip_features)
        return x_recon, mu, logvar, z


class ImageDataset64x64(Dataset):
    """Dataset for loading 64x64 images from NPZ files (auto-resizes if needed)"""
    
    def __init__(self, npz_paths: list, normalize: bool = True, target_size: int = 64):
        self.normalize = normalize
        self.target_size = target_size
        
        # Load all images from NPZ files
        self.images = []
        self.need_resize = False
        
        for npz_path in npz_paths:
            if not os.path.exists(npz_path):
                print(f"Warning: {npz_path} does not exist, skipping")
                continue
            data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
            frames = data['frame']  # (T, 3, H, W), uint8
            
            # Check if resize is needed
            if len(frames) > 0:
                sample_shape = frames[0].shape
                if sample_shape[1] != target_size or sample_shape[2] != target_size:
                    print(f"  Detected image size {sample_shape[1]}x{sample_shape[2]}, will resize to {target_size}x{target_size}")
                    self.need_resize = True
                    # Resize all frames
                    print(f"  Resizing {len(frames)} frames...")
                    resized_frames = []
                    for i in range(len(frames)):
                        if (i + 1) % 1000 == 0:
                            print(f"    Resized {i+1}/{len(frames)} frames")
                        frame = frames[i].transpose(1, 2, 0)  # (H, W, 3)
                        img = Image.fromarray(frame)
                        img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                        frame_resized = np.array(img_resized).transpose(2, 0, 1)  # (3, 64, 64)
                        resized_frames.append(frame_resized)
                    frames = np.array(resized_frames, dtype=np.uint8)
                    print(f"  ✓ Resized to {frames.shape}")
            
            self.images.append(frames)
            print(f"Loaded {npz_path}: {len(frames)} images")
        
        if not self.images:
            raise ValueError("No images loaded from NPZ files")
        
        # Concatenate all images
        self.images = np.concatenate(self.images, axis=0)
        print(f"Total images: {len(self.images)}")
        if self.need_resize:
            print(f"Note: Images were automatically resized to {target_size}x{target_size}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self.images) + idx
        
        # Get image: should be (3, 64, 64), uint8
        image = self.images[idx].astype(np.float32)
        
        # Convert to tensor
        image_t = torch.from_numpy(image)
        
        if image_t.dtype != torch.float32:
            image_t = image_t.float()
        
        # Normalize to [0, 1] (matches Sigmoid output)
        if self.normalize:
            image_t = image_t / 255.0  # [0, 1]
        
        return image_t


def save_model_64x64(model: SimpleVAE64x64, path: str):
    """Save 64x64 model state"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': model.latent_dim,
        'channels': model.channels,
        'use_skip_connections': model.use_skip_connections,
        'image_size': 64,
    }, path)
    print(f"Model saved to {path}")


def load_model_64x64(path: str, device: torch.device) -> SimpleVAE64x64:
    """Load 64x64 model state"""
    checkpoint = torch.load(path, map_location=device)
    
    # Handle different checkpoint formats
    if 'latent_dim' in checkpoint:
        latent_dim = checkpoint['latent_dim']
        channels = checkpoint.get('channels', 3)
        use_skip_connections = checkpoint.get('use_skip_connections', False)
        state_dict = checkpoint['model_state_dict']
    elif 'args' in checkpoint:
        args = checkpoint['args']
        latent_dim = args.get('latent_dim', 32)
        channels = args.get('channels', 3)
        use_skip_connections = args.get('use_skip_connections', False)
        state_dict = checkpoint['model_state_dict']
    else:
        latent_dim = 64  # Default for 64x64
        channels = 3
        use_skip_connections = False
        state_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
    
    model = SimpleVAE64x64(
        latent_dim=latent_dim,
        channels=channels,
        use_skip_connections=use_skip_connections
    )
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Model loaded from {path}")
    print(f"  - Latent dim: {latent_dim}")
    print(f"  - Image size: 64x64")
    print(f"  - Channels: {channels}")
    return model

