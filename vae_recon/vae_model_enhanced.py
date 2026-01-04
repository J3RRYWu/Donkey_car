import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import os
from typing import Tuple
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Try to load VGG16 with pretrained weights
        try:
            # New torchvision API (>=0.13)
            try:
                vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
            except:
                # Old torchvision API (<0.13)
                vgg = models.vgg16(pretrained=True).features
        except Exception as e:
            print(f"Warning: Failed to load VGG16 pretrained weights: {e}")
            print("Using VGG16 without pretrained weights (may affect perceptual loss quality)")
            vgg = models.vgg16(weights=None).features
        
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # conv1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # conv2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]) # conv3_3
        self.slice4 = nn.Sequential(*list(vgg.children())[16:23]) # conv4_3
        
        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, x, y):
        """Compute perceptual loss between x and y"""
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        x_norm = (x - mean) / std
        y_norm = (y - mean) / std
        
        # Resize to 224x224 if needed
        if x.size(-1) != 224:
            x_norm = F.interpolate(x_norm, size=(224, 224), mode='bilinear', align_corners=False)
            y_norm = F.interpolate(y_norm, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features and compute loss per layer
        h_x = x_norm
        h_y = y_norm
        
        loss = 0.0
        weights = [1.0, 1.0, 1.0, 1.0]  # Layer weights (can be adjusted)
        
        for i, slice_fn in enumerate([self.slice1, self.slice2, self.slice3, self.slice4]):
            h_x = slice_fn(h_x)
            h_y = slice_fn(h_y)
            # Normalize by number of elements to make it scale-invariant
            layer_loss = F.mse_loss(h_x, h_y, reduction='mean')
            loss += weights[i] * layer_loss
        
        # Normalize by number of layers to keep scale reasonable
        loss = loss / len([self.slice1, self.slice2, self.slice3, self.slice4])
        
        return loss


class EnhancedVAE(nn.Module):
    """
    Enhanced VAE for better reconstruction quality
    - Deeper architecture with residual connections
    - Larger latent dimension
    - Better feature extraction
    """
    
    def __init__(self, latent_dim: int = 32, image_size: int = 224, channels: int = 3, use_skip_connections: bool = True):
        super(EnhancedVAE, self).__init__()
        self.latent_dim = latent_dim  # Now refers to latent channels (not total dim)
        self.image_size = image_size
        self.channels = channels
        self.use_skip_connections = use_skip_connections
        
        # Encoder: Deeper with residual blocks
        # Input: (B, 3, 224, 224)
        # Store intermediate features for skip connections
        self.encoder_conv1 = nn.Sequential(  # 224 -> 112
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv2 = nn.Sequential(  # 112 -> 56
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv3 = nn.Sequential(  # 56 -> 28
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv4 = nn.Sequential(  # 28 -> 14
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv5 = nn.Sequential(  # 14 -> 7
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Final encoder layers to get 1024 channels at 7x7
        self.encoder_final = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        
        # Convolutional latent space: predict μ/σ directly from 7×7 feature map
        # Remove: encoder_pool, fc_mu, fc_logvar
        # Add: Conv2d layers to predict μ and σ
        self.to_mu = nn.Conv2d(1024, latent_dim, kernel_size=1)  # (B, latent_dim, 7, 7)
        self.to_logvar = nn.Conv2d(1024, latent_dim, kernel_size=1)  # (B, latent_dim, 7, 7)
        
        # Decoder: Start from 7×7 latent map (not vector)
        # Remove: decoder_fc (Linear layer)
        # Add: Conv2d to convert latent to decoder input
        self.decoder_start = nn.Sequential(
            nn.Conv2d(latent_dim, 1024, kernel_size=1),  # (B, 1024, 7, 7)
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        
        # Decoder blocks with skip connections
        # If using skip connections, need to concatenate encoder features
        if use_skip_connections:
            # Skip connection channel adjustment layers
            self.skip_conv2 = nn.Conv2d(512, 256, kernel_size=1)  # For encoder_conv4 (512 -> 256)
            self.skip_conv3 = nn.Conv2d(256, 128, kernel_size=1)  # For encoder_conv3 (256 -> 128)
            self.skip_conv4 = nn.Conv2d(128, 64, kernel_size=1)   # For encoder_conv2 (128 -> 64)
            self.skip_conv5 = nn.Conv2d(64, 32, kernel_size=1)    # For encoder_conv1 (64 -> 32)
            
            # Block 1: 7 -> 14 (concatenate with encoder_conv5 output: 512 channels)
            self.decoder_block1 = nn.ModuleList([
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 7 -> 14
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512 + 512, 512, kernel_size=3, stride=1, padding=1),  # Concatenate skip
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ])
            
            # Block 2: 14 -> 28 (concatenate with encoder_conv4 output: 512 channels -> 256)
            self.decoder_block2 = nn.ModuleList([
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 14 -> 28
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256 + 256, 256, kernel_size=3, stride=1, padding=1),  # Concatenate skip (after conv)
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ])
            
            # Block 3: 28 -> 56 (concatenate with encoder_conv3 output: 256 channels -> 128)
            self.decoder_block3 = nn.ModuleList([
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28 -> 56
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1),  # Concatenate skip (after conv)
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ])
            
            # Block 4: 56 -> 112 (concatenate with encoder_conv2 output: 128 channels -> 64)
            self.decoder_block4 = nn.ModuleList([
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56 -> 112
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),  # Concatenate skip (after conv)
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ])
            
            # Block 5: 112 -> 224 (concatenate with encoder_conv1 output: 64 channels -> 32)
            self.decoder_block5 = nn.ModuleList([
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 112 -> 224
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32 + 32, 32, kernel_size=3, stride=1, padding=1),  # Concatenate skip (after conv)
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            ])
        else:
            # No skip connections: simpler decoder
            self.decoder_block1 = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.decoder_block2 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.decoder_block3 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.decoder_block4 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.decoder_block5 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
        
        # Final output layer
        # Use Tanh + rescale instead of Sigmoid for better gradient flow
        self.decoder_output = nn.Sequential(
            nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output in [-1, 1], will be rescaled to [0, 1] in loss
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Encode image to latent space (convolutional)
        Returns: mu, logvar, skip_features
        """
        # Store intermediate features for skip connections
        skip_features = []
        
        # Encoder forward pass with skip connections
        h1 = self.encoder_conv1(x)  # (B, 64, 112, 112)
        skip_features.append(h1)
        
        h2 = self.encoder_conv2(h1)  # (B, 128, 56, 56)
        skip_features.append(h2)
        
        h3 = self.encoder_conv3(h2)  # (B, 256, 28, 28)
        skip_features.append(h3)
        
        h4 = self.encoder_conv4(h3)  # (B, 512, 14, 14)
        skip_features.append(h4)
        
        h5 = self.encoder_conv5(h4)  # (B, 512, 7, 7)
        skip_features.append(h5)
        
        # Final encoder layers
        h_final = self.encoder_final(h5)  # (B, 1024, 7, 7)
        
        # Predict μ and σ directly from 7×7 feature map (convolutional)
        mu = self.to_mu(h_final)  # (B, latent_dim, 7, 7)
        logvar = self.to_logvar(h_final)  # (B, latent_dim, 7, 7)
        
        return mu, logvar, skip_features
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick (convolutional)
        mu, logvar: (B, latent_dim, 7, 7)
        Returns: z (B, latent_dim, 7, 7)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, skip_features: list = None) -> torch.Tensor:
        """
        Decode latent to image (convolutional with skip connections)
        z: (B, latent_dim, 7, 7)
        skip_features: list of encoder features for skip connections
        Returns: x_recon (B, channels, 224, 224)
        """
        # Start from 7×7 latent map
        h = self.decoder_start(z)  # (B, 1024, 7, 7)
        
        if self.use_skip_connections:
            if skip_features is not None:
                # Block 1: 7 -> 14 (use skip_features[4]: encoder_conv5 output)
                h = self.decoder_block1[0](h)  # ConvTranspose: (B, 512, 14, 14)
                h = self.decoder_block1[1](h)  # BatchNorm
                h = self.decoder_block1[2](h)  # ReLU
                skip = skip_features[4]  # (B, 512, 7, 7) -> need to upsample to 14×14
                skip_up = F.interpolate(skip, size=(14, 14), mode='bilinear', align_corners=False)  # (B, 512, 14, 14)
                h = torch.cat([h, skip_up], dim=1)  # (B, 1024, 14, 14)
                h = self.decoder_block1[3](h)  # Conv: (B, 512, 14, 14)
                h = self.decoder_block1[4](h)  # BatchNorm
                h = self.decoder_block1[5](h)  # ReLU
                
                # Block 2: 14 -> 28 (use skip_features[3]: encoder_conv4 output)
                h = self.decoder_block2[0](h)  # ConvTranspose: (B, 256, 28, 28)
                h = self.decoder_block2[1](h)  # BatchNorm
                h = self.decoder_block2[2](h)  # ReLU
                skip = skip_features[3]  # (B, 512, 14, 14)
                skip_up = F.interpolate(skip, size=(28, 28), mode='bilinear', align_corners=False)  # (B, 512, 28, 28)
                skip_up = self.skip_conv2(skip_up)  # (B, 256, 28, 28)
                h = torch.cat([h, skip_up], dim=1)  # (B, 512, 28, 28)
                h = self.decoder_block2[3](h)  # Conv: (B, 256, 28, 28)
                h = self.decoder_block2[4](h)  # BatchNorm
                h = self.decoder_block2[5](h)  # ReLU
                
                # Block 3: 28 -> 56 (use skip_features[2]: encoder_conv3 output)
                h = self.decoder_block3[0](h)  # ConvTranspose: (B, 128, 56, 56)
                h = self.decoder_block3[1](h)  # BatchNorm
                h = self.decoder_block3[2](h)  # ReLU
                skip = skip_features[2]  # (B, 256, 28, 28)
                skip_up = F.interpolate(skip, size=(56, 56), mode='bilinear', align_corners=False)  # (B, 256, 56, 56)
                skip_up = self.skip_conv3(skip_up)  # (B, 128, 56, 56)
                h = torch.cat([h, skip_up], dim=1)  # (B, 256, 56, 56)
                h = self.decoder_block3[3](h)  # Conv: (B, 128, 56, 56)
                h = self.decoder_block3[4](h)  # BatchNorm
                h = self.decoder_block3[5](h)  # ReLU
                
                # Block 4: 56 -> 112 (use skip_features[1]: encoder_conv2 output)
                h = self.decoder_block4[0](h)  # ConvTranspose: (B, 64, 112, 112)
                h = self.decoder_block4[1](h)  # BatchNorm
                h = self.decoder_block4[2](h)  # ReLU
                skip = skip_features[1]  # (B, 128, 56, 56)
                skip_up = F.interpolate(skip, size=(112, 112), mode='bilinear', align_corners=False)  # (B, 128, 112, 112)
                skip_up = self.skip_conv4(skip_up)  # (B, 64, 112, 112)
                h = torch.cat([h, skip_up], dim=1)  # (B, 128, 112, 112)
                h = self.decoder_block4[3](h)  # Conv: (B, 64, 112, 112)
                h = self.decoder_block4[4](h)  # BatchNorm
                h = self.decoder_block4[5](h)  # ReLU
                
                # Block 5: 112 -> 224 (use skip_features[0]: encoder_conv1 output)
                h = self.decoder_block5[0](h)  # ConvTranspose: (B, 32, 224, 224)
                h = self.decoder_block5[1](h)  # BatchNorm
                h = self.decoder_block5[2](h)  # ReLU
                skip = skip_features[0]  # (B, 64, 112, 112)
                skip_up = F.interpolate(skip, size=(224, 224), mode='bilinear', align_corners=False)  # (B, 64, 224, 224)
                skip_up = self.skip_conv5(skip_up)  # (B, 32, 224, 224)
                h = torch.cat([h, skip_up], dim=1)  # (B, 64, 224, 224)
                h = self.decoder_block5[3](h)  # Conv: (B, 32, 224, 224)
                h = self.decoder_block5[4](h)  # BatchNorm
                h = self.decoder_block5[5](h)  # ReLU
            else:
                # Skip connections enabled but no skip features provided (e.g., from predictor)
                # Manually call ModuleList elements without skip concatenation
                # Block 1: 7 -> 14
                h = self.decoder_block1[0](h)  # ConvTranspose: (B, 512, 14, 14)
                h = self.decoder_block1[1](h)  # BatchNorm
                h = self.decoder_block1[2](h)  # ReLU
                # Skip concat step - just apply the conv layer with adjusted input channels
                # Since we can't concat, we need to adjust: decoder_block1[3] expects (B, 1024, 14, 14)
                # But we only have (B, 512, 14, 14), so we need a workaround
                # Actually, we should just apply a conv to match expected channels
                # For now, let's use a simpler approach: apply conv with proper channel adjustment
                h_temp = torch.zeros(h.size(0), 512, h.size(2), h.size(3), device=h.device, dtype=h.dtype)
                h = torch.cat([h, h_temp], dim=1)  # (B, 1024, 14, 14) - pad with zeros instead of skip
                h = self.decoder_block1[3](h)  # Conv: (B, 512, 14, 14)
                h = self.decoder_block1[4](h)  # BatchNorm
                h = self.decoder_block1[5](h)  # ReLU
                
                # Block 2: 14 -> 28
                h = self.decoder_block2[0](h)  # ConvTranspose: (B, 256, 28, 28)
                h = self.decoder_block2[1](h)  # BatchNorm
                h = self.decoder_block2[2](h)  # ReLU
                h_temp = torch.zeros(h.size(0), 256, h.size(2), h.size(3), device=h.device, dtype=h.dtype)
                h = torch.cat([h, h_temp], dim=1)  # (B, 512, 28, 28)
                h = self.decoder_block2[3](h)  # Conv: (B, 256, 28, 28)
                h = self.decoder_block2[4](h)  # BatchNorm
                h = self.decoder_block2[5](h)  # ReLU
                
                # Block 3: 28 -> 56
                h = self.decoder_block3[0](h)  # ConvTranspose: (B, 128, 56, 56)
                h = self.decoder_block3[1](h)  # BatchNorm
                h = self.decoder_block3[2](h)  # ReLU
                h_temp = torch.zeros(h.size(0), 128, h.size(2), h.size(3), device=h.device, dtype=h.dtype)
                h = torch.cat([h, h_temp], dim=1)  # (B, 256, 56, 56)
                h = self.decoder_block3[3](h)  # Conv: (B, 128, 56, 56)
                h = self.decoder_block3[4](h)  # BatchNorm
                h = self.decoder_block3[5](h)  # ReLU
                
                # Block 4: 56 -> 112
                h = self.decoder_block4[0](h)  # ConvTranspose: (B, 64, 112, 112)
                h = self.decoder_block4[1](h)  # BatchNorm
                h = self.decoder_block4[2](h)  # ReLU
                h_temp = torch.zeros(h.size(0), 64, h.size(2), h.size(3), device=h.device, dtype=h.dtype)
                h = torch.cat([h, h_temp], dim=1)  # (B, 128, 112, 112)
                h = self.decoder_block4[3](h)  # Conv: (B, 64, 112, 112)
                h = self.decoder_block4[4](h)  # BatchNorm
                h = self.decoder_block4[5](h)  # ReLU
                
                # Block 5: 112 -> 224
                h = self.decoder_block5[0](h)  # ConvTranspose: (B, 32, 224, 224)
                h = self.decoder_block5[1](h)  # BatchNorm
                h = self.decoder_block5[2](h)  # ReLU
                h_temp = torch.zeros(h.size(0), 32, h.size(2), h.size(3), device=h.device, dtype=h.dtype)
                h = torch.cat([h, h_temp], dim=1)  # (B, 64, 224, 224)
                h = self.decoder_block5[3](h)  # Conv: (B, 32, 224, 224)
                h = self.decoder_block5[4](h)  # BatchNorm
                h = self.decoder_block5[5](h)  # ReLU
        else:
            # No skip connections: simple decoder
            h = self.decoder_block1(h)  # (B, 512, 14, 14)
            h = self.decoder_block2(h)  # (B, 256, 28, 28)
            h = self.decoder_block3(h)  # (B, 128, 56, 56)
            h = self.decoder_block4(h)  # (B, 64, 112, 112)
            h = self.decoder_block5(h)  # (B, 32, 224, 224)
        
        # Final output
        x_recon = self.decoder_output(h)  # (B, channels, 224, 224)
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode, reparameterize, decode"""
        mu, logvar, skip_features = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, skip_features)
        return x_recon, mu, logvar, z


class ImageDataset(Dataset):
    """Dataset for loading images from NPZ files"""
    
    def __init__(self, npz_paths: list, image_size: int = 224, normalize: bool = True):
        self.image_size = image_size
        self.normalize = normalize
        
        # Load all images from NPZ files
        self.images = []
        for npz_path in npz_paths:
            if not os.path.exists(npz_path):
                print(f"Warning: {npz_path} does not exist, skipping")
                continue
            data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
            frames = data['frame']  # (T, 3, 224, 224), uint8
            self.images.append(frames)
            print(f"Loaded {npz_path}: {len(frames)} images")
        
        if not self.images:
            raise ValueError("No images loaded from NPZ files")
        
        # Concatenate all images
        self.images = np.concatenate(self.images, axis=0)
        print(f"Total images: {len(self.images)}")
        
        # Verify image shape
        if len(self.images) > 0:
            sample_shape = self.images[0].shape
            print(f"Image shape: {sample_shape}")
            if sample_shape[0] != 3 and sample_shape[2] == 3:
                print(f"Warning: Image shape {sample_shape} may need transpose (expected (3, H, W))")
            elif sample_shape[0] == 3:
                print(f"✓ Image format: (3, {sample_shape[1]}, {sample_shape[2]}) - correct")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self.images) + idx
        
        # Get image: should be (3, 224, 224), uint8
        image = self.images[idx].astype(np.float32)
        
        # Check and fix shape if needed
        if image.ndim == 3:
            # Check if channels are first or last
            if image.shape[0] == 3:
                # (3, H, W) - correct format
                pass
            elif image.shape[2] == 3:
                # (H, W, 3) - need to transpose
                image = image.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        else:
            raise ValueError(f"Unexpected image dimensions: {image.ndim}")
        
        # Convert to tensor
        image_t = torch.from_numpy(image)
        
        # Ensure correct dtype and shape
        if image_t.dtype != torch.float32:
            image_t = image_t.float()
        
        # Normalize to [0, 1]
        if self.normalize:
            image_t = image_t / 255.0
        
        # Verify final shape
        if image_t.shape != (3, self.image_size, self.image_size):
            raise ValueError(f"Final image shape mismatch: {image_t.shape}, expected (3, {self.image_size}, {self.image_size})")
        
        return image_t


def enhanced_vae_loss(x_recon: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor, 
                     beta: float = 1.0,
                     free_bits: float = 2.0,
                     use_perceptual: bool = True,
                     perceptual_weight: float = 0.1,
                     l1_weight: float = 0.7,
                     mse_weight: float = 0.3,
                     perceptual_loss_fn: nn.Module = None,
                     debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Enhanced VAE loss with perceptual loss
    - L1 + MSE reconstruction loss
    - Perceptual loss (VGG features)
    - KL divergence with free bits (per-dimension)
    
    Fixed KL loss implementation:
    - Use per-dimension free bits (not total free bits)
    - Clamp each dimension's KL to minimum free_bits/D
    """
    # Rescale decoder output from [-1, 1] to [0, 1] if using Tanh
    # Check if output is in [-1, 1] range (Tanh) or [0, 1] range (Sigmoid)
    if x_recon.min() < 0:
        # Tanh output: rescale to [0, 1]
        x_recon_scaled = (x_recon + 1.0) / 2.0
        x_recon_scaled = torch.clamp(x_recon_scaled, 0, 1)
    else:
        # Sigmoid output: already in [0, 1]
        x_recon_scaled = torch.clamp(x_recon, 0, 1)
    
    # Reconstruction loss: combination of L1 and MSE
    recon_loss_l1 = F.l1_loss(x_recon_scaled, x, reduction='mean')
    recon_loss_mse = F.mse_loss(x_recon_scaled, x, reduction='mean')
    recon_loss_pixel = l1_weight * recon_loss_l1 + mse_weight * recon_loss_mse
    
    # Perceptual loss (if enabled)
    recon_loss_perceptual = torch.tensor(0.0, device=x.device)
    if use_perceptual and perceptual_loss_fn is not None:
        try:
            # Use scaled reconstruction for perceptual loss
            x_recon_for_perceptual = x_recon_scaled
            x_for_perceptual = x
            
            # Check if inputs are valid
            if x_recon_for_perceptual.dim() == 4 and x_recon_for_perceptual.size(1) == 3:
                recon_loss_perceptual = perceptual_loss_fn(x_recon_for_perceptual, x_for_perceptual)
                if debug and torch.isnan(recon_loss_perceptual):
                    print("Warning: Perceptual loss is NaN")
            else:
                if debug:
                    print(f"Warning: Invalid input shape for perceptual loss: {x_recon_for_perceptual.shape}")
        except Exception as e:
            if debug:
                print(f"Warning: Perceptual loss failed: {e}")
            recon_loss_perceptual = torch.tensor(0.0, device=x.device)
    
    # Total reconstruction loss
    recon_loss = recon_loss_pixel + perceptual_weight * recon_loss_perceptual
    
    if beta == 0.0:
        kl_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        total_loss = recon_loss
    else:
        logvar_clamped = torch.clamp(logvar, min=-20.0, max=20.0)
        mu2 = mu.pow(2)
        kl_per_element = -0.5 * (1 + logvar_clamped - mu2 - logvar_clamped.exp())
        if free_bits > 0:
            latent_channels = kl_per_element.size(1)
            spatial_size = kl_per_element.size(2) * kl_per_element.size(3)
            total_elements = latent_channels * spatial_size
            free_bits_per_element = free_bits / total_elements
            kl_per_element_clamped = torch.clamp(kl_per_element, min=free_bits_per_element)
            kl_loss = torch.mean(torch.sum(kl_per_element_clamped.view(kl_per_element_clamped.size(0), -1), dim=1))
        else:
            kl_loss = torch.mean(torch.sum(kl_per_element.view(kl_per_element.size(0), -1), dim=1))
        total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss, recon_loss_perceptual


def save_model(model: EnhancedVAE, path: str):
    """Save model state"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': model.latent_dim,
        'image_size': model.image_size,
        'channels': model.channels,
        'use_skip_connections': model.use_skip_connections,
    }, path)
    print(f"Model saved to {path}")


def load_model(path: str, device: torch.device) -> EnhancedVAE:
    """Load model state"""
    checkpoint = torch.load(path, map_location=device)
    
    # Handle different checkpoint formats
    # Format 1: Saved by save_model() - has direct keys
    if 'latent_dim' in checkpoint:
        latent_dim = checkpoint['latent_dim']
        image_size = checkpoint['image_size']
        channels = checkpoint['channels']
        use_skip_connections = checkpoint.get('use_skip_connections', True)
        state_dict = checkpoint['model_state_dict']
    # Format 2: Saved by torch.save(checkpoint, ...) - has args dict
    elif 'args' in checkpoint:
        args = checkpoint['args']
        latent_dim = args.get('latent_dim', 256)
        image_size = args.get('image_size', 224)
        channels = args.get('channels', 3)
        use_skip_connections = args.get('use_skip_connections', True)
        state_dict = checkpoint['model_state_dict']
    # Format 3: Only state_dict (fallback)
    else:
        # Try to infer from state_dict keys
        latent_dim = 256  # Default for enhanced VAE
        image_size = 224
        channels = 3
        use_skip_connections = True
        state_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
    
    model = EnhancedVAE(
        latent_dim=latent_dim,
        image_size=image_size,
        channels=channels,
        use_skip_connections=use_skip_connections
    )
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Model loaded from {path}")
    print(f"  - Latent dim: {latent_dim}")
    print(f"  - Image size: {image_size}")
    print(f"  - Channels: {channels}")
    print(f"  - Skip connections: {use_skip_connections}")
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = EnhancedVAE(latent_dim=256, use_skip_connections=True).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224).to(device)
    x_recon, mu, logvar, z = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"Latent shape: {z.shape} (convolutional: C×H×W)")
    print(f"Mu shape: {mu.shape} (convolutional: C×H×W)")
    print(f"Logvar shape: {logvar.shape} (convolutional: C×H×W)")
    
    # Test loss
    perceptual_loss_fn = PerceptualLoss().to(device)
    loss, recon_loss, kl_loss, perceptual_loss = enhanced_vae_loss(
        x_recon, x, mu, logvar, beta=1.0, free_bits=2.0,
        use_perceptual=True, perceptual_weight=0.1,
        perceptual_loss_fn=perceptual_loss_fn
    )
    print(f"Total loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"Perceptual loss: {perceptual_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")

