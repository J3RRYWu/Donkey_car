import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
from typing import Tuple, Optional

# Prefer importing as a proper package (repo root on PYTHONPATH).
# Fallback to the older "sys.path insert" behavior for backwards compatibility.
try:
    from vae_recon.vae_model_enhanced import EnhancedVAE, load_model as load_vae_model_224
    from vae_recon.vae_model_64x64 import SimpleVAE64x64, load_model_64x64
    HAS_VAE_224 = True
    HAS_VAE_64 = True
except Exception:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vae_recon'))
        from vae_model_enhanced import EnhancedVAE, load_model as load_vae_model_224
        from vae_model_64x64 import SimpleVAE64x64, load_model_64x64
        HAS_VAE_224 = True
        HAS_VAE_64 = True
    except ImportError:
        print("Warning: Could not import VAE models from vae_recon. Will use default encoder/decoder.")
        EnhancedVAE = None
        SimpleVAE64x64 = None
        load_vae_model_224 = None
        load_model_64x64 = None
        HAS_VAE_224 = False
        HAS_VAE_64 = False

class VAEPredictor(nn.Module):
    """
    VAE Predictor for trajectory prediction
    - Uses frozen VAE encoder/decoder from vae_recon
    - LSTM Predictor: 15 frames -> predict next frame
    - Encoder: Image -> Convolutional latent (B, latent_dim, H, W)
      where H=W=7 for 224x224 images, H=W=4 for 64x64 images
    - Predictor: LSTM on flattened latent -> next latent
    - Decoder: Latent -> Image
    """
    
    def __init__(self, latent_dim: int = 256, image_size: int = 224, channels: int = 3,
                 action_dim: int = 0, predictor_type: str = "lstm", hidden_size: int = 256,
                 residual_prediction: bool = True, vae_model_path: Optional[str] = None,
                 freeze_vae: bool = True):
        super(VAEPredictor, self).__init__()
        self.latent_dim = latent_dim  # Now refers to latent channels (convolutional)
        self.image_size = image_size
        self.channels = channels
        self.action_dim = action_dim
        self.predictor_type = predictor_type.lower()
        self.hidden_size = hidden_size
        self.residual_prediction = residual_prediction
        self.freeze_vae = freeze_vae
        
        # Load VAE encoder and decoder if path provided
        if vae_model_path:
            print(f"Loading VAE model from {vae_model_path}")
            # Detect VAE type by checking checkpoint or image_size
            checkpoint = torch.load(vae_model_path, map_location=torch.device('cpu'))
            
            # Try to detect image size from checkpoint
            detected_image_size = image_size  # Default to provided image_size
            if 'args' in checkpoint:
                args = checkpoint['args']
                if 'image_size' in args:
                    detected_image_size = args['image_size']
            elif 'image_size' in checkpoint:
                detected_image_size = checkpoint['image_size']
            
            # Load appropriate VAE model
            if detected_image_size == 64 and HAS_VAE_64 and load_model_64x64 is not None:
                print(f"Detected 64x64 VAE model")
                vae_model = load_model_64x64(vae_model_path, torch.device('cpu'))
                self.latent_spatial_size = 4  # 4×4 spatial size for 64x64
                self.image_size = 64  # Update image size
            elif detected_image_size == 224 and HAS_VAE_224 and load_vae_model_224 is not None:
                print(f"Detected 224x224 VAE model")
                vae_model = load_vae_model_224(vae_model_path, torch.device('cpu'))
                self.latent_spatial_size = 7  # 7×7 spatial size for 224x224
                self.image_size = 224  # Update image size
            else:
                # Fallback: try to load 224x224 first, then 64x64
                if HAS_VAE_224 and load_vae_model_224 is not None:
                    try:
                        vae_model = load_vae_model_224(vae_model_path, torch.device('cpu'))
                        self.latent_spatial_size = 7
                        self.image_size = 224
                        print("Loaded as 224x224 VAE (fallback)")
                    except:
                        if HAS_VAE_64 and load_model_64x64 is not None:
                            vae_model = load_model_64x64(vae_model_path, torch.device('cpu'))
                            self.latent_spatial_size = 4
                            self.image_size = 64
                            print("Loaded as 64x64 VAE (fallback)")
                        else:
                            raise ValueError(f"Could not load VAE model from {vae_model_path}")
                elif HAS_VAE_64 and load_model_64x64 is not None:
                    vae_model = load_model_64x64(vae_model_path, torch.device('cpu'))
                    self.latent_spatial_size = 4
                    self.image_size = 64
                    print("Loaded as 64x64 VAE (fallback)")
                else:
                    raise ValueError(f"Could not load VAE model from {vae_model_path}")
            
            # Extract encoder and decoder from VAE model
            self.vae_encoder = vae_model
            self.vae_decoder = vae_model  # Same model has both encode and decode
            
            # Get latent_dim from VAE model
            self.latent_dim = vae_model.latent_dim
            
            # Freeze VAE if requested
            if self.freeze_vae:
                for param in self.vae_encoder.parameters():
                    param.requires_grad = False
                print("VAE encoder/decoder frozen")
        else:
            # Fallback: use default encoder/decoder (not recommended)
            print("Warning: Using default encoder/decoder (VAE not loaded)")
            self.vae_encoder = None
            self.vae_decoder = None
            self.latent_spatial_size = 1
            self.latent_channels = latent_dim
            self.latent_flat_dim = latent_dim
            
            # Default encoder (will not be used if VAE is loaded)
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.fc_mu = nn.Linear(512, latent_dim)
            self.fc_logvar = nn.Linear(512, latent_dim)
            
            # Default decoder (will not be used if VAE is loaded)
            self.dec_size = 64
            self.dec_fc = nn.Linear(latent_dim, 128 * 4 * 4)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 3, kernel_size=1)
            )
        
        # Get actual latent dimension from VAE (convolutional: C×H×W)
        if self.vae_encoder is not None:
            # Convolutional latent: (B, latent_dim, H, W) where latent_dim is channels
            self.latent_channels = vae_model.latent_dim  # Number of channels
            self.latent_flat_dim = self.latent_channels * self.latent_spatial_size * self.latent_spatial_size
        else:
            self.latent_channels = latent_dim
            self.latent_flat_dim = latent_dim
        
        # Predictor modules: LSTM, GRU, or MLP
        predictor_input_dim = self.latent_flat_dim + action_dim
        if self.predictor_type == "lstm":
            # LSTM dynamics: (B,T,input)->(B,T,H) then project to latent
            self.lstm = nn.LSTM(input_size=predictor_input_dim,
                               hidden_size=hidden_size,
                               num_layers=2,  # 2 layers for better capacity
                               batch_first=True,
                               dropout=0.2 if self.predictor_type == "lstm" else 0.0)
            self.dropout_pred = nn.Dropout(0.2)
            self.lstm_out = nn.Linear(hidden_size, self.latent_flat_dim)
        elif self.predictor_type == "gru":
            # GRU dynamics (backward compatibility)
            self.gru = nn.GRU(input_size=predictor_input_dim,
                               hidden_size=hidden_size,
                               num_layers=1,
                               batch_first=True)
            self.dropout_pred = nn.Dropout(0.2)
            self.gru_out = nn.Linear(hidden_size, self.latent_flat_dim)
        else:
            # Default MLP per-step
            self.predictor = nn.Sequential(
                nn.Linear(predictor_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, self.latent_flat_dim)
            )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image to latent space
        Returns: mu, logvar
        If using VAE: mu/logvar are (B, C, H, W) where H=W=7 for 224x224, H=W=4 for 64x64
        Otherwise: mu/logvar are (B, D) (vector)
        """
        if self.vae_encoder is not None:
            # VAE encoder: returns mu, logvar, skip_features
            # 64x64 VAE returns empty list for skip_features, 224x224 returns actual features
            mu, logvar, skip_features = self.vae_encoder.encode(x)
            # Store skip_features if needed (for 224x224 models)
            if skip_features and len(skip_features) > 0:
                self._last_skip_features = skip_features
            else:
                self._last_skip_features = None
            return mu, logvar
        else:
            # Default encoder
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick
        mu, logvar: (B, C, H, W) for VAE (H=W=7 for 224x224, H=W=4 for 64x64) or (B, D) for default
        Returns: z with same shape
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode_images(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to images. 
        If using VAE: 
          - 224x224: z is (..., C, 7, 7) -> (..., 3, 224, 224)
          - 64x64: z is (..., C, 4, 4) -> (..., 3, 64, 64)
        Otherwise: z is (..., D) -> (..., 3, 64, 64)
        """
        if self.vae_decoder is not None:
            # VAE decoder: z is convolutional
            # 64x64 VAE doesn't use skip_features, 224x224 can use None for prediction
            if z.dim() == 4:
                # (B, C, H, W) where H=W=7 for 224x224, H=W=4 for 64x64
                x_recon = self.vae_decoder.decode(z, skip_features=None)  # No skip features for prediction
                return x_recon
            elif z.dim() == 5:
                # (B, T, C, H, W)
                B, T = z.shape[:2]
                z_flat = z.reshape(B * T, *z.shape[2:])
                x_flat = self.vae_decoder.decode(z_flat, skip_features=None)
                x_recon = x_flat.reshape(B, T, *x_flat.shape[1:])
                return x_recon
            else:
                # Flatten and reshape
                original_shape = z.shape
                z_flat = z.view(-1, *z.shape[-3:])  # (N, C, H, W)
                x_flat = self.vae_decoder.decode(z_flat, skip_features=None)
                x_recon = x_flat.view(*original_shape[:-3], *x_flat.shape[1:])
                return x_recon
        else:
            # Default decoder
            z_flat = z.view(-1, z.size(-1))
            x = self.dec_fc(z_flat)
            x = x.view(-1, 128, 4, 4)
            x = self.decoder(x)
            actual_size = x.size(-1)
            x = x.view(*z.shape[:-1], 3, actual_size, actual_size)
            return x
    
    def predict(self, z: torch.Tensor, a: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict next latent state
        
        Input z: 
        - If VAE: (B, C, H, W) or (B, T, C, H, W) - convolutional latent
          where H=W=7 for 224x224 images, H=W=4 for 64x64 images
        - Otherwise: (B, D) or (B, T, D) - vector latent
        
        Returns: predicted latent with same shape as input
        """
        # Flatten convolutional latent to vector for LSTM/MLP
        original_shape = z.shape
        is_sequence = z.dim() == 5 if self.vae_encoder else z.dim() == 3
        
        if self.vae_encoder is not None:
            # Convolutional latent: (B, C, H, W) or (B, T, C, H, W)
            if z.dim() == 4:
                # (B, C, H, W) -> (B, C*H*W)
                z_flat = z.view(z.size(0), -1)
            elif z.dim() == 5:
                # (B, T, C, H, W) -> (B, T, C*H*W)
                B, T = z.shape[:2]
                z_flat = z.view(B, T, -1)
            else:
                z_flat = z.view(z.size(0), -1)
        else:
            # Vector latent: already flat
            z_flat = z
        
        # Remember input for residual connection
        original_input_flat = z_flat
        
        # Concatenate actions if provided
        if a is not None:
            if z_flat.dim() == 2:
                # (B, D) + (B, A) -> (B, D+A)
                z_flat = torch.cat([z_flat, a], dim=-1)
            elif z_flat.dim() == 3:
                # (B, T, D) + (B, T, A) -> (B, T, D+A)
                z_flat = torch.cat([z_flat, a], dim=-1)
        
        # Predict using LSTM, GRU, or MLP
        if self.predictor_type == "lstm":
            if z_flat.dim() == 2:
                z_flat = z_flat.unsqueeze(1)  # (B, D+A) -> (B, 1, D+A)
            out, _ = self.lstm(z_flat)
            out = self.dropout_pred(out)
            out = self.lstm_out(out)
            # Return last step if single step, else sequence
            out = out[:, -1, :] if out.size(1) == 1 else out
        elif self.predictor_type == "gru":
            if z_flat.dim() == 2:
                z_flat = z_flat.unsqueeze(1)
            out, _ = self.gru(z_flat)
            out = self.dropout_pred(out)
            out = self.gru_out(out)
            out = out[:, -1, :] if out.size(1) == 1 else out
        else:
            # MLP
            out = self.predictor(z_flat)
        
        # Residual connection: z_next = z + f(z,a)
        if self.residual_prediction:
            # Strip actions if present
            base = original_input_flat
            if base.size(-1) > self.latent_flat_dim:
                base = base[..., :self.latent_flat_dim]
            
            if out.dim() == 3:  # (B, T, D) - sequence prediction
                if base.dim() == 3 and base.size(1) == out.size(1):
                    out = out + base
                elif base.dim() == 2:
                    B = out.size(0)
                    T = out.size(1)
                    base_seq = base.unsqueeze(1).expand(B, T, -1)
                    out = out + base_seq
            elif out.dim() == 2:  # (B, D) - single step
                if base.dim() == 3 and base.size(1) == 1:
                    base = base[:, 0, :]
                out = out + base
        
        # Reshape back to convolutional format if using VAE
        if self.vae_encoder is not None:
            if len(original_shape) == 4:
                # (B, C*H*W) -> (B, C, H, W) where H=W=7 for 224x224, H=W=4 for 64x64
                out = out.view(original_shape)
            elif len(original_shape) == 5:
                # (B, T, C*H*W) -> (B, T, C, H, W)
                out = out.view(original_shape)
        
        return out
    
    def forward(self, x: torch.Tensor, a: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode and predict.
        If x is (B,3,H,W) -> single step. If x is (B,T,3,H,W) -> sequence.
        H=W=224 for 224x224 VAE, H=W=64 for 64x64 VAE.
        Returns: z, z_pred, mu, logvar
        """
        if x.dim() == 5:
            # Sequence: (B, T, 3, H, W) where H=W=224 for 224x224, H=W=64 for 64x64
            B, T = x.shape[:2]
            x_flat = x.reshape(B*T, *x.shape[2:])
            mu, logvar = self.encode(x_flat)
            z = self.reparameterize(mu, logvar)
            
            # Reshape for sequence prediction
            if self.vae_encoder is not None:
                # Convolutional: (B*T, C, H, W) -> (B, T, C, H, W) where H=W=7 for 224x224, H=W=4 for 64x64
                z = z.reshape(B, T, *z.shape[1:])
                mu = mu.reshape(B, T, *mu.shape[1:])
                logvar = logvar.reshape(B, T, *logvar.shape[1:])
            else:
                # Vector: (B*T, D) -> (B, T, D)
                z = z.reshape(B, T, -1)
                mu = mu.reshape(B, T, -1)
                logvar = logvar.reshape(B, T, -1)
            
            if a is not None and a.dim() == 3:
                # a: (B,T,action_dim)
                z_pred = self.predict(z, a)
            else:
                z_pred = self.predict(z)
            return z, z_pred, mu, logvar
        else:
            # Single step: (B, 3, H, W)
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            z_pred = self.predict(z, a)
            return z, z_pred, mu, logvar


class TrajectoryDataset(Dataset):
    """Dataset for loading trajectory data from NPZ files lazily (memory-mapped).

    Avoids loading all frames into RAM. Sequences do not cross file boundaries.
    """
    
    def __init__(self, npz_paths: list, sequence_length: int = 16, 
                 image_size: int = 224, normalize: bool = True):
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.normalize = normalize
        
        # Hold per-file memory-mapped arrays and lengths
        self.files = []  # list of dicts with frames, states, actions, length, path
        self.seq_counts = []  # available starting positions per file
        self.cum_seq = []  # cumulative counts for global indexing
        cum = 0
        
        for npz_path in npz_paths:
            if not os.path.exists(npz_path):
                continue
            data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
            frames = data['frame']  # (T, 3, H, W), uint8
            states = data['state']  # (T, 4), float32
            actions = data['action']  # (T, 2), float32
            T = len(frames)
            n_seq = max(0, T - self.sequence_length + 1)
            self.files.append({
                'frames': frames,
                'states': states,
                'actions': actions,
                'length': T,
                'path': npz_path,
            })
            self.seq_counts.append(n_seq)
            cum += n_seq
            self.cum_seq.append(cum)
            print(f"Loaded {npz_path}: {T} frames, sequences={n_seq}")
        
        self.total_sequences = self.cum_seq[-1] if self.cum_seq else 0
        print(f"Total sequences: {self.total_sequences}")
        
        # Compute action statistics across all data for normalization (sample-weighted)
        all_actions = []
        for f in self.files:
            a = f['actions']  # memmap (T, 2)
            all_actions.append(a)
        # Concatenate all actions to compute global stats
        if all_actions:
            all_actions_concat = np.concatenate(all_actions, axis=0)
            self.act_mean = np.mean(all_actions_concat, axis=0)
            self.act_std = np.std(all_actions_concat, axis=0) + 1e-6
        else:
            self.act_mean = np.zeros(2, dtype=np.float32)
            self.act_std = np.ones(2, dtype=np.float32)
        print(f"Action normalization stats - mean: {self.act_mean}, std: {self.act_std}")
        
    def __len__(self):
        return self.total_sequences
    
    def _locate(self, global_idx: int):
        # binary search over cumulative sequence counts
        import bisect
        file_idx = bisect.bisect_right(self.cum_seq, global_idx)
        prev_cum = 0 if file_idx == 0 else self.cum_seq[file_idx - 1]
        local_start = global_idx - prev_cum
        return file_idx, local_start
    
    def __getitem__(self, idx):
        if idx < 0:
            idx = self.total_sequences + idx
        file_idx, local_start = self._locate(idx)
        f = self.files[file_idx]
        s = local_start
        e = s + self.sequence_length
        
        # Slice lazily from memmap
        frames_seq = f['frames'][s:e]  # uint8 (16, 3, H, W)
        states_seq = f['states'][s:e]  # (16, 4)
        actions_seq = f['actions'][s:e]  # (16, 2)
        
        # Input: frames 1-15, Target: frames 2-16
        input_frames = frames_seq[:-1]
        target_frames = frames_seq[1:]
        
        # Convert to tensors lazily, normalize at this step to avoid duplicating arrays
        input_frames_t = torch.from_numpy(input_frames.astype(np.float32))
        target_frames_t = torch.from_numpy(target_frames.astype(np.float32))
        if self.normalize:
            input_frames_t = input_frames_t / 255.0
            target_frames_t = target_frames_t / 255.0
        
        # Normalize actions using pre-computed statistics (mean/std across all data)
        actions_t = torch.from_numpy(actions_seq.astype(np.float32))
        mean = torch.from_numpy(self.act_mean.astype(np.float32))
        std = torch.from_numpy(self.act_std.astype(np.float32))
        actions_t = (actions_t - mean) / std
        # Clamp outliers to [-3,3] then scale to [-1,1]
        actions_t = torch.clamp(actions_t, -3.0, 3.0) / 3.0
        
        return {
            'input_frames': input_frames_t,
            'target_frames': target_frames_t,
            'states': torch.from_numpy(states_seq),
            'actions': actions_t
        }


def predictor_loss(z_pred: torch.Tensor, z_target: torch.Tensor, reduction: str = 'mean') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute prediction loss (MSE in latent space)
    z_pred: (B*T, D) or (B, T, D) or (B, T, C, H, W)
    z_target: Same shape as z_pred
    """
    # Flatten if needed
    if z_pred.dim() == 5:
        z_pred_flat = z_pred.reshape(z_pred.size(0) * z_pred.size(1), -1)
        z_target_flat = z_target.reshape(z_target.size(0) * z_target.size(1), -1)
    elif z_pred.dim() == 3:
        z_pred_flat = z_pred.reshape(-1, z_pred.size(-1))
        z_target_flat = z_target.reshape(-1, z_target.size(-1))
    else:
        z_pred_flat = z_pred
        z_target_flat = z_target
    
    mse_loss = F.mse_loss(z_pred_flat, z_target_flat, reduction=reduction)
    return mse_loss, mse_loss


def train_epoch(model: VAEPredictor, dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, device: torch.device,
                beta: float = 1.0,  # Unused when VAE is frozen, kept for compatibility
                scaler: Optional[amp.GradScaler] = None,
                input_noise_std: float = 0.0,
                free_bits_nats: float = 0.0,  # Unused when VAE is frozen, kept for compatibility
                use_actions: bool = False,
                target_jitter_scale: float = 0.0,
                detach_target: bool = True,  # Should be True when VAE is frozen
                open_loop_steps: int = 0,
                open_loop_weight: float = 0.0) -> dict:
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_open_loop = 0.0
    num_open_loop_batches = 0  # Count batches where open-loop loss was computed
    num_batches = 0
    agg_mu_mean = 0.0
    agg_mu_std  = 0.0
    agg_lv_mean = 0.0
    agg_lv_std  = 0.0
    raw_kl_sum = 0.0
    
    for batch in dataloader:
        input_frames = batch['input_frames'].to(device)  # (B, 15, 3, H, W)
        target_frames = batch['target_frames'].to(device)  # (B, 15, 3, H, W)
        B, T = input_frames.shape[:2]
        # Optional input noise
        if input_noise_std > 0:
            input_frames = torch.clamp(
                input_frames + torch.randn_like(input_frames) * float(input_noise_std), 0.0, 1.0
            )
        # Prepare actions sequence if using them
        actions_seq = None
        if use_actions and 'actions' in batch:
            # actions for t=1..15 to predict 2..16
            actions_seq = batch['actions'][:, :-1, :].to(device)  # (B,15,action_dim)
        
        optimizer.zero_grad(set_to_none=True)

        # Initialize open_loop_loss outside AMP context
        open_loop_loss = torch.tensor(0.0, device=device)
        
        use_amp = scaler is not None and device.type == 'cuda'
        if use_amp:
            with amp.autocast('cuda'):
                # Encode full sequences (avoid calling forward/predict)
                B, T = input_frames.shape[:2]
                in_flat = input_frames.reshape(B*T, *input_frames.shape[2:])
                mu_in, lv_in = model.encode(in_flat)
                z_input = mu_in    # 不采样，直接用均值
                
                # Reshape based on whether using VAE (convolutional) or default (vector)
                if model.vae_encoder is not None:
                    # Convolutional: (B*T, C, H, W) -> (B, T, C, H, W)
                    z_input = z_input.reshape(B, T, *z_input.shape[1:])
                    mu = mu_in.reshape(B, T, *mu_in.shape[1:])
                    logvar = lv_in.reshape(B, T, *lv_in.shape[1:])
                else:
                    # Vector: (B*T, D) -> (B, T, D)
                    z_input = z_input.reshape(B, T, -1)
                    mu = mu_in.reshape(B, T, -1)
                    logvar = lv_in.reshape(B, T, -1)
                # Sequence prediction (optionally apply action dropout)
                if model.predictor_type in ['lstm', 'gru']:
                    # Apply action dropout if needed
                    if actions_seq is not None and hasattr(model, '_act_drop') and model._act_drop > 0:
                        mask = (torch.rand_like(actions_seq[..., :1]) > float(model._act_drop)).float()
                        actions_seq = actions_seq * mask
                    # predict() handles flattening and concatenation internally
                    z_pred_seq = model.predict(z_input, actions_seq)  # (B, T, C, H, W) or (B, T, D)
                else:
                    # MLP predictor (predict() handles flattening internally)
                    if actions_seq is not None and hasattr(model, '_act_drop') and model._act_drop > 0:
                        mask = (torch.rand_like(actions_seq[..., :1]) > float(model._act_drop)).float()
                        actions_seq = actions_seq * mask
                    # predict() handles flattening and reshaping for both vector and convolutional latents
                    z_pred_seq = model.predict(z_input, actions_seq)
                # Targets: encode target frames
                # Encode targets directly (avoid extra predictor compute)
                B, T = target_frames.shape[:2]
                tf_flat = target_frames.reshape(B*T, *target_frames.shape[2:])
                if detach_target:
                    with torch.no_grad():
                        mu_t, logvar_t = model.encode(tf_flat)
                else:
                    mu_t, logvar_t = model.encode(tf_flat)
                
                # Reshape based on whether using VAE (convolutional) or default (vector)
                if model.vae_encoder is not None:
                    # Convolutional: (B*T, C, H, W) -> (B, T, C, H, W)
                    mu_t = mu_t.reshape(B, T, *mu_t.shape[1:])
                    logvar_t = logvar_t.reshape(B, T, *logvar_t.shape[1:])
                else:
                    # Vector: (B*T, D) -> (B, T, D)
                    mu_t = mu_t.reshape(B, T, -1)
                    logvar_t = logvar_t.reshape(B, T, -1)
                # Target: use mu directly (no jitter needed since encoder is frozen)
                z_target_seq = mu_t
                
                # Compute loss over sequence (FP32)
                # Flatten for loss computation
                if model.vae_encoder is not None:
                    # Convolutional: flatten (B, T, C, H, W) -> (B*T, C*H*W)
                    z_pred_flat = z_pred_seq.float().reshape(-1, z_pred_seq.shape[2] * z_pred_seq.shape[3] * z_pred_seq.shape[4])
                    z_target_flat = z_target_seq.float().reshape(-1, z_target_seq.shape[2] * z_target_seq.shape[3] * z_target_seq.shape[4])
                else:
                    # Vector: (B, T, D) -> (B*T, D)
                    z_pred_flat = z_pred_seq.float().reshape(-1, z_pred_seq.size(-1))
                    z_target_flat = z_target_seq.float().reshape(-1, z_target_seq.size(-1))
                
                # Simple MSE loss (VAE encoder/decoder are frozen)
                # Only MSE: predicted latent vs target latent (NOT image reconstruction loss)
                loss, mse_loss = predictor_loss(z_pred_flat, z_target_flat)
                recon_loss = mse_loss
                kl_loss = torch.tensor(0.0, device=z_pred_flat.device)  # No KL loss when encoder is frozen
                
                # Open-loop rollout loss (multi-step prediction)
                if open_loop_steps > 0 and open_loop_weight > 0:
                    # Get first frame's latent for open-loop rollout
                    z_start = z_input[:, 0, ...]  # (B, C, H, W) or (B, D)
                    
                    # Get target latents for open-loop steps
                    rollout_steps = min(open_loop_steps, T - 1)
                    
                    if rollout_steps > 0:
                        # Encode target frames for open-loop steps (if we have enough frames)
                        if target_frames.shape[1] >= rollout_steps + 1:
                            # Get target frames for rollout: frames 1 to rollout_steps+1
                            target_rollout_frames = target_frames[:, :rollout_steps+1, ...]  # (B, rollout_steps+1, 3, H, W)
                            target_rollout_flat = target_rollout_frames.reshape(B*(rollout_steps+1), *target_rollout_frames.shape[2:])
                            
                            # Encode target frames (detached, no grad)
                            with torch.no_grad():
                                mu_rollout_target, _ = model.encode(target_rollout_flat)
                            
                            # Reshape target latents
                            if model.vae_encoder is not None:
                                mu_rollout_target = mu_rollout_target.reshape(B, rollout_steps+1, *mu_rollout_target.shape[1:])
                            else:
                                mu_rollout_target = mu_rollout_target.reshape(B, rollout_steps+1, -1)
                            
                            # Open-loop rollout: start from z_start, predict step by step
                            z_rollout = z_start  # (B, C, H, W) or (B, D)
                            rollout_losses = []
                            
                            # Get actions for rollout if available
                            actions_rollout = None
                            if actions_seq is not None and actions_seq.shape[1] >= rollout_steps:
                                actions_rollout = actions_seq[:, :rollout_steps, :]  # (B, rollout_steps, action_dim)
                            
                            for step in range(rollout_steps):
                                # Predict next latent
                                if actions_rollout is not None:
                                    # Single step prediction
                                    if z_rollout.dim() == 4:  # Convolutional: (B, C, H, W)
                                        z_rollout_expanded = z_rollout.unsqueeze(1)  # (B, 1, C, H, W)
                                    else:  # Vector: (B, D)
                                        z_rollout_expanded = z_rollout.unsqueeze(1)  # (B, 1, D)
                                    
                                    a_step = actions_rollout[:, step:step+1, :]  # (B, 1, action_dim)
                                    z_next_pred = model.predict(z_rollout_expanded, a_step)  # (B, 1, C, H, W) or (B, 1, D)
                                    z_next_pred = z_next_pred.squeeze(1)  # (B, C, H, W) or (B, D)
                                else:
                                    # Single step prediction without actions
                                    if z_rollout.dim() == 4:  # Convolutional: (B, C, H, W)
                                        z_rollout_expanded = z_rollout.unsqueeze(1)  # (B, 1, C, H, W)
                                    else:  # Vector: (B, D)
                                        z_rollout_expanded = z_rollout.unsqueeze(1)  # (B, 1, D)
                                    
                                    z_next_pred = model.predict(z_rollout_expanded, None)  # (B, 1, C, H, W) or (B, 1, D)
                                    z_next_pred = z_next_pred.squeeze(1)  # (B, C, H, W) or (B, D)
                                
                                # Target latent for this step (frame step+1)
                                z_target_step = mu_rollout_target[:, step+1, ...]  # (B, C, H, W) or (B, D)
                                
                                # Compute MSE loss for this step
                                if model.vae_encoder is not None:
                                    # Convolutional: flatten
                                    z_pred_flat_step = z_next_pred.float().reshape(B, -1)
                                    z_target_flat_step = z_target_step.float().reshape(B, -1)
                                else:
                                    # Vector: already flat
                                    z_pred_flat_step = z_next_pred.float()
                                    z_target_flat_step = z_target_step.float()
                                
                                step_loss = F.mse_loss(z_pred_flat_step, z_target_flat_step, reduction='mean')
                                rollout_losses.append(step_loss)
                                
                                # Use predicted latent for next step (open-loop)
                                z_rollout = z_next_pred.detach()  # Detach to prevent gradient accumulation
                            
                            # Average rollout loss
                            if rollout_losses:
                                open_loop_loss = torch.stack(rollout_losses).mean().to(device)
                        
                        # Add open-loop loss to total loss
                        loss = loss + open_loop_weight * open_loop_loss
                
            scaler.scale(loss).backward()
            # Unscale before clipping under AMP
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Encode (avoid calling forward/predict)
            B, T = input_frames.shape[:2]
            in_flat = input_frames.reshape(B*T, *input_frames.shape[2:])
            mu_in, lv_in = model.encode(in_flat)
            z_input = mu_in    # 不采样，直接用均值
            
            # Reshape based on whether using VAE (convolutional) or default (vector)
            if model.vae_encoder is not None:
                # Convolutional: (B*T, C, H, W) -> (B, T, C, H, W)
                z_input = z_input.reshape(B, T, *z_input.shape[1:])
                mu = mu_in.reshape(B, T, *mu_in.shape[1:])
                logvar = lv_in.reshape(B, T, *lv_in.shape[1:])
            else:
                # Vector: (B*T, D) -> (B, T, D)
                z_input = z_input.reshape(B, T, -1)
                mu = mu_in.reshape(B, T, -1)
                logvar = lv_in.reshape(B, T, -1)
            
            # Predict sequence (optionally apply action dropout)
            if model.predictor_type in ['lstm', 'gru']:
                if actions_seq is not None and hasattr(model, '_act_drop') and model._act_drop > 0:
                    mask = (torch.rand_like(actions_seq[..., :1]) > float(model._act_drop)).float()
                    actions_seq = actions_seq * mask
                # Pass sequences directly to predict which handles residual connection
                z_pred_seq = model.predict(z_input, actions_seq)
            else:
                # MLP predictor (predict() handles flattening internally)
                if actions_seq is not None and hasattr(model, '_act_drop') and model._act_drop > 0:
                    mask = (torch.rand_like(actions_seq[..., :1]) > float(model._act_drop)).float()
                    actions_seq = actions_seq * mask
                # predict() handles flattening and reshaping for both vector and convolutional latents
                z_pred_seq = model.predict(z_input, actions_seq)
            # Targets
            # Encode targets directly (avoid extra predictor compute)
            B, T = target_frames.shape[:2]
            tf_flat = target_frames.reshape(B*T, *target_frames.shape[2:])
            if detach_target:
                with torch.no_grad():
                    mu_t, logvar_t = model.encode(tf_flat)
            else:
                mu_t, logvar_t = model.encode(tf_flat)
            
            # Reshape based on whether using VAE (convolutional) or default (vector)
            if model.vae_encoder is not None:
                # Convolutional: (B*T, C, H, W) -> (B, T, C, H, W)
                mu_t = mu_t.reshape(B, T, *mu_t.shape[1:])
                logvar_t = logvar_t.reshape(B, T, *logvar_t.shape[1:])
            else:
                # Vector: (B*T, D) -> (B, T, D)
                mu_t = mu_t.reshape(B, T, -1)
                logvar_t = logvar_t.reshape(B, T, -1)
            # Target: use mu directly (no jitter needed since encoder is frozen)
            z_target_seq = mu_t
            
            # Loss FP32 - Simple MSE loss
            # Flatten for loss computation
            if model.vae_encoder is not None:
                # Convolutional: flatten (B, T, C, H, W) -> (B*T, C*H*W)
                z_pred_flat = z_pred_seq.float().reshape(-1, z_pred_seq.shape[2] * z_pred_seq.shape[3] * z_pred_seq.shape[4])
                z_target_flat = z_target_seq.float().reshape(-1, z_target_seq.shape[2] * z_target_seq.shape[3] * z_target_seq.shape[4])
            else:
                # Vector: (B, T, D) -> (B*T, D)
                z_pred_flat = z_pred_seq.float().reshape(-1, z_pred_seq.size(-1))
                z_target_flat = z_target_seq.float().reshape(-1, z_target_seq.size(-1))
            
            # Simple MSE loss (VAE encoder/decoder are frozen)
            # Only MSE loss, no auxiliary losses needed
            loss, mse_loss = predictor_loss(z_pred_flat, z_target_flat)
            recon_loss = mse_loss
            kl_loss = torch.tensor(0.0, device=z_pred_flat.device)  # No KL loss when encoder is frozen
            
            # Open-loop rollout loss (multi-step prediction) - same as AMP branch
            if open_loop_steps > 0 and open_loop_weight > 0:
                # Get first frame's latent for open-loop rollout
                z_start = z_input[:, 0, ...]  # (B, C, H, W) or (B, D)
                
                # Get target latents for open-loop steps
                rollout_steps = min(open_loop_steps, T - 1)
                
                if rollout_steps > 0:
                    # Encode target frames for open-loop steps (if we have enough frames)
                    if target_frames.shape[1] >= rollout_steps + 1:
                        # Get target frames for rollout: frames 1 to rollout_steps+1
                        target_rollout_frames = target_frames[:, :rollout_steps+1, ...]  # (B, rollout_steps+1, 3, H, W)
                        target_rollout_flat = target_rollout_frames.reshape(B*(rollout_steps+1), *target_rollout_frames.shape[2:])
                        
                        # Encode target frames (detached, no grad)
                        with torch.no_grad():
                            mu_rollout_target, _ = model.encode(target_rollout_flat)
                        
                        # Reshape target latents
                        if model.vae_encoder is not None:
                            mu_rollout_target = mu_rollout_target.reshape(B, rollout_steps+1, *mu_rollout_target.shape[1:])
                        else:
                            mu_rollout_target = mu_rollout_target.reshape(B, rollout_steps+1, -1)
                        
                        # Open-loop rollout: start from z_start, predict step by step
                        z_rollout = z_start  # (B, C, H, W) or (B, D)
                        rollout_losses = []
                        
                        # Get actions for rollout if available
                        actions_rollout = None
                        if actions_seq is not None and actions_seq.shape[1] >= rollout_steps:
                            actions_rollout = actions_seq[:, :rollout_steps, :]  # (B, rollout_steps, action_dim)
                        
                        for step in range(rollout_steps):
                            # Predict next latent
                            if actions_rollout is not None:
                                # Single step prediction
                                if z_rollout.dim() == 4:  # Convolutional: (B, C, H, W)
                                    z_rollout_expanded = z_rollout.unsqueeze(1)  # (B, 1, C, H, W)
                                else:  # Vector: (B, D)
                                    z_rollout_expanded = z_rollout.unsqueeze(1)  # (B, 1, D)
                                
                                a_step = actions_rollout[:, step:step+1, :]  # (B, 1, action_dim)
                                z_next_pred = model.predict(z_rollout_expanded, a_step)  # (B, 1, C, H, W) or (B, 1, D)
                                z_next_pred = z_next_pred.squeeze(1)  # (B, C, H, W) or (B, D)
                            else:
                                # Single step prediction without actions
                                if z_rollout.dim() == 4:  # Convolutional: (B, C, H, W)
                                    z_rollout_expanded = z_rollout.unsqueeze(1)  # (B, 1, C, H, W)
                                else:  # Vector: (B, D)
                                    z_rollout_expanded = z_rollout.unsqueeze(1)  # (B, 1, D)
                                
                                z_next_pred = model.predict(z_rollout_expanded, None)  # (B, 1, C, H, W) or (B, 1, D)
                                z_next_pred = z_next_pred.squeeze(1)  # (B, C, H, W) or (B, D)
                            
                            # Target latent for this step (frame step+1)
                            z_target_step = mu_rollout_target[:, step+1, ...]  # (B, C, H, W) or (B, D)
                            
                            # Compute MSE loss for this step
                            if model.vae_encoder is not None:
                                # Convolutional: flatten
                                z_pred_flat_step = z_next_pred.float().reshape(B, -1)
                                z_target_flat_step = z_target_step.float().reshape(B, -1)
                            else:
                                # Vector: already flat
                                z_pred_flat_step = z_next_pred.float()
                                z_target_flat_step = z_target_step.float()
                            
                            step_loss = F.mse_loss(z_pred_flat_step, z_target_flat_step, reduction='mean')
                            rollout_losses.append(step_loss)
                            
                            # Use predicted latent for next step (open-loop)
                            z_rollout = z_next_pred.detach()  # Detach to prevent gradient accumulation
                        
                        # Average rollout loss
                        if rollout_losses:
                            open_loop_loss = torch.stack(rollout_losses).mean()
                    
                    # Add open-loop loss to total loss
                    loss = loss + open_loop_weight * open_loop_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        # Accumulate open-loop loss if it was computed
        if open_loop_steps > 0 and open_loop_weight > 0:
            # open_loop_loss is initialized to 0.0, but will be > 0 if computed
            open_loop_val = open_loop_loss.item()
            if open_loop_val > 1e-8:  # Only count if actually computed (not just initialized)
                total_open_loop += open_loop_val
                num_open_loop_batches += 1
        num_batches += 1

        # Track encoder statistics (per-batch)
        # Flatten for statistics computation
        if model.vae_encoder is not None:
            mu_flat = mu.float().reshape(-1, mu.shape[2] * mu.shape[3] * mu.shape[4]) if mu.dim() == 5 else mu.float().reshape(-1, -1)
            lv_flat = logvar.float().reshape(-1, logvar.shape[2] * logvar.shape[3] * logvar.shape[4]) if logvar.dim() == 5 else logvar.float().reshape(-1, -1)
        else:
            mu_flat = mu.float().reshape(-1, mu.size(-1)) if mu.dim() == 3 else mu.float()
            lv_flat = logvar.float().reshape(-1, logvar.size(-1)) if logvar.dim() == 3 else logvar.float()
        
        agg_mu_mean += float(mu_flat.mean().item())
        agg_mu_std  += float(mu_flat.std().item())
        agg_lv_mean += float(lv_flat.mean().item())
        agg_lv_std  += float(lv_flat.std().item())

        # KL loss is 0 when encoder is frozen (for compatibility with return dict)
        raw_kl_sum += 0.0
    
    # Compute average open-loop loss only from batches where it was computed
    avg_open_loop = total_open_loop / num_open_loop_batches if num_open_loop_batches > 0 else 0.0
    
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon / num_batches,
        'kl_loss': total_kl / num_batches,
        'open_loop_loss': avg_open_loop,
        'raw_kl': raw_kl_sum / num_batches,
        'mu_mean': agg_mu_mean / num_batches,
        'mu_std':  agg_mu_std / num_batches,
        'logvar_mean': agg_lv_mean / num_batches,
        'logvar_std':  agg_lv_std / num_batches
    }


def validate_epoch(model: VAEPredictor, dataloader: DataLoader, device: torch.device) -> dict:
    """Validate one epoch"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_frames = batch['input_frames'].to(device)
            target_frames = batch['target_frames'].to(device)
            
            actions_seq = None
            if 'actions' in batch and model.action_dim > 0:
                actions_seq = batch['actions'][:, :-1, :].to(device)
            
            # Forward pass
            z_seq, z_pred_seq, mu_seq, logvar_seq = model(input_frames, actions_seq)
            
            # Encode target frames
            B, T = target_frames.shape[:2]
            target_flat = target_frames.reshape(B*T, *target_frames.shape[2:])
            mu_target, _ = model.encode(target_flat)
            
            if model.vae_encoder is not None:
                z_target = mu_target.reshape(B, T, *mu_target.shape[1:])
            else:
                z_target = mu_target.reshape(B, T, -1)
            
            # Compute loss
            loss, mse_loss = predictor_loss(z_pred_seq, z_target)
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'mse': avg_mse,
    }


def save_model(model: VAEPredictor, path: str, epoch: int = 0, optimizer: Optional[torch.optim.Optimizer] = None,
               best_loss: float = float('inf'), train_history: Optional[dict] = None, args: Optional[dict] = None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'latent_dim': model.latent_dim,
        'image_size': model.image_size,
        'channels': model.channels,
        'action_dim': getattr(model, 'action_dim', 0),
        'predictor_type': model.predictor_type,
        'hidden_size': model.hidden_size,
        'residual_prediction': model.residual_prediction,
        'best_loss': best_loss,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if train_history is not None:
        checkpoint['train_history'] = train_history
    
    if args is not None:
        checkpoint['args'] = args
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path: str, device: torch.device, vae_model_path_override: Optional[str] = None,
               freeze_vae_override: Optional[bool] = None) -> VAEPredictor:
    """Load model from checkpoint.

    Notes:
    - Some older predictor checkpoints may not store `vae_model_path`. In that case, pass
      `vae_model_path_override` (recommended) so the model can be constructed with the correct
      convolutional latent shape (e.g., 64x4x4).
    - If checkpoint tensors have shape mismatches (e.g., due to different latent size),
      we will skip mismatched keys and load what we can.
    """
    checkpoint = torch.load(path, map_location=device)
    
    # Parse checkpoint format
    if 'latent_dim' in checkpoint:
        # Format 1: Direct checkpoint
        latent_dim = checkpoint['latent_dim']
        image_size = checkpoint.get('image_size', 224)
        channels = checkpoint.get('channels', 3)
        action_dim = checkpoint.get('action_dim', 0)
        predictor_type = checkpoint.get('predictor_type', 'lstm')
        hidden_size = checkpoint.get('hidden_size', 256)
        residual_prediction = checkpoint.get('residual_prediction', True)
        vae_model_path = checkpoint.get('vae_model_path', None)
        freeze_vae = checkpoint.get('freeze_vae', True)
        state_dict = checkpoint['model_state_dict']
    elif 'args' in checkpoint:
        # Format 2: Training checkpoint with args dict
        args = checkpoint['args']
        latent_dim = args.get('latent_dim', 256)  # Default 256 if using VAE
        image_size = args.get('image_size', 224)  # Default for VAE
        channels = 3  # Default
        action_dim = 2 if args.get('use_actions', False) else 0
        predictor_type = args.get('predictor', 'lstm')
        hidden_size = args.get('hidden_size', 256)
        residual_prediction = args.get('residual_prediction', False)  # Default False based on our changes
        vae_model_path = args.get('vae_model_path', None)
        freeze_vae = args.get('freeze_vae', True)
        state_dict = checkpoint['model_state_dict']
    else:
        raise ValueError(f"Unknown checkpoint format in {path}. Keys: {list(checkpoint.keys())}")

    # Apply overrides if provided
    if vae_model_path_override is not None:
        vae_model_path = vae_model_path_override
    if freeze_vae_override is not None:
        freeze_vae = freeze_vae_override
    
    model = VAEPredictor(
        latent_dim=latent_dim,
        image_size=image_size,
        channels=channels,
        action_dim=action_dim,
        predictor_type=predictor_type,
        hidden_size=hidden_size,
        residual_prediction=residual_prediction,
        vae_model_path=vae_model_path,
        freeze_vae=freeze_vae
    )

    # Robust load: skip mismatched shapes (strict=False does NOT ignore shape mismatch).
    try:
        load_res = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"[load_model] Warning: state_dict shape mismatch, will skip incompatible keys.\n  {e}")
        model_sd = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape") and v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        load_res = model.load_state_dict(filtered, strict=False)
        print(f"[load_model] Loaded {len(filtered)} keys, skipped {len(skipped)} keys due to mismatch.")

    if getattr(load_res, "missing_keys", None):
        if load_res.missing_keys:
            print(f"  [load_model] Missing keys: {len(load_res.missing_keys)}")
    if getattr(load_res, "unexpected_keys", None):
        if load_res.unexpected_keys:
            print(f"  [load_model] Unexpected keys: {len(load_res.unexpected_keys)}")
    model.to(device)
    print(f"Model loaded from {path}")
    print(f"  - Latent dim: {latent_dim}")
    print(f"  - Predictor type: {predictor_type}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Residual prediction: {residual_prediction}")
    print(f"  - Action dim: {action_dim}")
    if vae_model_path:
        print(f"  - VAE model: {vae_model_path}")
        print(f"  - VAE frozen: {freeze_vae}")
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test with VAE (if available)
    vae_path = "../vae_recon/checkpoints_enhanced/best_model.pt"
    if os.path.exists(vae_path):
        print(f"Testing with VAE model: {vae_path}")
        model = VAEPredictor(
            latent_dim=256,
            predictor_type='lstm',
            hidden_size=256,
            vae_model_path=vae_path,
            freeze_vae=True
        ).to(device)
    else:
        print("Testing with default encoder/decoder (VAE not found)")
        model = VAEPredictor(latent_dim=16, predictor_type='lstm').to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  - Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass (single frame)
    x = torch.randn(2, 3, 224, 224).to(device)
    z, z_pred, mu, logvar = model(x)
    print(f"\nSingle frame test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Latent shape: {z.shape}")
    print(f"  Predicted latent shape: {z_pred.shape}")
    print(f"  Mu shape: {mu.shape}")
    print(f"  Logvar shape: {logvar.shape}")

