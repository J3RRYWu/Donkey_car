#!/usr/bin/env python3
"""
Standard visualizations: predictions, rollout images, video generation
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from predictor.core.vae_predictor import VAEPredictor, TrajectoryDataset
from predictor.evaluation.eval_utils import overlay_sbs_labels


def visualize_predictions(model: VAEPredictor, dataloader: DataLoader, device: torch.device,
                          num_samples: int = 3, rollout_steps: int = 15, save_dir: str = './eval_results'):
    """Check 3: Visualize GT, VAE reconstruction, LSTM prediction"""
    print("\n" + "="*70)
    print("Check 3: Visualize Predictions")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        input_frames = batch['input_frames'].to(device)  # (B, 15, 3, 64, 64)
        target_frames = batch['target_frames'].to(device)  # (B, 15, 3, 64, 64)
        B = input_frames.shape[0]
        T_in = input_frames.shape[1]
        target_offset = int(batch.get('target_offset', torch.tensor(1))[0].item()) if isinstance(batch.get('target_offset', 1), torch.Tensor) else int(batch.get('target_offset', 1))
        start_idx = target_offset - 1
        if start_idx < 0 or start_idx >= T_in:
            raise ValueError(f"Bad alignment: need 0 <= target_offset-1 < input_length. "
                             f"Got target_offset={target_offset}, input_length={T_in}.")
        
        # Get actions if available
        actions_seq = None
        if 'actions' in batch and model.action_dim > 0:
            actions_full = batch['actions'].to(device)
            actions_seq = actions_full[:, 0:T_in, :]  # align with rollout start
        
        num_samples = min(num_samples, B)
        
        for sample_idx in range(num_samples):
            print(f"Processing sample {sample_idx + 1}/{num_samples}...")
            
            # Get single sample
            input_seq = input_frames[sample_idx:sample_idx+1]  # (1, 15, 3, 64, 64)
            target_seq = target_frames[sample_idx:sample_idx+1]  # (1, 15, 3, 64, 64)
            
            # Get actions for this sample
            a_seq = None
            if actions_seq is not None:
                a_seq = actions_seq[sample_idx:sample_idx+1]  # (1, 15, action_dim)
            
            # Encode first frame
            first_frame = input_seq[:, start_idx, ...]  # (1, 3, 64, 64)
            mu_start, _ = model.encode(first_frame)
            z_start = mu_start  # (1, C, 4, 4) or (1, D)
            
            # VAE reconstruction of first frame (for reference)
            if model.vae_encoder is not None:
                # Use full VAE forward to get proper reconstruction with skip connections
                vae = model.vae_encoder  # EnhancedVAE instance
                vae.eval()
                with torch.no_grad():
                    recon_first, _, _, _ = vae(first_frame)  # (1, 3, 64, 64)
                # Rescale from [-1,1] to [0,1] if needed
                if recon_first.min() < 0:
                    recon_first = (recon_first + 1.0) / 2.0
                recon_first = torch.clamp(recon_first, 0, 1)
            else:
                recon_first = model.decode_images(z_start)
                if recon_first.min() < 0:
                    recon_first = (recon_first + 1.0) / 2.0
                recon_first = torch.clamp(recon_first, 0, 1)
            
            # Rollout predictions
            predicted_images = []
            z_current = z_start
            
            for step in range(rollout_steps):
                # Predict next latent
                if model.vae_encoder is not None:
                    z_current_expanded = z_current.unsqueeze(1)  # (1, 1, C, 4, 4)
                else:
                    z_current_expanded = z_current.unsqueeze(1)  # (1, 1, D)
                
                # Get action if available
                a_step = None
                if a_seq is not None and a_seq.shape[1] > 0:
                    # Use action aligned to transition (start_idx + step) -> (start_idx + step + 1)
                    action_idx = min(start_idx + step, a_seq.shape[1] - 1)
                    a_step = a_seq[:, action_idx:action_idx+1, :]  # (1, 1, action_dim)
                
                # Predict
                z_next = model.predict(z_current_expanded, a_step)  # (1, 1, C, 4, 4) or (1, 1, D)
                z_next = z_next.squeeze(1)  # (1, C, 4, 4) or (1, D)
                
                # Decode to image (for visualization only)
                # Use GT frame's skip features to decode predicted latent (cheating for visualization)
                if model.vae_encoder is not None:
                    # Get skip features from corresponding GT frame
                    t_for_skip = min(step, target_seq.shape[1] - 1)
                    x_gt_t = target_seq[:, t_for_skip, ...]  # (1, 3, 64, 64)
                    with torch.no_grad():
                        _, _, skip_t = model.vae_encoder.encode(x_gt_t)  # Get encoder's skip features
                    # Decode predicted latent with GT skip features
                    pred_img = model.vae_decoder.decode(z_next, skip_t)  # Use z_next + skip_t
                else:
                    pred_img = model.decode_images(z_next)
                
                # Clamp and convert to numpy
                if pred_img.min() < 0:  # Tanh output
                    pred_img = (pred_img + 1.0) / 2.0
                pred_img = torch.clamp(pred_img, 0, 1)
                predicted_images.append(pred_img[0].cpu().numpy().transpose(1, 2, 0))  # (64, 64, 3)
                
                # Use predicted latent for next step
                z_current = z_next
            
            # Get GT images (target frames)
            gt_images = []
            for t in range(min(rollout_steps, target_seq.shape[1])):
                gt_img = target_seq[:, t, ...].cpu().numpy().transpose(0, 2, 3, 1)[0]  # (64, 64, 3)
                gt_images.append(gt_img)
            
            # Get VAE reconstructions of target frames (for comparison)
            vae_recon_images = []
            
            if model.vae_encoder is not None:
                # Use full VAE forward to get proper reconstruction with skip connections
                vae = model.vae_encoder  # EnhancedVAE instance
                vae.eval()
                with torch.no_grad():
                    # Process each target frame individually to get proper skip connections
                    for t in range(min(rollout_steps, target_seq.shape[1])):
                        x_t = target_seq[:, t, ...]  # (1, 3, 64, 64)
                        recon_t, _, _, _ = vae(x_t)  # Full forward with skip connections
                        
                        # Rescale from [-1,1] to [0,1] if needed
                        if recon_t.min() < 0:
                            recon_t = (recon_t + 1.0) / 2.0
                        recon_t = torch.clamp(recon_t, 0, 1)
                        
                        recon_img = recon_t[0].cpu().numpy().transpose(1, 2, 0)  # (64, 64, 3)
                        vae_recon_images.append(recon_img)
            else:
                # Fallback for non-VAE models
                target_flat = target_seq.reshape(-1, 3, 64, 64)
                mu_target, _ = model.encode(target_flat)
                z_target = mu_target
                recon_target = model.decode_images(z_target)  # (N, 3, 64, 64)
                
                if recon_target.min() < 0:  # Tanh output
                    recon_target = (recon_target + 1.0) / 2.0
                recon_target = torch.clamp(recon_target, 0, 1)
                
                for t in range(min(rollout_steps, target_seq.shape[1])):
                    recon_img = recon_target[t].cpu().numpy().transpose(1, 2, 0)  # (64, 64, 3)
                    vae_recon_images.append(recon_img)
            
            # Create visualization
            num_steps_to_show = min(rollout_steps, 10)  # Show first 10 steps
            fig, axes = plt.subplots(3, num_steps_to_show, figsize=(num_steps_to_show * 2, 6))
            
            if num_steps_to_show == 1:
                axes = axes.reshape(3, 1)
            
            for step in range(num_steps_to_show):
                # Row 1: GT images
                if step < len(gt_images):
                    axes[0, step].imshow(gt_images[step])
                axes[0, step].axis('off')
                if step == 0:
                    axes[0, step].set_title('GT', fontsize=10)
                
                # Row 2: VAE reconstruction
                if step < len(vae_recon_images):
                    axes[1, step].imshow(vae_recon_images[step])
                axes[1, step].axis('off')
                if step == 0:
                    axes[1, step].set_title('VAE Recon', fontsize=10)
                
                # Row 3: LSTM prediction
                if step < len(predicted_images):
                    axes[2, step].imshow(predicted_images[step])
                axes[2, step].axis('off')
                if step == 0:
                    axes[2, step].set_title('LSTM Pred', fontsize=10)
            
            plt.suptitle(f'Sample {sample_idx + 1}: GT vs VAE Recon vs LSTM Prediction (Open-loop)', fontsize=12)
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, f'prediction_sample_{sample_idx + 1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved visualization to {save_path}")


def visualize_rollout_images(model: VAEPredictor, dataset: TrajectoryDataset, device: torch.device, 
                             save_path: str, rollout_steps: int = 30, sample_idx: int = 0):
    """Visualize 30-step rollout: GT vs VAE Recon vs Predictor Rollout"""
    model.eval()
    
    if model.vae_encoder is None:
        print("Warning: VAE encoder not available, skipping rollout visualization")
        return
    
    vae = model.vae_encoder
    vae.eval()
    
    # Get sample from dataset
    sample = dataset[sample_idx]
    input_frames = sample['input_frames'].unsqueeze(0).to(device)  # (1, 15, 3, 64, 64)
    target_frames = sample['target_frames'].unsqueeze(0).to(device)  # (1, 15, 3, 64, 64)
    actions = sample.get('actions', None)
    # Only use actions if the predictor was trained/configured with action_dim > 0.
    if actions is not None and getattr(model, "action_dim", 0) > 0:
        actions = actions.unsqueeze(0).to(device)  # (1, 15, action_dim)
    else:
        actions = None
    
    # Combine input and target frames for full sequence
    all_frames = torch.cat([input_frames, target_frames], dim=1)  # (1, 30, 3, 64, 64)
    T = all_frames.shape[1]
    
    with torch.no_grad():
        # Encode first frame
        first_frame = all_frames[:, 0, ...]  # (1, 3, 64, 64)
        mu0, logvar0, _ = vae.encode(first_frame)
        z0 = vae.reparameterize(mu0, logvar0)  # (1, C, 4, 4)
        
        # Build GT latents
        gt_latents = []
        for t in range(min(rollout_steps, T)):
            frame_t = all_frames[:, t, ...]  # (1, 3, 64, 64)
            mu_t, logvar_t, _ = vae.encode(frame_t)
            z_t = vae.reparameterize(mu_t, logvar_t)
            gt_latents.append(z_t)
        
        # Predictor rollout latents
        pred_latents = []
        z_prev = z0.clone()
        for t in range(rollout_steps):
            # Get action if available (use last available action if t exceeds sequence length)
            a = None
            if actions is not None and actions.shape[1] > 0:
                action_idx = min(t, actions.shape[1] - 1)
                a = actions[:, action_idx:action_idx+1, :]  # (1, 1, action_dim)
            
            # Predict next latent
            z_prev_expanded = z_prev.unsqueeze(1)  # (1, 1, C, 4, 4)
            z_pred = model.predict(z_prev_expanded, a)  # (1, 1, C, 4, 4)
            z_pred = z_pred.squeeze(1)  # (1, C, 4, 4)
            pred_latents.append(z_pred)
            z_prev = z_pred
        
        # Decode to images
        gt_imgs = []
        recon_imgs = []
        pred_imgs = []
        
        for t in range(rollout_steps):
            # Get frame for skip features (use last available frame if t exceeds sequence length)
            frame_idx = min(t, T - 1)
            frame_t = all_frames[:, frame_idx, ...]  # (1, 3, 64, 64)
            _, _, skip_t = vae.encode(frame_t)
            
            # GT image (only if we have GT latent for this step)
            if t < len(gt_latents):
                gt_img = vae.decode(gt_latents[t], skip_t)  # (1, 3, 64, 64)
                if gt_img.min() < 0:
                    gt_img = (gt_img + 1.0) / 2.0
                gt_img = torch.clamp(gt_img, 0, 1)
                gt_imgs.append(gt_img[0].cpu().permute(1, 2, 0).numpy())
                # VAE reconstruction (same as GT for visualization)
                recon_imgs.append(gt_img[0].cpu().permute(1, 2, 0).numpy())
            else:
                # Beyond GT sequence, use predicted latent decoded with last frame's skip features
                pred_img = vae.decode(pred_latents[t], skip_t)
                if pred_img.min() < 0:
                    pred_img = (pred_img + 1.0) / 2.0
                pred_img = torch.clamp(pred_img, 0, 1)
                # Use predicted as placeholder for GT/VAE rows
                gt_imgs.append(pred_img[0].cpu().permute(1, 2, 0).numpy())
                recon_imgs.append(pred_img[0].cpu().permute(1, 2, 0).numpy())
            
            # Predicted image (use GT skip features for better visualization)
            pred_img = vae.decode(pred_latents[t], skip_t)  # (1, 3, 64, 64)
            if pred_img.min() < 0:
                pred_img = (pred_img + 1.0) / 2.0
            pred_img = torch.clamp(pred_img, 0, 1)
            pred_imgs.append(pred_img[0].cpu().permute(1, 2, 0).numpy())
    
    # Plot
    fig, axes = plt.subplots(3, rollout_steps, figsize=(rollout_steps * 1.5, 9))
    if rollout_steps == 1:
        axes = axes.reshape(3, 1)
    
    fig.suptitle(f"30-step GT vs VAE Recon vs Predictor Rollout (Sample {sample_idx})", fontsize=16)
    
    for t in range(rollout_steps):
        if t < len(gt_imgs):
            axes[0, t].imshow(gt_imgs[t])
            axes[0, t].axis("off")
            if t == 0:
                axes[0, t].set_title("GT", fontsize=8)
            else:
                axes[0, t].set_title(f"{t}", fontsize=6)
        
        if t < len(recon_imgs):
            axes[1, t].imshow(recon_imgs[t])
            axes[1, t].axis("off")
            if t == 0:
                axes[1, t].set_title("VAE", fontsize=8)
            else:
                axes[1, t].set_title(f"{t}", fontsize=6)
        
        if t < len(pred_imgs):
            axes[2, t].imshow(pred_imgs[t])
            axes[2, t].axis("off")
            if t == 0:
                axes[2, t].set_title("Pred", fontsize=8)
            else:
                axes[2, t].set_title(f"{t}", fontsize=6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 30-step rollout visualization to {save_path}")


def generate_prediction_video(model: VAEPredictor, dataset: TrajectoryDataset, device: torch.device,
                              save_path: str, rollout_steps: int = 100, sample_idx: int = 0, fps: int = 10,
                              debug_rollout: bool = False,
                              action_mode: str = "repeat_last",
                              video_layout: str = "pred"):
    """Generate video of long-term prediction rollout"""
    if not HAS_IMAGEIO:
        print("Warning: imageio not available, cannot generate video")
        return
    
    model.eval()
    
    if model.vae_encoder is None:
        print("Warning: VAE encoder not available, skipping video generation")
        return
    
    vae = model.vae_encoder
    vae.eval()
    
    # Get sample from dataset
    sample = dataset[sample_idx]
    input_frames = sample['input_frames'].unsqueeze(0).to(device)  # (1, 15, 3, 64, 64)
    target_frames = sample['target_frames'].unsqueeze(0).to(device)  # (1, 15, 3, 64, 64)
    actions = sample.get('actions', None)
    if actions is not None and getattr(model, "action_dim", 0) > 0:
        actions = actions.unsqueeze(0).to(device)  # (1, 16, action_dim) (dataset returns length=sequence_length)
    else:
        actions = None

    # Optionally fetch a longer continuous action sequence directly from the underlying NPZ (memmap),
    # instead of repeating the last action when t exceeds dataset sequence length.
    actions_long = None
    gt_frames_u8 = None  # optional (rollout_steps, H, W, 3) uint8 for left side of video
    file_idx = None
    local_start = None
    f = None
    if action_mode == "from_npz":
        try:
            file_idx, local_start = dataset._locate(sample_idx)
            f = dataset.files[file_idx]
            s = int(local_start)
            # We want actions for transitions s->s+1, ..., s+rollout_steps-1 -> s+rollout_steps
            raw_actions = f["actions"]  # memmap (T, A)
            avail = int(raw_actions.shape[0] - s)
            take = min(rollout_steps, max(0, avail))
            if take <= 0:
                raise ValueError("No available actions for this sample index.")
            a_np = np.array(raw_actions[s:s+take], dtype=np.float32)  # (take, A)
            # Normalize actions exactly like TrajectoryDataset.__getitem__
            mean = dataset.act_mean.astype(np.float32)
            std = dataset.act_std.astype(np.float32)
            a_np = (a_np - mean) / std
            a_np = np.clip(a_np, -3.0, 3.0) / 3.0
            # Pad to rollout_steps by repeating last action if needed
            if take < rollout_steps:
                pad = np.repeat(a_np[-1:, :], rollout_steps - take, axis=0)
                a_np = np.concatenate([a_np, pad], axis=0)
            actions_long = torch.from_numpy(a_np).unsqueeze(0).to(device)  # (1, rollout_steps, A)
            print(f"[video] action_mode=from_npz: loaded continuous actions from {f.get('path','<npz>')} start={s}, steps={rollout_steps}")

            # Also load GT frames from the same NPZ for side-by-side comparison if requested.
            # pred_latents[t] corresponds to frame (t+1), so we take GT frames from s+1.
            if video_layout == "gt_pred":
                raw_frames = f["frames"]  # memmap (T, 3, H, W), uint8
                Tframes = int(raw_frames.shape[0])
                start_gt = s + 1
                end_gt = start_gt + rollout_steps
                if start_gt >= Tframes:
                    raise ValueError("GT frames not available for this start index.")
                gt_take = max(0, min(rollout_steps, Tframes - start_gt))
                gt = np.array(raw_frames[start_gt:start_gt + gt_take], dtype=np.uint8)  # (gt_take,3,H,W)
                # pad with last GT frame if short
                if gt_take < rollout_steps:
                    pad = np.repeat(gt[-1:, ...], rollout_steps - gt_take, axis=0)
                    gt = np.concatenate([gt, pad], axis=0)
                gt_frames_u8 = gt.transpose(0, 2, 3, 1)  # (rollout_steps,H,W,3)
        except Exception as e:
            print(f"[video] Warning: action_mode=from_npz failed ({e}); falling back to repeat_last.")
            actions_long = None
            action_mode = "repeat_last"
    
    # Combine input and target frames for full sequence
    all_frames = torch.cat([input_frames, target_frames], dim=1)  # (1, 30, 3, 64, 64)
    T = all_frames.shape[1]
    
    print(f"Generating {rollout_steps}-step prediction video...")
    
    with torch.no_grad():
        # Encode first frame
        first_frame = all_frames[:, 0, ...]  # (1, 3, 64, 64)
        mu0, logvar0, _ = vae.encode(first_frame)
        z0 = vae.reparameterize(mu0, logvar0)  # (1, C, 4, 4)
        
        # Build GT latents (only for available frames)
        gt_latents = []
        for t in range(min(rollout_steps, T)):
            frame_t = all_frames[:, t, ...]  # (1, 3, 64, 64)
            mu_t, logvar_t, _ = vae.encode(frame_t)
            z_t = vae.reparameterize(mu_t, logvar_t)
            gt_latents.append(z_t)
        
        # Predictor rollout latents
        pred_latents = []
        z_prev = z0.clone()
        for t in range(rollout_steps):
            # Get action if available (use last available action if t exceeds sequence length)
            a = None
            if actions_long is not None:
                a = actions_long[:, t:t+1, :]  # (1,1,A)
            elif actions is not None and actions.shape[1] > 0:
                action_idx = min(t, actions.shape[1] - 1)
                a = actions[:, action_idx:action_idx+1, :]  # (1, 1, action_dim)
            
            # Predict next latent
            z_prev_expanded = z_prev.unsqueeze(1)  # (1, 1, C, 4, 4)
            z_pred = model.predict(z_prev_expanded, a)  # (1, 1, C, 4, 4)
            z_pred = z_pred.squeeze(1)  # (1, C, 4, 4)
            # clone() avoids any chance of accidental aliasing
            pred_latents.append(z_pred.detach().clone())
            z_prev = z_pred
            
            if (t + 1) % 10 == 0:
                print(f"  Predicted step {t + 1}/{rollout_steps}")
        
        # Decode to images and create video frames
        print("Decoding images...")
        video_frames = []
        debug_stats = []
        prev_lat = None
        prev_img_f = None
        prev_img_u8 = None
        
        for t in range(rollout_steps):
            # Decode predicted latent without GT skip features (to see real prediction degradation)
            # Use None for skip_features to use the decoder's fallback (zero-padding)
            pred_img = vae.decode(pred_latents[t], None)  # (1, 3, 64, 64)
            if pred_img.min() < 0:
                pred_img = (pred_img + 1.0) / 2.0
            pred_img = torch.clamp(pred_img, 0, 1)
            pred_img_f = pred_img[0].detach().cpu()  # (3, H, W) float in [0,1]
            pred_img_np = pred_img_f.permute(1, 2, 0).numpy()  # (H, W, 3)
            pred_img_u8 = (pred_img_np * 255.0).round().astype(np.uint8)
            
            # Video layout
            if video_layout == "gt_pred":
                if gt_frames_u8 is None:
                    # Fallback: use available frames from dataset sample (input+target), padded.
                    # all_frames has shape (1, T_all, 3, H, W). Pred at t corresponds to frame index t+1.
                    frame_idx = min(t + 1, T - 1)
                    gt_img_u8 = (all_frames[0, frame_idx].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                else:
                    gt_img_u8 = gt_frames_u8[t]
                # left=GT, right=Pred
                frame_u8 = np.concatenate([gt_img_u8, pred_img_u8], axis=1)
                frame_u8 = overlay_sbs_labels(frame_u8, left_label="GT", right_label="PR")
                video_frames.append(frame_u8)
            else:
                # pred-only (default)
                video_frames.append(pred_img_u8)

            if debug_rollout:
                lat_max = None
                imgf_max = None
                imgu8_max = None
                if prev_lat is not None:
                    lat_max = float((pred_latents[t] - prev_lat).abs().max().item())
                if prev_img_f is not None:
                    imgf_max = float((pred_img_f - prev_img_f).abs().max().item())
                if prev_img_u8 is not None:
                    # uint8 difference after quantization (what you see in the mp4)
                    imgu8_max = int(np.abs(pred_img_u8.astype(np.int16) - prev_img_u8.astype(np.int16)).max())
                debug_stats.append({
                    "t": t,
                    "latent_max_abs_delta": lat_max,
                    "img_float_max_abs_delta": imgf_max,
                    "img_uint8_max_abs_delta": imgu8_max,
                })
                prev_lat = pred_latents[t]
                prev_img_f = pred_img_f
                prev_img_u8 = pred_img_u8
            
            if (t + 1) % 10 == 0:
                print(f"  Decoded step {t + 1}/{rollout_steps}")

        if debug_rollout:
            # Save debug deltas alongside the video
            try:
                import json
                debug_path = os.path.splitext(save_path)[0] + "_debug.json"
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(debug_stats, f, indent=2)
                print(f"Saved rollout debug stats to {debug_path}")
                # Quick hint if it 'freezes' due to uint8 quantization
                tail = [d for d in debug_stats[-10:] if d.get("img_uint8_max_abs_delta") is not None]
                if tail:
                    max_u8 = max(d["img_uint8_max_abs_delta"] for d in tail)
                    if max_u8 == 0:
                        print("Note: last frames are identical AFTER uint8 quantization (mp4). The float prediction may still change slightly.")
            except Exception as e:
                print(f"Warning: failed to write debug stats: {e}")
        
        # Save video
        print(f"Saving video to {save_path}...")
        imageio.mimsave(save_path, video_frames, fps=fps, codec='libx264', quality=8)
        print(f"Video saved to {save_path} ({len(video_frames)} frames, {fps} fps)")
