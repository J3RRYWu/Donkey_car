#!/usr/bin/env python3
"""
Utility functions for evaluation: image metrics (PSNR, SSIM), image processing
"""

import torch
import torch.nn.functional as F
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def to_01(x: torch.Tensor) -> torch.Tensor:
    """Convert model outputs to [0,1] range if they look like tanh outputs."""
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


def psnr_from_mse(mse: float, data_range: float = 1.0, eps: float = 1e-12) -> float:
    """Calculate PSNR from MSE."""
    mse = float(max(mse, eps))
    return float(10.0 * np.log10((data_range ** 2) / mse))


def gaussian_window(window_size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    """Create a Gaussian window for SSIM calculation."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)
    return w


def ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0, 
         window_size: int = 11, sigma: float = 1.5) -> float:
    """SSIM over batch, computed on grayscale for speed. Expects img in [0,1], shape (B,3,H,W)."""
    img1 = to_01(img1)
    img2 = to_01(img2)
    # grayscale
    if img1.size(1) == 3:
        w_rgb = img1.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        img1g = (img1 * w_rgb).sum(dim=1, keepdim=True)
        img2g = (img2 * w_rgb).sum(dim=1, keepdim=True)
    else:
        img1g = img1
        img2g = img2

    window = gaussian_window(window_size, sigma, device=img1.device, dtype=img1.dtype)
    pad = window_size // 2
    mu1 = F.conv2d(img1g, window, padding=pad)
    mu2 = F.conv2d(img2g, window, padding=pad)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1g * img1g, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2g * img2g, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(img1g * img2g, window, padding=pad) - mu12

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12)
    return float(ssim_map.mean().item())


def overlay_sbs_labels(frame_u8: np.ndarray, left_label: str = "GT", right_label: str = "PR") -> np.ndarray:
    """Overlay 'GT'/'PR' labels on a side-by-side uint8 frame (H, W, 3).
    If Pillow is unavailable, returns the frame unchanged.
    """
    if not HAS_PIL:
        return frame_u8
    try:
        h, w, _ = frame_u8.shape
        img = Image.fromarray(frame_u8, mode="RGB")
        draw = ImageDraw.Draw(img)
        # Try a default font; fall back to PIL default
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Put labels near top-left of each half, with a dark background box.
        half = w // 2
        pad = 6
        y = 6

        def draw_label(x: int, text: str):
            bbox = draw.textbbox((x + pad, y), text, font=font)
            # Background rectangle slightly padded
            bx0, by0, bx1, by1 = bbox
            bg = (0, 0, 0)
            fg = (255, 255, 255)
            draw.rectangle((bx0 - 4, by0 - 2, bx1 + 4, by1 + 2), fill=bg)
            draw.text((x + pad, y), text, fill=fg, font=font)

        draw_label(0, left_label)
        draw_label(half, right_label)

        return np.array(img)
    except Exception:
        return frame_u8


def effective_horizon_from_curve(curve: dict, predicate) -> int:
    """Largest horizon h such that predicate(curve[h]) is True. curve values may be None."""
    best = 0
    for h in sorted(curve.keys()):
        v = curve[h]
        if v is None:
            continue
        try:
            if predicate(float(v)):
                best = int(h)
        except Exception:
            continue
    return best
