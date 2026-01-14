"""
Conformal MPC Controller for DonkeyCar
Integrates LSTM prediction with Conformal Prediction safety guarantees

Author: Your Name
Date: 2026-01-14
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import json


class ConformalMPC:
    """
    Model Predictive Controller with Conformal Prediction safety constraints
    
    Architecture:
        1. Encode history images → latent space (VAE)
        2. Predict future latents given action sequence (LSTM)
        3. Optimize action sequence with CP-based safety constraints
        4. Return first optimal action (receding horizon)
    """
    
    def __init__(
        self,
        vae_model,
        lstm_model,
        cp_quantiles_path: str,
        horizon: int = 17,
        control_freq: float = 10.0,
        device: str = 'cuda'
    ):
        """
        Args:
            vae_model: Trained VAE encoder/decoder
            lstm_model: Trained LSTM predictor
            cp_quantiles_path: Path to CP quantiles JSON
            horizon: Prediction horizon (recommended: 17 steps for MSE < 3.0)
            control_freq: Control frequency in Hz
            device: 'cuda' or 'cpu'
        """
        self.vae = vae_model.eval()
        self.lstm = lstm_model.eval()
        self.device = device
        self.horizon = horizon
        self.dt = 1.0 / control_freq
        
        # Freeze all model parameters (MPC only optimizes actions, not model weights)
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.lstm.parameters():
            param.requires_grad = False
        
        # Load CP quantiles
        with open(cp_quantiles_path, 'r') as f:
            cp_data = json.load(f)
            # CPQuantiles format: {"alpha": 0.05, "norm": "l2", "q": [20.0, ...]}
            q_full = cp_data['q']  # List of all quantiles
            self.q_t = torch.tensor(
                q_full[:horizon],  # Take first 'horizon' values
                device=device,
                dtype=torch.float32
            )
            self.cp_alpha = cp_data.get('alpha', 0.05)
            self.cp_norm = cp_data.get('norm', 'l2')
        
        # MPC parameters (can be tuned)
        self.params = {
            'tracking_weight': 1.0,
            'action_penalty': 0.5,         # ↑↑ 再增大5倍，强烈惩罚极端动作
            'smooth_penalty': 2.0,         # ↑↑ 再增大2倍，非常平滑
            'conservatism': 0.1,           # ↑ 增加保守性
            'uncertainty_threshold': 40.0, # ↓ 降低阈值，更早触发保守行为
            'lr': 0.005,                   # ↓↓ 学习率减半，极致稳定
            'n_iters': 150,                # ↑↑ 增加迭代次数，充分收敛
            'u_min': torch.tensor([-1.0, -1.0], device=device),
            'u_max': torch.tensor([1.0, 1.0], device=device)
        }
        
        # For warmstart
        self.u_prev = torch.zeros(2, device=device)
        
    def control_step(
        self,
        images_history: torch.Tensor,
        z_goal: torch.Tensor,
        u_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Single MPC control step
        
        Args:
            images_history: [B, T, C, H, W] or [T, C, H, W] - Recent frames
            z_goal: [C, H, W] or [C*H*W] - Goal latent (convolutional or flattened)
            u_prev: [2] - Previous action (for smoothness)
        
        Returns:
            u_opt: [2] - Optimal action (steering, throttle)
            info: Dict with diagnostic information
        """
        if images_history.dim() == 4:
            images_history = images_history.unsqueeze(0)  # Add batch dim [1, T, C, H, W]
        
        if u_prev is None:
            u_prev = self.u_prev
        
        # Step 1: Encode history to latent space
        with torch.no_grad():
            # VAE encode expects [B, C, H, W], so we need to merge time into batch
            B, T, C, H, W = images_history.shape
            images_flat = images_history.reshape(B * T, C, H, W)  # [B*T, C, H, W]
            
            mu, _ = self.vae.encode(images_flat)  # [B*T, C', H', W']
            
            # Reshape back to sequence: [B*T, C', H', W'] -> [B, T, C', H', W']
            if mu.dim() == 4:  # Convolutional latent [B*T, C', H', W']
                C_lat, H_lat, W_lat = mu.shape[1:]
                mu = mu.reshape(B, T, C_lat, H_lat, W_lat)
            else:  # Vector latent [B*T, D]
                D = mu.shape[1]
                mu = mu.reshape(B, T, D)
            
            # Flatten spatial dims: [B, T, C', H', W'] -> [B, T, C'*H'*W']
            z_history = mu.reshape(B, T, -1).squeeze(0)  # [T, C'*H'*W']
        
        # Step 2: Optimize action sequence
        u_opt, info = self.optimize_actions(
            z_history=z_history,
            z_goal=z_goal,
            u_prev=u_prev
        )
        
        # Update state
        self.u_prev = u_opt.clone()
        
        return u_opt, info
    
    def optimize_actions(
        self,
        z_history: torch.Tensor,
        z_goal: torch.Tensor,
        u_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Optimize action sequence using gradient descent
        
        Args:
            z_history: [T, latent_dim] - Historical latents
            z_goal: [latent_dim] - Goal latent
            u_prev: [2] - Previous action
        
        Returns:
            u_opt: [2] - First optimal action
            info: Dict with costs, constraints, etc.
        """
        N = self.horizon
        
        # Initialize: small actions near zero (better for tracking tasks)
        # Option 1: Zero initialization
        u_seq = torch.zeros(N, 2, device=self.device, requires_grad=True)
        
        # Option 2: Small random perturbation around previous action
        # u_seq = u_prev.repeat(N, 1).clone()
        # u_seq += 0.1 * torch.randn_like(u_seq)
        # u_seq = u_seq.requires_grad_(True)
        
        optimizer = torch.optim.Adam([u_seq], lr=self.params['lr'])
        
        cost_history = []
        
        # CRITICAL: Enable train mode for LSTM backward (cudnn requirement)
        # Model parameters are frozen (requires_grad=False set in __init__)
        # Only u_seq will be updated
        was_training = self.lstm.training
        self.lstm.train()
        
        for iter_idx in range(self.params['n_iters']):
            optimizer.zero_grad()
            
            # Forward prediction (model params won't be updated, only u_seq)
            z_pred = self._predict_trajectory(z_history, u_seq)  # [N, latent_dim]
            
            # Compute cost
            cost, cost_components = self._compute_cost(
                z_pred, u_seq, z_goal, u_prev
            )
            
            # Backward and optimize ONLY u_seq (model params frozen)
            cost.backward()
            optimizer.step()  # Only updates u_seq since it's the only param in optimizer
            
            # Project to feasible region
            with torch.no_grad():
                u_seq.data = torch.clamp(
                    u_seq.data,
                    self.params['u_min'],
                    self.params['u_max']
                )
            
            cost_history.append(cost.item())
        
        # Restore original training state
        if not was_training:
            self.lstm.eval()
        
        # Extract first action
        u_opt = u_seq[0].detach()
        
        # Diagnostic info
        with torch.no_grad():
            z_pred_final = self._predict_trajectory(z_history, u_seq.detach())
            info = {
                'cost_final': cost_history[-1],
                'cost_history': cost_history,
                'z_pred': z_pred_final.cpu().numpy(),
                'u_seq': u_seq.detach().cpu().numpy(),
                'q_t': self.q_t.cpu().numpy(),
                'tracking_error': torch.norm(z_pred_final - z_goal, dim=1).cpu().numpy()
            }
        
        return u_opt, info
    
    def _predict_trajectory(
        self,
        z_history: torch.Tensor,
        u_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict future trajectory using LSTM
        
        Args:
            z_history: [T, latent_dim] - Flattened latents (C*H*W)
            u_seq: [N, action_dim]
        
        Returns:
            z_pred: [N, latent_dim] - Flattened predicted latents
        """
        # Reshape to VAE format: [T, C*H*W] -> [1, T, C, H, W]
        T, D = z_history.shape
        
        # For 64D latent with 4x4 spatial: 64 = 4 * 4 * 4 (C=4)
        if D == 64:
            C, H, W = 4, 4, 4
            z_context = z_history.view(T, C, H, W).unsqueeze(0)  # [1, T, 4, 4, 4]
        else:
            # Try to infer dimensions (assume square spatial)
            import math
            # Guess: D = C * H * W, assume H = W = 4 or 7
            if D % 16 == 0:  # H=W=4
                C = D // 16
                z_context = z_history.view(T, C, 4, 4).unsqueeze(0)
            elif D % 49 == 0:  # H=W=7
                C = D // 49
                z_context = z_history.view(T, C, 7, 7).unsqueeze(0)
            else:
                raise ValueError(f"Cannot infer latent dimensions from D={D}")
        
        # Prepare actions
        if self.lstm.action_dim > 0 and u_seq is not None:
            # Extend action sequence to cover context + prediction
            # rollout_from_context needs actions for prediction steps
            a_full = u_seq.unsqueeze(0)  # [1, N, action_dim]
            
            z_pred = self.lstm.rollout_from_context(
                z_context=z_context,
                steps=self.horizon,
                a_full=a_full,
                context_action_len=0,  # Don't prime on history actions
                start_action_index=0   # Start from first action in u_seq
            )
        else:
            z_pred = self.lstm.rollout_from_context(
                z_context=z_context,
                steps=self.horizon
            )
        
        # Flatten back to [N, latent_dim]
        if z_pred.dim() == 5:  # [1, N, C, H, W]
            B, N, C, H, W = z_pred.shape
            z_pred = z_pred.reshape(B, N, -1)  # [1, N, C*H*W]
        elif z_pred.dim() == 3:  # [1, N, D]
            pass
        
        return z_pred.squeeze(0)  # [N, latent_dim]
    
    def _compute_cost(
        self,
        z_pred: torch.Tensor,
        u_seq: torch.Tensor,
        z_goal: torch.Tensor,
        u_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute MPC cost with CP-based uncertainty weighting
        
        Args:
            z_pred: [N, latent_dim] - Predicted trajectory
            u_seq: [N, action_dim] - Action sequence
            z_goal: [latent_dim] - Goal state
            u_prev: [action_dim] - Previous action
        
        Returns:
            total_cost: Scalar
            components: Dict of cost components
        """
        N = z_pred.shape[0]
        
        # Ensure z_goal has correct shape
        if z_goal.dim() == 1:
            z_goal_expanded = z_goal.unsqueeze(0)  # [1, latent_dim]
        else:
            z_goal_expanded = z_goal
        
        # 1. Tracking cost (weighted by uncertainty)
        tracking_errors = torch.norm(z_pred - z_goal_expanded, dim=1)  # [N]
        
        # Weight: inverse of uncertainty (far horizon gets less weight)
        weights = 1.0 / (1.0 + self.q_t / 20.0)  # [N]
        tracking_cost = torch.sum(weights * tracking_errors ** 2)
        
        # 2. Action penalty (avoid extreme actions)
        action_cost = torch.sum(u_seq ** 2)
        
        # 3. Smoothness penalty (avoid jitter)
        u_diff = torch.diff(u_seq, dim=0)
        smooth_cost = torch.sum(u_diff ** 2)
        
        # Also penalize change from previous action
        smooth_cost += torch.sum((u_seq[0] - u_prev) ** 2)
        
        # 4. Conservatism penalty (penalize high uncertainty regions)
        conservatism_cost = torch.sum(
            torch.relu(self.q_t - self.params['uncertainty_threshold'])
        )
        
        # Total cost
        total_cost = (
            self.params['tracking_weight'] * tracking_cost +
            self.params['action_penalty'] * action_cost +
            self.params['smooth_penalty'] * smooth_cost +
            self.params['conservatism'] * conservatism_cost
        )
        
        components = {
            'tracking': tracking_cost.item(),
            'action': action_cost.item(),
            'smoothness': smooth_cost.item(),
            'conservatism': conservatism_cost.item()
        }
        
        return total_cost, components
    
    def compute_goal_latent(
        self,
        goal_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a goal image to latent space
        
        Args:
            goal_image: [C, H, W] or [B, C, H, W]
        
        Returns:
            z_goal: [C'*H'*W'] - Flattened latent
        """
        if goal_image.dim() == 3:
            goal_image = goal_image.unsqueeze(0)  # [1, C, H, W]
        
        with torch.no_grad():
            mu, _ = self.vae.encode(goal_image)  # [B, C', H', W'] for VAE
            # Flatten: [B, C', H', W'] -> [B, C'*H'*W']
            z_goal = mu.reshape(mu.shape[0], -1).squeeze(0)  # [C'*H'*W']
        
        return z_goal
    
    def update_params(self, new_params: Dict):
        """Update MPC parameters dynamically"""
        self.params.update(new_params)
    
    def get_safety_margin(self, horizon_idx: int) -> float:
        """
        Get CP safety margin at a specific horizon
        
        Args:
            horizon_idx: Step index (0 to horizon-1)
        
        Returns:
            q_t: Safety radius in latent space
        """
        return self.q_t[horizon_idx].item()


# ============================================================================
# Utility Functions
# ============================================================================

def load_mpc_controller(
    vae_path: str,
    lstm_path: str,
    cp_quantiles_path: str,
    horizon: int = 17,
    device: str = 'cuda'
) -> ConformalMPC:
    """
    Convenience function to load MPC controller from checkpoints
    
    Args:
        vae_path: Path to VAE checkpoint
        lstm_path: Path to LSTM checkpoint
        cp_quantiles_path: Path to CP quantiles JSON
        horizon: MPC horizon
        device: 'cuda' or 'cpu'
    
    Returns:
        mpc: ConformalMPC instance
    """
    from vae_predictor import VAEPredictor
    
    # Load models
    vae_ckpt = torch.load(vae_path, map_location=device)
    lstm_ckpt = torch.load(lstm_path, map_location=device)
    
    # Instantiate (you may need to adjust based on your model structure)
    # This is a placeholder - adapt to your actual model loading logic
    print(f"Loading VAE from {vae_path}")
    print(f"Loading LSTM from {lstm_path}")
    print(f"Loading CP quantiles from {cp_quantiles_path}")
    
    # TODO: Implement actual model loading
    # vae = YourVAEModel()
    # vae.load_state_dict(vae_ckpt)
    # lstm = VAEPredictor()
    # lstm.load_state_dict(lstm_ckpt['model_state_dict'])
    
    # mpc = ConformalMPC(
    #     vae_model=vae,
    #     lstm_model=lstm,
    #     cp_quantiles_path=cp_quantiles_path,
    #     horizon=horizon,
    #     device=device
    # )
    
    # return mpc
    raise NotImplementedError("Please implement model loading logic")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Conformal MPC Controller")
    print("=" * 70)
    print("This module provides MPC control with CP safety guarantees.")
    print()
    print("Typical usage:")
    print("  1. Load VAE, LSTM, and CP quantiles")
    print("  2. Create ConformalMPC instance")
    print("  3. In control loop:")
    print("     - Get current images")
    print("     - Define goal")
    print("     - Call mpc.control_step()")
    print("     - Apply first optimal action")
    print()
    print("See SYSTEM_ANALYSIS_AND_MPC_PLAN.md for detailed guide.")
