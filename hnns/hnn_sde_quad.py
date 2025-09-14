# pip install torchsde (Python >=3.8 and PyTorch >=1.6.0)
# https://github.com/google-research/torchsde
# https://github.com/google-research/torchsde/blob/master/DOCUMENTATION.md
from torchsde import sdeint, SDEStratonovich

# pip install torchdiffeq
# from torchdiffeq import odeint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import os
# Add the parent folder of `mbrl_dynamics_net` to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import Dict, List, Union, Tuple, Optional, Callable
from mbrl_dynamics_net.utils.buffer import OfflineDatasetLoader
from mbrl_dynamics_net.utils.logger import make_log_dirs
from mbrl_dynamics_net.utils.scaler import StandardScaler
from mbrl_dynamics_net.utils.state_features import StateFeatures
from torch.utils.tensorboard import SummaryWriter

from mbrl_dynamics_net.utils.analyze_features import get_dataset_stats
from mbrl_dynamics_net.utils import logger 
# Log directory path
logger.set_root(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "log"))

from datetime import datetime
import wandb
import joblib

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim) -> None:  
        super(AutoEncoder, self).__init__()
        assert latent_dim % 2 == 0, "latent_dim must be even for q, p split"

        self.latent_dim = latent_dim

        # Encoder: maps physics state -> latent [q, p]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, latent_dim)
        )

        # Decoder: latent [q, p] -> reconstructed physics state
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, input_dim)
        )

    def forward(self, s):
        """
        Full autoencoder pass.
        Args:
            s: [batch, input_dim] (dof_pos + dof_vel) - physics_state
        Returns:
            recon: reconstructed physics state
            (q, p): canonical split of latent
            u: full latent vector [q, p]
        """
        u = self.encoder(s)               

        # Decoding: Canonical states back to the state
        s_reconstructed = self.decoder(u)

        q, p = torch.chunk(u, 2, dim=-1)  # Split canonical variables (enforce q/p split)
        return s_reconstructed, (q, p), u

    def encode(self, s):
        """Encode physics state into latent [q, p]."""
        u = self.encoder(s)
        q, p = torch.chunk(u, 2, dim=-1)
        return q, p, u

    def decode(self, u):
        """Decode latent [q, p] back to physics state."""
        return self.decoder(u)

class HamiltonianSDE(SDEStratonovich):
    def __init__(self, latent_dim=24, action_dim=12, context_dim=34):
        super().__init__(noise_type="diagonal")
        self.sde_type = "stratonovich"
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        assert latent_dim % 2 == 0, "latent_dim must be even"
        self.K = latent_dim // 2

        # Neural network representing the Hamiltonian H(q, p)
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)   # scalar Hamiltonian
        )

        # Neural network for diffusion term g(u, a)
        # Modeled as a diagonal matrix (noise applied independently per dimension).
        # self.diffusion_net = nn.Sequential(
        #     nn.Linear(latent_dim + action_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, latent_dim)  # diagonal noise
        # )

        # Diffusion net can depend on (u, a, c)
        self.diffusion_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim + context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)  # diagonal diffusion
        )

        # Neural network for external forces Q(a)
        # Assumes state-independent generalized forces
        # Forces depend only on control torques
        # self.force_net = nn.Sequential(
        #     nn.Linear(action_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, self.K)  # K = DoF
        # )

        # To model contacts, damping, or state-dependent actuation
        # self.force_net = nn.Sequential(
        #     nn.Linear(context_dim + action_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, self.K) # Output generalized forces
        # )

        # context conditions forces/noise
        self.force_net = nn.Sequential(
            nn.Linear(action_dim + context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.K)  # generalized forces
        )

    def hamiltonian_drift(self, t, u, a=None):
        if a is None:
            a = torch.zeros(u.shape[0], self.action_dim, device=u.device)
        
        q, p = torch.chunk(u, 2, dim=-1)

        # Ensure q and p require gradients
        q.requires_grad_(True)
        p.requires_grad_(True)

        # Compute Hamiltonian input
        H_in = torch.cat([q, p], dim=-1)      # H_in: (64, 24)

        H_scalar = self.hamiltonian_net(H_in)
        grads = torch.autograd.grad(
            outputs=H_scalar,
            inputs=(q, p),
            grad_outputs=torch.ones_like(H_scalar),
            retain_graph=True,
            create_graph=True
        )

        # Extract gradients for each sample in the batch
        dq_dt = grads[1]

        # Generalized forces depend only on actions Q(a_t)
        dp_dt = -grads[0] + self.force_net(a) # Fine if actions = direct torques
        # State-dependent forces case Q(u_t,a_t)
        # dp_dt = -grads[0] + self.force_net(torch.cat([u, a], dim=-1))

        du_dt = torch.cat([dq_dt, dp_dt], dim=-1)

        return du_dt

    def stochastic_diffusion(self, t, u, a=None, c=None):
        if a is None:
            a = torch.zeros(u.shape[0], self.action_dim, device=u.device)
        return self.diffusion_net(torch.cat([u, a], dim=-1))

class ActionSDE(SDEStratonovich):
    """Wrapper for an SDE model that injects a fixed action vector into both the drift (f) 
    and diffusion (g) functions during integration.

    This is useful when integrating stochastic dynamics models (in our case, HamiltonianSDE) 
    where the action is constant over the integration window (matching dt), such as in 
    short-horizon model prediction or single-step rollout.

    Attributes:
        sde_type (str): Inherited from base_sde; specifies the SDE interpretation ('ito' or 'stratonovich').
                        Required by torchsde solvers to choose the correct numerical integration scheme.
        base_sde (SDEStratonovich): The underlying SDE model defining f(t, u, a) and g(t, u, a).
        a_t (Tensor): The fixed action to be passed into the base SDE's drift and diffusion during integration.
    """
    def __init__(self, base_sde, a_t, c_t):
        super().__init__(noise_type=base_sde.noise_type)
        self.sde_type = base_sde.sde_type
        self.base_sde = base_sde
        self.a_t = a_t
        self.c_t = c_t    

    def f(self, t, u):
        return self.base_sde.hamiltonian_drift(t, u, self.a_t)
    
    def g(self, t, u):
        return self.base_sde.stochastic_diffusion(t, u, self.a_t, self.c_t)

class RewardDecoder(nn.Module):
    def __init__(self, latent_dim, action_dim, context_dim=0) -> None:
        super().__init__()

        self.reward_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, u, a, c=None):
        """
        Args:
            u    : the canonical state vector in latent space - [q, p]
            a    : actions
            c    : context (optional tensor, e.g. gait, commands)
        """
        # u = torch.cat([q, p], dim=-1)

        if c is not None:
            x = torch.cat([u, a, c], dim=-1)
        else:
            x = torch.cat([u, a], dim=-1)

        return self.reward_net(x)   # shape: [B, 1]

class HNNSDE(nn.Module):
    def __init__(self, input_dim, latent_dim, action_dim, context_dim, device='cpu') -> None:
        super().__init__()
        self.device = device
        self.autoencoder = AutoEncoder(input_dim, latent_dim).to(device)
        self.ode_func = HamiltonianSDE(latent_dim, action_dim).to(device)
        self.reward_decoder = RewardDecoder(latent_dim, action_dim, context_dim).to(device)
        self.features = StateFeatures()
        # self.latent_dim = latent_dim
        # self.action_dim = action_dim

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"].reshape(-1, 1)
        return obss, actions, next_obss, rewards

    def predict_state_reward(self, s_t, a_t, dt):
        """Predict next state and reward stochastically.
        a_t stays constant during the short integration window (matching dt).

        Args:
            s_t          : Current full state [batch, state_dim=58]
            a_t          : Current action [batch, action_dim]
            dt           : Integration step size
        """

        # Split state into physics vs. context 
        dof_pos_idx = self.features["dof_pos"]   # slice(18, 30)
        dof_vel_idx = self.features["dof_vel"]   # slice(30, 42)

        # Direct slicing (no autoencoder)
        # q = s_t[:, dof_pos_idx]
        # p = s_t[:, dof_vel_idx]
        # u = torch.cat([q, p], dim=-1)  # [batch, 24]

        # Physics-only autoencoder
        physics_s_t = torch.cat([s_t[:, dof_pos_idx], s_t[:, dof_vel_idx]], dim=-1)
        q, p, u = self.autoencoder.encode(physics_s_t)

        # Context = all features except physics
        context_mask = torch.ones(s_t.shape[1], dtype=torch.bool, device=s_t.device)
        context_mask[dof_pos_idx] = False
        context_mask[dof_vel_idx] = False
        c_t = s_t[:, context_mask]  # [batch, context_dim]

        # Predict reward (conditioned on physics, action, context) 
        r_pred = self.reward_decoder(u, a_t, c_t)

        # Integrate Hamiltonian SDE forward
        t_span = torch.tensor([0, dt], dtype=torch.float32, device=self.device)

        # Pass both action and context to SDE class
        sde_with_action = ActionSDE(self.ode_func, a_t, c_t)

        u_traj = sdeint(
            sde_with_action,   # HamiltonianSDE subclass
            u,                 # initial latent [q,p]
            t_span,            # integration interval
            method='heun',     # Stratonovich-compatible integrator
            dt=dt
        )

        u_next = u_traj[-1]  # [batch, 24]
        q_next, p_next = torch.chunk(u_next, 2, dim=-1)

        # Reconstruct full next state (58D)
        s_tp1_pred = s_t.clone()

        # Update physics
        s_tp1_pred[:, dof_pos_idx] = q_next
        s_tp1_pred[:, dof_vel_idx] = p_next

        # Update "prev_action" slot in context with current
        prev_action_idx = self.features["actions"]
        s_tp1_pred[:, prev_action_idx] = a_t

        # Other context (commands, stance width, etc.) carried forward unchanged

        return s_tp1_pred, r_pred


    # def predict_state_reward(self, s_t, a_t, dt):
    #     """Predict next state and reward stochastically.
    #     a_t stays constant during the short integration window (matching dt).

    #     Args:
    #         s_t          : Current state [batch, state_dim]
    #         a_t          : Current action [batch, action_dim]
    #         dt           : Integration step size
    #     """

    #     # Encode observation to canonical (q, p) and full latent u
    #     q, p, u = self.autoencoder.encode(s_t)

    #     # Predict reward from q, p, and a_t
    #     r_pred = self.reward_decoder(q, p, a_t)

    #     # Integrate canonical state forward one time step
    #     t_span = torch.tensor([0, dt], dtype=torch.float32, device=self.device)
    #     # u_traj = odeint(lambda t, u_: self.ode_func(t, u_, a_t),
    #     #                 u,
    #     #                 t_span,
    #     #                 method='rk4',
    #     #                 options={'step_size': dt})

    #     sde_with_action = ActionSDE(self.ode_func, a_t)

    #     u_traj = sdeint(sde_with_action,              # HamiltonianSDE subclass
    #                     u,                          # initial latent [q,p]
    #                     t_span,                     # tensor([0., dt])
    #                     method='heun',              # Milstein/Heun for Stratonovich
    #                     dt=dt,                      # integration step
    #                     # names={'drift': 'f', 
    #                     #        'diffusion': 'g'},   # match torchsde API
    #                     # args=(a_t,),                # pass action to drift & diffusion
    #                 )

    #     # next canonical state
    #     u_next = u_traj[-1]                         # Shape: [batch_size, latent_dim]

    #     # Decode back to predicted next observation
    #     s_t_plus1_pred = self.autoencoder.decode(u_next)
    #     return s_t_plus1_pred, r_pred

    def compute_loss(self, s_t, a_t, s_tp1_true, r_true, dt, alpha, num_rollouts=20):
        rollout_preds, reward_preds = [], []

        for _ in range(num_rollouts):
            s_pred, r_pred = self.predict_state_reward(s_t, a_t, dt)
            rollout_preds.append(s_pred)
            reward_preds.append(r_pred)

        # Average over Brownian rollouts
        s_pred_mean = torch.stack(rollout_preds, dim=0).mean(dim=0)
        r_pred_mean = torch.stack(reward_preds, dim=0).mean(dim=0)

        # Physics indices (pos + vel only) 
        dof_pos_idx = self.features["dof_pos"]               # slice(18, 30)
        dof_vel_idx = self.features["dof_vel"]               # slice(30, 42)

        physics_t = torch.cat([s_t[:, dof_pos_idx], s_t[:, dof_vel_idx]], dim=-1)
        physics_tp1_true = torch.cat([s_tp1_true[:, dof_pos_idx], s_tp1_true[:, dof_vel_idx]], dim=-1)

        # Encode/decode physics state for reconstruction
        _, _, u = self.autoencoder.encode(physics_t)         # input only pos+vel
        physics_recon = self.autoencoder.decode(u)           # reconstruct pos+vel
        loss_recon = F.mse_loss(physics_recon, physics_t)

        # State prediction loss (only on physics part)
        u_pred = torch.cat([s_pred_mean[:, dof_pos_idx], s_pred_mean[:, dof_vel_idx]], dim=-1)
        loss_state = F.mse_loss(u_pred, physics_tp1_true)

        # Reward loss
        reward_loss_fn = nn.SmoothL1Loss()
        loss_reward = reward_loss_fn(r_pred_mean, r_true)

        # Composite loss
        total_loss = alpha * (loss_recon + loss_state) + (1 - alpha) * loss_reward
        return total_loss, loss_recon, loss_state, loss_reward


class HNNSDETrainer:
    def __init__(self, model, data, batch_size, lr, dt, alpha, log_dirs, holdout_ratio=0.15, retrain=False, device='cpu'):
        self.model = model.to(device)
        self.data = data
        self.device = device
        self.batch_size = batch_size
        self.dt = dt
        self.alpha = alpha
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.log_dirs = log_dirs
        
        # Prepare data for batching (convert numpy arrays to torch tensors)
        obss, actions, next_obss, rewards = self.model.format_samples_for_training(data)
        
        # self.dataset =  TensorDataset(torch.tensor(obss, dtype=torch.float32),
        #                                 torch.tensor(actions, dtype=torch.float32),
        #                                 torch.tensor(next_obss, dtype=torch.float32),
        #                                 torch.tensor(rewards, dtype=torch.float32))

        # Indices of constant features to drop 
        # [gait (8-10), durations (11), body_roll (14), stance_width (15), stance_length (16), aux_reward (17)]
        # constant_idx = [8, 9, 10, 11, 14, 15, 16, 17]
        # # Remove constant features from observations
        # obss = np.delete(obss, constant_idx, axis=1)
        # next_obss = np.delete(next_obss, constant_idx, axis=1)

        data_size = obss.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size

        # Initialize scalers 
        # # StandardScaler for normalizing inputs
        self.obs_scaler = StandardScaler(name="obs") 
        self.act_scaler = StandardScaler(name="act") 
        self.rew_scaler = StandardScaler(name="rew") 

        if retrain == True:
            # train_dataset, holdout_dataset = random_split(self.dataset, [train_size, holdout_size])
            # Seed for reproducibility
            # g = torch.Generator().manual_seed(42)
            # train_dataset, holdout_dataset = random_split(self.dataset, [train_size, holdout_size], generator=g)

            indices = np.arange(data_size)
            np.random.shuffle(indices)
            train_idx, holdout_idx = indices[:train_size], indices[train_size:]

            # Save indices for reproducibility
            np.save(os.path.join(log_dirs, "train_idx.npy"), train_idx)
            np.save(os.path.join(log_dirs, "holdout_idx.npy"), holdout_idx)

            # # Fit scalers on train split only
            self.obs_scaler.fit(obss[train_idx]) 
            self.act_scaler.fit(actions[train_idx]) 
            self.rew_scaler.fit(rewards[train_idx])

            # print(self.obs_scaler.mu, self.obs_scaler.std)
            # print(self.act_scaler.mu, self.act_scaler.std)
            # print(self.rew_scaler.mu, self.rew_scaler.std)

            # Transform all data with fitted scalers
            # Both train + holdout 
            obss = self.obs_scaler.transform(obss) 
            actions = self.act_scaler.transform(actions) 
            next_obss = self.obs_scaler.transform(next_obss) # same obs scaler 
            rewards = self.rew_scaler.transform(rewards)

        else:
            train_idx, holdout_idx = self.load(log_dirs)

            # print(self.obs_scaler.mu, self.obs_scaler.std)
            # print(self.act_scaler.mu, self.act_scaler.std)
            # print(self.rew_scaler.mu, self.rew_scaler.std)

            obss = self.obs_scaler.transform(obss) 
            actions = self.act_scaler.transform(actions) 
            next_obss = self.obs_scaler.transform(next_obss) # same obs scaler 
            rewards = self.rew_scaler.transform(rewards)

        self.train_dataset = TensorDataset(
            torch.tensor(obss[train_idx], dtype=torch.float32),
            torch.tensor(actions[train_idx], dtype=torch.float32),
            torch.tensor(next_obss[train_idx], dtype=torch.float32),
            torch.tensor(rewards[train_idx], dtype=torch.float32),
        )
            
        self.holdout_dataset = TensorDataset(
            torch.tensor(obss[holdout_idx], dtype=torch.float32),
            torch.tensor(actions[holdout_idx], dtype=torch.float32),
            torch.tensor(next_obss[holdout_idx], dtype=torch.float32),
            torch.tensor(rewards[holdout_idx], dtype=torch.float32),
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.holdout_loader = DataLoader(self.holdout_dataset, batch_size=64, shuffle=False)

    def train_one_epoch(self, weights):
        self.model.train()
        total_loss, total_recon_loss, total_state_loss, total_reward_loss = 0, 0, 0, 0

        # for name, param in self.model.named_parameters():
        #     print(name, param.data.mean().item(), param.grad is not None)

        for batch in self.train_loader:
            obss, actions, next_obss, rewards = [x.to(self.device) for x in batch]

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass: compute the total loss (state + reward prediction loss)
            loss, loss_recon, loss_state, loss_reward = self.model.compute_loss(
                obss, actions, next_obss, rewards, self.dt, self.alpha, weights=weights
            )
            
            # Backpropagation and optimization
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_state_loss += loss_state.item()
            total_reward_loss += loss_reward.item()
        
        mean_loss = total_loss / len(self.train_loader)
        mean_recon_loss = total_recon_loss / len(self.train_loader)
        mean_state_loss = total_state_loss / len(self.train_loader)
        mean_reward_loss = total_reward_loss / len(self.train_loader)

        return mean_loss, mean_recon_loss, mean_state_loss, mean_reward_loss

    def evaluate_holdout(self, num_rollouts=20):
        """Evaluate model on holdout/validation set."""
        self.model.eval()
        total_loss, total_recon, total_state, total_reward = 0, 0, 0, 0

        for batch in self.holdout_loader:
            obss, actions, next_obss, rewards = [x.to(self.device) for x in batch]

            loss, loss_recon, loss_state, loss_reward = self.model.compute_loss(
                    obss, actions, next_obss, rewards, self.dt, self.alpha, num_rollouts=num_rollouts)

            with torch.no_grad():
                total_loss += loss.item()
                total_recon += loss_recon.item()
                total_state += loss_state.item()
                total_reward += loss_reward.item()

        mean_loss = total_loss / len(self.holdout_loader)
        mean_recon = total_recon / len(self.holdout_loader)
        mean_state = total_state / len(self.holdout_loader)
        mean_reward = total_reward / len(self.holdout_loader)

        return mean_loss, mean_recon, mean_state, mean_reward

    def stochastic_evaluate_holdout(self, num_samples=10):
        """
        Evaluate holdout loss stochastically by resampling Brownian noise.

        Args:
            num_samples: how many stochastic rollouts per (s,a) pair to sample.

        Returns:
            dict with mean ± std for [total, recon, state, reward] losses
        """
        self.model.eval()

        all_total, all_recon, all_state, all_reward = [], [], [], []

        for batch in self.holdout_loader:
            obss, actions, next_obss, rewards = [x.to(self.device) for x in batch]

            # Repeat evaluation with multiple stochastic rollouts
            for _ in range(num_samples):
                loss, loss_recon, loss_state, loss_reward = self.model.compute_loss(
                    obss, actions, next_obss, rewards, self.dt, self.alpha,
                    num_rollouts=1  # <-- DO NOT average across rollouts
                )

                all_total.append(loss.item())
                all_recon.append(loss_recon.item())
                all_state.append(loss_state.item())
                all_reward.append(loss_reward.item())

        # Convert to tensors for stats
        all_total = torch.tensor(all_total)
        all_recon = torch.tensor(all_recon)
        all_state = torch.tensor(all_state)
        all_reward = torch.tensor(all_reward)

        result = {
            "total_mean": all_total.mean().item(),
            "total_std": all_total.std().item(),
            "recon_mean": all_recon.mean().item(),
            "recon_std": all_recon.std().item(),
            "state_mean": all_state.mean().item(),
            "state_std": all_state.std().item(),
            "reward_mean": all_reward.mean().item(),
            "reward_std": all_reward.std().item(),
        }

        # print("Stochastic Holdout Evaluation:")
        # for k, v in result.items():
        #     print(f"{k}: {v:.6f}")

        return result

    def train(self, 
            weights = None,
            num_epochs=1000, 
            wandb = None, 
            tensorboard_writer = None, 
            patience=5,  # stop if no improvement for 20 epochs
            save_path="best_model.pth",
            improvement_threshold = 0.01, # 1% (how much "relative" improvement we require)
            ):
    
        best_holdout_loss = float('inf')
        patience_counter = 0
        for epoch in range(num_epochs):
            train_loss, train_recon, train_state, train_reward = self.train_one_epoch(weights)
            val_loss, val_recon, val_state, val_reward = self.evaluate_holdout()

            # Logging
            if wandb is not None:
                wandb.log({
                    "epoch": epoch,
                    "loss/train_total": train_loss,
                    "loss/train_recon": train_recon,
                    "loss/train_state":train_state,
                    "loss/train_reward": train_reward,
                    "loss/val_total": val_loss,
                    "loss/val_recon": val_recon,
                    "loss/val_state":val_state,
                    "loss/val_reward": val_reward,
                })
            else:
                tensorboard_writer.add_scalar("loss/train_total", train_loss, epoch)
                tensorboard_writer.add_scalar("loss/train_recon", train_recon, epoch)
                tensorboard_writer.add_scalar("loss/train_state", train_state, epoch)
                tensorboard_writer.add_scalar("loss/train_reward", train_reward, epoch)
                tensorboard_writer.add_scalar("loss/val_total", val_loss, epoch)
                tensorboard_writer.add_scalar("loss/val_recon", val_recon, epoch)
                tensorboard_writer.add_scalar("loss/val_state", val_state, epoch)
                tensorboard_writer.add_scalar("loss/val_reward", val_reward, epoch)

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            # print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {mean_total_loss:.4f}, Recon Loss: {mean_recon_loss:.4f}, State Loss: {mean_state_loss:.4f}, Reward Loss: {mean_reward_loss:.4f}")

            # Early stopping + checkpointing
            # New loss must be at least 1% lower than the best so far
            # relative_improvement = (best_holdout_loss - val_loss) / best_holdout_loss > 0.01
            # val_loss < best_holdout_loss - 0.01*best_holdout_loss
            # val_loss < best_holdout_loss*(1-0.01)
            if val_loss < best_holdout_loss * (1 - improvement_threshold):  # >1% improvement
                # # significant improvement
                best_holdout_loss = val_loss
                patience_counter = 0

                # Save model checkpoint, scalar
                # self.save(epoch, val_loss, save_path)
                self.save(save_path)
            else:
                # No improvement
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. "
                      f"No improvement in {patience} epochs.")
                break

        print(f"Best model was saved at {save_path} with Val Loss {best_holdout_loss:.4f}")

    def save(self, save_path):
        torch.save({
            #"epoch": epoch,
            #"val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, os.path.join(save_path, "best_model.pth"))

        # Save scalers
        # joblib.dump(self.obs_scaler, os.path.join(save_path, "obs_scaler.pkl"))
        # joblib.dump(self.act_scaler, os.path.join(save_path, "act_scaler.pkl"))
        # self.obs_scaler.save_scaler(save_path)
        # self.act_scaler.save_scaler(save_path)
        self.obs_scaler.save_scaler_combined(save_path)
        self.act_scaler.save_scaler_combined(save_path)
        self.rew_scaler.save_scaler_combined(save_path)

        # print(f"Checkpoint saved at epoch {epoch+1} with Val Loss {val_loss:.4f}")

    def load(self, load_path):
        # Load model
        checkpoint = torch.load(os.path.join(load_path, "best_model.pth"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scalers
        # obs_scaler = joblib.load(os.path.join(save_path, "obs_scaler.pkl"))
        # act_scaler = joblib.load(os.path.join(save_path, "act_scaler.pkl"))
        # self.obs_scaler.load_scaler(load_path)
        # self.act_scaler.load_scaler(load_path)
        self.obs_scaler.load_scaler_combined(load_path)
        self.act_scaler.load_scaler_combined(load_path)
        self.rew_scaler.load_scaler_combined(load_path)

        # Load saved indices
        train_idx = np.load(os.path.join(load_path, "train_idx.npy"))
        holdout_idx = np.load(os.path.join(load_path, "holdout_idx.npy"))

        return train_idx, holdout_idx

    def evaluate_multistep_rollout(self, horizons=[5, 10, 20, 50], num_rollouts=100):
        """
        Evaluate multi-step rollout prediction error of the dynamics model.

        Args:
            model: Trained dynamics model.
            horizon: List of rollout horizons to test (e.g., [5, 10, 20, 50]).
            dt: Integration timestep used during rollout.
            num_rollouts: Number of random rollouts sampled from dataset for evaluation.

        Returns:
            mse_dict: Dict {horizon: (scaled_mse, real_mse)} for each horizon. Mean Squared Error across horizon steps

            #rollout_preds: Predicted rollout trajectories.
            #rollout_truth: Ground-truth rollout trajectories.
        """
        self.model.eval()

        # obss = torch.tensor(self.data["observations"], dtype=torch.float32).to(self.device)
        # actions = torch.tensor(self.data["actions"], dtype=torch.float32).to(self.device)
        # next_obss = torch.tensor(self.data["next_observations"], dtype=torch.float32).to(self.device)

        obss, actions, next_obss, rewards = self.model.format_samples_for_training(self.data)

        data_size = obss.shape[0]

        obss = self.obs_scaler.transform(obss) 
        actions = self.act_scaler.transform(actions) 
        next_obss = self.obs_scaler.transform(next_obss) # same obs scaler 

        obss = torch.tensor(obss, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        next_obss = torch.tensor(next_obss, dtype=torch.float32).to(self.device)

        mse_dict = {}

        for horizon in horizons:
            rollout_preds_all, rollout_truth_all = [], []
            
            # collect rollouts
            for _ in range(num_rollouts):
                # Random starting index (ensure enough horizon steps ahead exist)
                idx = torch.randint(0, data_size - horizon - 1, (1,)).item()

                s_seq = obss[idx : idx + horizon + 1]     # [horizon+1, state_dim]
                a_seq = actions[idx : idx + horizon]      # [horizon, action_dim]

                # Ground-truth rollout (skip initial state)
                rollout_truth = s_seq[1:]                 # [horizon, state_dim]

                # Predict rollout
                s_pred = s_seq[0].unsqueeze(0)            # initial state [1, state_dim]
                rollout_pred = []
                
                for t in range(horizon):
                    # allow gradients inside predict_state_reward (Hamiltonian dynamics needs autograd)
                    s_pred, _ = self.model.predict_state_reward(s_pred, a_seq[t].unsqueeze(0), self.dt)
                    rollout_pred.append(s_pred.squeeze(0))  # remove batch dim

                rollout_pred = torch.stack(rollout_pred)   # [horizon, state_dim]

                rollout_preds_all.append(rollout_pred)
                rollout_truth_all.append(rollout_truth)

            with torch.no_grad():
                # Normalized/Scaled Space
                rollout_preds_all = torch.stack(rollout_preds_all)   # [num_rollouts, horizon, state_dim]
                rollout_truth_all = torch.stack(rollout_truth_all)   # [num_rollouts, horizon, state_dim]
                # Compute scaled-space error 
                mse_rollout_norm = F.mse_loss(rollout_preds_all, rollout_truth_all)

                mse_dict[horizon] = mse_rollout_norm
                print(f"Rollout Horizon {horizon}: Scaled MSE={mse_rollout_norm:.6f}")

        return mse_dict

    def evaluate_multistep_rollout_with_variance(self, horizons=[5, 10, 20, 50], num_rollouts=100):
        """
        Evaluate multi-step rollout prediction error of the dynamics model.
        Returns both mean and variance across multiple rollouts.

        ######## HOLDOUT DATASET USED ######## 

        Args:
            horizons: list of rollout horizons to test
            num_rollouts: number of random rollouts sampled

        Returns:
            mse_dict: {horizon: {
                "scaled_mean": ..., "scaled_std": ...,
            }}
        """
        # gait, durations, body_roll, stance_width, stance_length, aux_reward
        self.constant_idx = [8, 9, 10, 11, 14, 15, 16, 17]  

        self.model.eval()

        # Unpack tensors from self.holdout_dataset 
        obss, actions, next_obss, rewards = [
            tensor.clone().to(self.device) for tensor in self.holdout_dataset.tensors
        ]
        data_size = obss.shape[0]

        mse_dict = {}

        for horizon in horizons:
            rollout_s_preds_all, rollout_s_truth_all = [], []
            rollout_r_preds_all, rollout_r_truth_all = [], []

            # collect rollouts
            for _ in range(num_rollouts):
                idx = torch.randint(0, data_size - horizon - 1, (1,)).item()

                s_seq = obss[idx : idx + horizon + 1]
                a_seq = actions[idx : idx + horizon]
                r_seq = rewards[idx+1 : idx+horizon+1]

                rollout_s_truth = s_seq[1:]  # ground-truth rollout
                s_pred = s_seq[0].unsqueeze(0)

                rollout_s_pred, rollout_r_pred = [], []
                for t in range(horizon):
                    s_pred, r_pred = self.model.predict_state_reward(
                        s_pred, a_seq[t].unsqueeze(0), self.dt
                    )
                    rollout_s_pred.append(s_pred.squeeze(0))
                    rollout_r_pred.append(r_pred.squeeze(0))

                rollout_s_preds_all.append(torch.stack(rollout_s_pred))
                rollout_r_preds_all.append(torch.stack(rollout_r_pred))
                rollout_s_truth_all.append(rollout_s_truth)
                rollout_r_truth_all.append(r_seq)

            # stack rollouts
            rollout_s_preds_all = torch.stack(rollout_s_preds_all)   # [num_rollouts, horizon, state_dim]
            rollout_s_truth_all = torch.stack(rollout_s_truth_all)
            rollout_r_preds_all = torch.stack(rollout_r_preds_all)   # [num_rollouts, horizon, 1]
            rollout_r_truth_all = torch.stack(rollout_r_truth_all)

            with torch.no_grad():
                # Scaled Space
                # State Evaluation
                mse_scaled_per_rollout = F.mse_loss(
                    rollout_s_preds_all, rollout_s_truth_all, reduction="none"
                ).mean(dim=(1, 2))  # [num_rollouts]

                s_scaled_mean = mse_scaled_per_rollout.mean().item()
                s_scaled_std = mse_scaled_per_rollout.std().item()

                # Reward Evaluation
                mse_r_scaled_per_rollout = F.mse_loss(
                    rollout_r_preds_all, rollout_r_truth_all, reduction="none"
                ).mean(dim=(1, 2))  # [num_rollouts]

                r_scaled_mean = mse_r_scaled_per_rollout.mean().item()
                r_scaled_std = mse_r_scaled_per_rollout.std().item()

                # Real (inverse-transform) space
                rollout_s_preds_real = self.obs_scaler.inverse_transform(
                    rollout_s_preds_all.cpu().numpy().reshape(-1, obss.shape[-1])
                ).reshape(num_rollouts, horizon, -1)

                rollout_s_truth_real = self.obs_scaler.inverse_transform(
                    rollout_s_truth_all.cpu().numpy().reshape(-1, obss.shape[-1])
                ).reshape(num_rollouts, horizon, -1)

                # Drop constant features
                rollout_s_preds_real = np.delete(rollout_s_preds_real, self.constant_idx, axis=2)
                rollout_s_truth_real = np.delete(rollout_s_truth_real, self.constant_idx, axis=2)

                rollout_s_preds_real = torch.tensor(rollout_s_preds_real, dtype=torch.float32)
                rollout_s_truth_real = torch.tensor(rollout_s_truth_real, dtype=torch.float32)

                # Reward (In Real Space)
                rollout_r_preds_real = self.rew_scaler.inverse_transform(
                    rollout_r_preds_all.cpu().numpy().reshape(-1, rewards.shape[-1])
                ).reshape(num_rollouts, horizon, -1)

                rollout_r_truth_real = self.rew_scaler.inverse_transform(
                    rollout_r_truth_all.cpu().numpy().reshape(-1, rewards.shape[-1])
                ).reshape(num_rollouts, horizon, -1)

                rollout_r_preds_real = torch.tensor(rollout_r_preds_real, dtype=torch.float32)
                rollout_r_truth_real = torch.tensor(rollout_r_truth_real, dtype=torch.float32)

                # Computed in scaled space
                # State
                #self.plot_global_rollout_error_curve(rollout_s_preds_all, rollout_s_truth_all, horizons)
                # Reward
                #self.plot_global_rollout_error_curve(rollout_r_preds_all, rollout_r_truth_all, horizons, title="Global Reward Rollout Error Curve"               # Plot featurewise errors
                
                self.plot_global_state_reward_error_curves(rollout_s_preds_all, rollout_s_truth_all, rollout_r_preds_all, rollout_r_truth_all, horizons)

                # Computed in real/unnormalized space
                # self.plot_featurewise_rollout_errors(rollout_s_preds_real, rollout_s_truth_real, horizons, save_csv_path="featurewise_mse.csv")
                self.plot_featurewise_rollout_errors(rollout_s_preds_real, rollout_s_truth_real, horizons, 
                                                    reward_preds=rollout_r_preds_real,
                                                    reward_truth=rollout_r_truth_real,
                                                    save_csv_path="featurewise_mse.csv")

            mse_dict[horizon] = {
                "state_scaled_mean": s_scaled_mean,
                "state_scaled_std": s_scaled_std,
                "reward_scaled_mean": r_scaled_mean,
                "reward_scaled_std": r_scaled_std,
            }

            print(
                f"Horizon={horizon}: \n"
                f"\tState Scaled MSE= [{s_scaled_mean:.6f} ± {s_scaled_std:.6f}]\n"
                f"\tReward Scaled MSE= [{r_scaled_mean:.6f} ± {r_scaled_std:.6f}]"
            )

        return mse_dict

    def plot_global_state_reward_error_curves(self, state_preds, state_truth, reward_preds, reward_truth, horizons):
        errors_state = (state_preds - state_truth).pow(2)   # [N, H, D]
        errors_reward = (reward_preds - reward_truth).pow(2)  # [N, H, 1]

        horizon_axis = np.arange(1, errors_state.shape[1] + 1)

        # State global error
        mean_state = errors_state.mean((0, 2)).cpu().numpy()
        std_state  = errors_state.mean(2).std(0).cpu().numpy()

        # Reward global error
        mean_reward = errors_reward.mean((0, 2)).cpu().numpy()
        std_reward  = errors_reward.mean(2).std(0).cpu().numpy()

        plt.figure(figsize=(7,5))

        plt.plot(horizon_axis, mean_state, label="State Mean MSE", color="blue")
        plt.fill_between(horizon_axis, mean_state-std_state, mean_state+std_state, alpha=0.3, color="blue")

        plt.plot(horizon_axis, mean_reward, label="Reward Mean MSE", color="red")
        plt.fill_between(horizon_axis, mean_reward-std_reward, mean_reward+std_reward, alpha=0.3, color="red")

        plt.title("Global Rollout Error Curves (State vs Reward)")
        plt.xlabel("Horizon")
        plt.ylabel("Mean Squared Error")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_featurewise_rollout_errors(self, preds, truth, horizons, 
                                        reward_preds=None, reward_truth=None, save_csv_path=None):
        """
        Feature-wise error plots
        Plot per-feature rollout prediction errors with variance bands.

        Args:
            preds: torch.Tensor, shape [num_rollouts, horizon, state_dim] predicted rollout
            truth: torch.Tensor, shape [num_rollouts, horizon, state_dim] ground-truth rollout
            reward_preds: torch.Tensor, shape [num_rollouts, horizon, 1], predicted rewards (real space)
            reward_truth: torch.Tensor, shape [num_rollouts, horizon, 1], ground-truth rewards (real space)
            horizons: int, horizon length used in rollouts
            num_rollouts: int, number of rollouts
        """

        # State feature slices (constants removed)
        feature_slices = {
            "gravity_vector": slice(0, 3),
            "x_vel": slice(3, 4),
            "y_vel": slice(4, 5),
            "yaw_vel": slice(5, 6),
            "body_height": slice(6, 7),
            "step_freq": slice(7, 8),
            "footswing_height": slice(8, 9),
            "body_pitch": slice(9, 10),
            "dof_pos": slice(10, 22),
            "dof_vel": slice(22, 34),
            "actions": slice(34, 46),
            "clock_inputs": slice(46, 50),
        }

        # Compute squared errors per-dim for states
        errors = (preds - truth).pow(2)   # [N, H, D]
        N, H, D = errors.shape

        # If reward is given, compute squared error
        if reward_preds is not None and reward_truth is not None:
            reward_errors = (reward_preds - reward_truth).pow(2)  # [N, H, 1]
            feature_slices["reward"] = slice(D, D+1)  # virtual index for plotting
            # Concatenate reward errors with state errors so plotting loop works uniformly
            errors = torch.cat([errors, reward_errors], dim=2)

        # Horizon axis
        horizon_axis = np.arange(1, H + 1)

        n_features = len(feature_slices)
        ncols = 3
        nrows = (n_features + ncols - 1) // ncols  # ceil division

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3*nrows), sharex=True)
        axes = axes.ravel()  # flatten so we can index like 1D

        feature_stats = {}
        data_dict = {"horizon": horizon_axis}  # For CSV

        for ax, (feat, sl) in zip(axes, feature_slices.items()):
            # Select dims corresponding to this feature
            feat_err = errors[:, :, sl]   # [N, H, dim_of_feature]
            feat_err = feat_err.mean(-1)  # avg over dims → [N, H]

            # Mean & std across rollouts
            mean_curve = feat_err.mean(0).cpu().numpy()
            std_curve = feat_err.std(0).cpu().numpy()

            feature_stats[feat] = (mean_curve, std_curve)

            # Add mean_curve to data_dict for CSV
            data_dict[f"{feat}_mean"] = mean_curve
            data_dict[f"{feat}_std"] = std_curve

            # Plot
            ax.plot(horizon_axis, mean_curve, label=f"{feat} error")
            ax.fill_between(horizon_axis, mean_curve - std_curve, mean_curve + std_curve,
                            alpha=0.3)
            ax.set_title(feat, pad=-10)
            ax.set_ylabel("MSE")
            ax.grid(True)

        # Hide unused plots
        for i in range(len(feature_slices), len(axes)):
            fig.delaxes(axes[i])

        axes[-1].set_xlabel("Horizon")
        plt.tight_layout()
        plt.show()

        # Save CSV if requested
        if save_csv_path:
            df = pd.DataFrame(data_dict)
            df.to_csv(os.path.join(self.log_dirs, save_csv_path), index=False)
            print(f"Saved feature-wise rollout errors to {os.path.join(self.log_dirs, save_csv_path)}")

    def plot_global_rollout_error_curve(self, preds, truth, horizons, title="Global State Rollout Error Curve"):
        # Compute squared errors per-dim
        errors = (preds - truth).pow(2)   # [N, H, D]
        N, H, D = errors.shape

        # Horizon axis
        horizon_len = errors.shape[1]
        horizon_axis = np.arange(1, horizon_len + 1)

        mean_global = errors.mean((0, 2)).cpu().numpy()  # avg over rollouts+features -> [H]
        std_global  = errors.mean(2).std(0).cpu().numpy()  # std across rollouts, averaged over features

        plt.figure(figsize=(7,5))
        plt.plot(horizon_axis, mean_global, label="Global Mean MSE", color="blue")
        plt.fill_between(horizon_axis, mean_global-std_global, mean_global+std_global,
                         alpha=0.3, color="blue")
        plt.title(title)
        plt.xlabel("Horizon")
        plt.ylabel("Mean Squared Error")
        plt.grid(True)
        plt.legend()
        plt.show()

        print("Summary at selected horizons:")
        for h in horizons:
            if h <= H:
                print(f"H={h}: Mean MSE={mean_global[h-1]:.6f} ± {std_global[h-1]:.6f}")

    def plot_rollout_mse_with_variance(self, horizons, scaled_stats, num_rollouts=50):
        mse_scaled_means, mse_scaled_stds = scaled_stats
        # mse_real_means, mse_real_stds = real_stats

        plt.figure(figsize=(8, 6))

        # Plot with shaded variance bands
        plt.plot(horizons, mse_scaled_means, label="Scaled MSE", marker='o')
        plt.fill_between(horizons,
                         np.array(mse_scaled_means) - np.array(mse_scaled_stds),
                         np.array(mse_scaled_means) + np.array(mse_scaled_stds),
                         alpha=0.2)

        # plt.plot(horizons, mse_real_means, label="Real-space MSE", marker='o')
        # plt.fill_between(horizons,
        #                  np.array(mse_real_means) - np.array(mse_real_stds),
        #                  np.array(mse_real_means) + np.array(mse_real_stds),
        #                  alpha=0.2)

        plt.xlabel("Rollout Horizon (steps)")
        plt.ylabel("Mean Squared Error")
        plt.title(f"Rollout Prediction Error Growth\n num_rollouts: {num_rollouts}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_with_stats(self, horizons=[20], num_rollouts=20, dataset_stats=None):
        """
        Evaluate model rollout errors and compare with dataset-level statistics.

        Args:
            horizons: list of rollout horizons to test
            num_rollouts: number of stochastic rollouts
            dataset_stats: dict of {feature: {"mean": np.array, "std": np.array}}
                           (from your global stats computation)
        """
        import pandas as pd
        feature_slices = {
            "gravity_vector": slice(0, 3),
            "x_vel": slice(3, 4),
            "y_vel": slice(4, 5),
            "yaw_vel": slice(5, 6),
            "body_height": slice(6, 7),
            "step_freq": slice(7, 8),
            "gait": slice(8, 11),
            "durations": slice(11, 12),
            "footswing_height": slice(12, 13),
            "body_pitch": slice(13, 14),
            "body_roll": slice(14, 15),
            "stance_width": slice(15, 16),
            "stance_length": slice(16, 17),
            "aux_reward": slice(17, 18),
            "dof_pos": slice(18, 30),
            "dof_vel": slice(30, 42),
            "actions": slice(42, 54),
            "clock_inputs": slice(54, 58),
        }
        self.model.eval()
        obss, actions, next_obss, rewards = [
            tensor.clone().to(self.device) for tensor in self.holdout_dataset.tensors
        ]
        data_size = obss.shape[0]

        results = []

        for horizon in horizons:
            preds_all, truth_all = [], []

            for _ in range(num_rollouts):
                idx = torch.randint(0, data_size - horizon - 1, (1,)).item()
                s_seq = obss[idx : idx + horizon + 1]
                a_seq = actions[idx : idx + horizon]

                rollout_truth = s_seq[1:]
                s_pred = s_seq[0].unsqueeze(0)

                rollout_pred = []
                for t in range(horizon):
                    s_pred, _ = self.model.predict_state_reward(
                        s_pred, a_seq[t].unsqueeze(0), self.dt
                    )
                    rollout_pred.append(s_pred.squeeze(0))
                preds_all.append(torch.stack(rollout_pred))
                truth_all.append(rollout_truth)

            preds_all = torch.stack(preds_all)   # [N, H, D]
            truth_all = torch.stack(truth_all)

            # Flatten over rollouts and horizon
            preds_flat = preds_all.reshape(-1, preds_all.shape[-1]).detach().cpu().numpy()
            truth_flat = truth_all.reshape(-1, truth_all.shape[-1]).detach().cpu().numpy()

            # Compare per feature group
            for feat, sl in feature_slices.items():
                pred_feat = preds_flat[:, sl]
                true_feat = truth_flat[:, sl]

                # Means
                pred_mean = pred_feat.mean(axis=0)
                true_mean = true_feat.mean(axis=0)

                # Bias (relative to dataset mean if given)
                if dataset_stats is not None:
                    dataset_mean = dataset_stats[feat]["mean"]
                    dataset_std = dataset_stats[feat]["std"]
                else:
                    dataset_mean = true_mean
                    dataset_std = true_feat.std(axis=0)

                bias = np.abs(pred_mean - dataset_mean)

                # RMSE
                rmse = np.sqrt(((pred_feat - true_feat) ** 2).mean(axis=0))
                rel_rmse = rmse / (dataset_std + 1e-8)

                # Save results
                for i in range(len(pred_mean)):
                    results.append({
                        "feature": feat,
                        "dim": i,
                        "dataset_mean": dataset_mean[i],
                        "pred_mean": pred_mean[i],
                        "bias": bias[i],
                        "dataset_std": dataset_std[i],
                        "rmse": rmse[i],
                        "rmse/std": rel_rmse[i]
                    })

        df = pd.DataFrame(results)
        return df

def train_dynamics_model():
    import argparse
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="aliengo")
    parser.add_argument("--seed", type=int, default=42)
    # Auto-Encoder
    parser.add_argument("--input_dim", type=int, default=24)
    parser.add_argument("--action_dim", type=int, default=12)
    parser.add_argument("--latent_dim", type=int, default=2*12) # 2*K canonical states (q, p), K is DoF
    # HNN-SDE Trainer
    parser.add_argument("--lr", type=float, default=3e-4)       # learning rate
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.02)         # from control_dt (0.02 => 50 Hz)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_load_path", type=str, default="mbrl_dynamics_net/dataset/PreprocessedDataset/train")
    parser.add_argument("--preprocess", type=bool, default=True)
    parser.add_argument('--run_name', type=str, default=f"HNNSDE_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help='used for logging to distingush different runs')
    # parser.add_argument('--run_name', type=str, default=f"HNNSDE", help='used for logging')
    parser.add_argument('--retrain', type=bool, default=True, help='flag to initiate training')
    parser.add_argument('--horizons', type=int, nargs='*', default=[20], help='List of rollout horizons to test')

    args = parser.parse_args()

    config = {
        "dynamic_module": {
            "input_dim": args.input_dim,
            "action_dim": args.action_dim,
            "latent_dim": args.latent_dim,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dt": args.dt,
            "alpha": args.alpha,
            "class": "HNN-SDE",
            },
        "meta": {
            "device": args.device,
            "seed": args.seed,
            "data_load_path": args.data_load_path,
            "preprocess": args.preprocess,
            "run_name": args.run_name,
            }
        }
    print(config)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Clear unused GPU memory
    torch.cuda.empty_cache()

    # Aliengo Offline Data
    data = OfflineDatasetLoader().get_dataset(args.data_load_path, preprocess=args.preprocess)
    for key, value in data.items():
        print(f"{key}: {np.array(value).shape}")

    use_wandb = True
    try:
        # Attempt to initialize wandb and start tracking
        wandb.init(
            project="anubhav1772-itmo-university",
            name=args.run_name,
            config = config,
            resume="never", # fresh run
        )
        print("W&B initialized successfully")
    except wandb.errors.errors.CommError as e:
        # In case of an error with wandb, catch the exception and use TensorBoard instead
        print(f"W&B error occurred: {e}. \nUsing TensorBoard for logging instead.")
        use_wandb = False

        # Generate dynamic log directory using timestamp
        # tensorboard_log_dir = os.path.join("runs", f"DYN_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        # Logger
        log_dirs = make_log_dirs(args.task, 'dynamics', args.seed, vars(args), run_name=args.run_name)
        print(f"log_dirs = {log_dirs}")
        tensorboard_writer = SummaryWriter(log_dir=os.path.join(log_dirs, "tensorboard"))

        model = HNNSDE(args.input_dim, args.latent_dim, args.action_dim, device=args.device)
        hnnsde_trainer = HNNSDETrainer(model, data, 
                              batch_size=args.batch_size, 
                              lr=args.lr, 
                              dt=args.dt, 
                              alpha=args.alpha,
                              log_dirs=log_dirs, 
                              retrain=args.retrain,
                              device=args.device)        

        if args.retrain or not os.path.isfile(os.path.join(log_dirs, "best_model.pth")):
            print("Training from scratch...")

            # Only dof_pos (indices 18-29) and dof_vel (indices 30-41) contribute
            weights = torch.zeros(args.input_dim, device=args.device)
            weights[18:30] = 1.0   # dof_pos (12 dims)
            weights[30:42] = 1.0   # dof_vel (12 dims)

            # Normalize to keep loss scale stable
            weights = weights / weights.mean()

            if use_wandb:
                noda_trainer.train(
                    weights=weights,
                    num_epochs=args.num_epochs, 
                    wandb=wandb, 
                    save_path=log_dirs
                )
            else:
                noda_trainer.train(
                    weights=weights,
                    num_epochs=args.num_epochs, 
                    tensorboard_writer=tensorboard_writer,
                    save_path=log_dirs
                )
                # Ensures all logs are written
                tensorboard_writer.flush()   
                tensorboard_writer.close()

        else:
            print("Using pretrained dynamics...")

    # stats = noda_trainer.stochastic_evaluate_holdout(num_samples=20)
    # print("Stochastic Holdout Evaluation:")
    # for k, v in stats.items():
    #     print(f"{k}: {v:.6f}")

    # mse_dict = noda_trainer.evaluate_multistep_rollout(horizons=args.horizons, num_rollouts=50)
    
    # mse_rollout_norm, mse_rollout_real = [], []
    # for v in mse_dict.values():
    #     mse_rollout_norm.append(v[0].item())
    #     mse_rollout_real.append(v[1].item())

    # plt.figure(figsize=(8,5))
    # plt.plot(args.horizons, mse_rollout_norm, marker="o", label="Scaled MSE")
    # plt.plot(args.horizons, mse_rollout_real, marker="s", label="Real MSE")

    # plt.xlabel("Rollout Horizon")
    # plt.ylabel("MSE")
    # plt.title("Multi-step Rollout MSE vs Horizon")
    # plt.legend()
    # plt.grid(True)

    # plt.show()

    ###########################
    # mse_dict = noda_trainer.evaluate_multistep_rollout_with_variance(horizons=args.horizons, num_rollouts=20)

    # mse_scaled_means, mse_scaled_stds = [], []

    # for horizon in args.horizons:
    #     mse_scaled_means.append(mse_dict[horizon]["state_scaled_mean"])
    #     mse_scaled_stds.append(mse_dict[horizon]["state_scaled_std"])

    # scaled_stats = (mse_scaled_means, mse_scaled_stds)

    # noda_trainer.plot_rollout_mse_with_variance(args.horizons, scaled_stats, num_rollouts=20)

    # stats_df = noda_trainer.evaluate_with_stats(dataset_stats=get_dataset_stats()) 
    # print(stats_df)


    # Encode state
    # q, p, u = autoencoder.encode(s_t)
     
    # actions           (12,)
    # observations      (58,)
    # next_observations (58,)
    # terminals         (1,)
    # rewards           (1,)

if __name__ == '__main__':
    train_dynamics_model()
