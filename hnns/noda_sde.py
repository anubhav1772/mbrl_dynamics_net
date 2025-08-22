# pip install torchsde (Python >=3.8 and PyTorch >=1.6.0)
# https://github.com/google-research/torchsde
# https://github.com/google-research/torchsde/blob/master/DOCUMENTATION.md
from torchsde import sdeint, SDEStratonovich

# pip install torchdiffeq
# from torchdiffeq import odeint

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader, TensorDataset
import sys
import os
# Add the parent folder of `mbrl_dynamics_net` to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import Dict, List, Union, Tuple, Optional, Callable
from mbrl_dynamics_net.utils.buffer import OfflineDatasetLoader
from mbrl_dynamics_net.utils.logger import make_log_dirs
from mbrl_dynamics_net.utils.scaler import StandardScaler
from torch.utils.tensorboard import SummaryWriter

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

        # Encoder: state -> [q, p]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

        # Decoder: [q, p] -> state
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, s):
        u = self.encoder(s)               # latent state: u = [q, p]

        # Decoding: Canonical states back to the state
        s_reconstructed = self.decoder(u)

        q, p = torch.chunk(u, 2, dim=-1)  # Split canonical variables
        return s_reconstructed, (q, p), u

    def encode(self, s):
        u = self.encoder(s)
        q, p = torch.chunk(u, 2, dim=-1)
        return q, p, u

    def decode(self, u):
        return self.decoder(u)

class HamiltonianSDE(SDEStratonovich):
    def __init__(self, latent_dim, action_dim):
        super().__init__(noise_type="diagonal")
        self.sde_type = "stratonovich"
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        assert latent_dim % 2 == 0, "latent_dim must be even"
        self.K = latent_dim // 2

        # Drift network 
        self.drift = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)  # scalar output
        )

        # Diffusion network
        self.diffusion = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)  # diagonal noise
        )

        # External force model Q(a)
        self.force = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.K)  # K = DoF
        )

    def f(self, t, u, a=None):
        if a is None:
            a = torch.zeros(u.shape[0], self.action_dim, device=u.device)
        
        q, p = torch.chunk(u, 2, dim=-1)

        # Ensure q and p require gradients
        q.requires_grad_(True)
        p.requires_grad_(True)

        # Compute Hamiltonian input
        H_in = torch.cat([q, p], dim=-1)  # H_in: (64, 24)

        H_scalar = self.drift(H_in)
        grads = torch.autograd.grad(
            outputs=H_scalar,
            inputs=(q, p),
            grad_outputs=torch.ones_like(H_scalar),
            retain_graph=True,
            create_graph=True
        )

        # Extract gradients for each sample in the batch
        dq_dt = grads[1]
        dp_dt = -grads[0] + self.force(a)

        du_dt = torch.cat([dq_dt, dp_dt], dim=-1)

        return du_dt

    def g(self, t, u, a=None):
        if a is None:
            a = torch.zeros(u.shape[0], self.action_dim, device=u.device)
        return self.diffusion(torch.cat([u, a], dim=-1))

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
    def __init__(self, base_sde, a_t):
        super().__init__(noise_type=base_sde.noise_type)
        self.sde_type = base_sde.sde_type
        self.base_sde = base_sde
        self.a_t = a_t
    
    def f(self, t, u):
        return self.base_sde.f(t, u, self.a_t)
    
    def g(self, t, u):
        return self.base_sde.g(t, u, self.a_t)

class RewardDecoder(nn.Module):
    def __init__(self, latent_dim, action_dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, q, p, a):
        u = torch.cat([q, p], dim=-1)
        return self.net(torch.cat([u, a], dim=-1))  # shape: [B, 1]

class NODA(nn.Module):
    def __init__(self, input_dim, latent_dim, action_dim, device='cpu') -> None:
        super().__init__()
        self.device = device
        self.autoencoder = AutoEncoder(input_dim, latent_dim).to(device)
        self.ode_func = HamiltonianSDE(latent_dim, action_dim).to(device)
        self.reward_decoder = RewardDecoder(latent_dim, action_dim).to(device)
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
            s_t          : Current state [batch, state_dim]
            a_t          : Current action [batch, action_dim]
            dt           : Integration step size
        """

        # Encode observation to canonical (q, p) and full latent u
        q, p, u = self.autoencoder.encode(s_t)

        # Predict reward from q, p, and a_t
        r_pred = self.reward_decoder(q, p, a_t)

        # Integrate canonical state forward one time step
        t_span = torch.tensor([0, dt], dtype=torch.float32, device=self.device)
        # u_traj = odeint(lambda t, u_: self.ode_func(t, u_, a_t),
        #                 u,
        #                 t_span,
        #                 method='rk4',
        #                 options={'step_size': dt})

        sde_with_action = ActionSDE(self.ode_func, a_t)

        u_traj = sdeint(sde_with_action,              # HamiltonianSDE subclass
                        u,                          # initial latent [q,p]
                        t_span,                     # tensor([0., dt])
                        method='heun',              # Milstein/Heun for Stratonovich
                        dt=dt,                      # integration step
                        # names={'drift': 'f', 
                        #        'diffusion': 'g'},   # match torchsde API
                        # args=(a_t,),                # pass action to drift & diffusion
                    )

        # next canonical state
        u_next = u_traj[-1]                         # Shape: [batch_size, latent_dim]

        # Decode back to predicted next observation
        s_t_plus1_pred = self.autoencoder.decode(u_next)
        return s_t_plus1_pred, r_pred

    # def compute_loss(self, s_t, a_t, s_tp1_true, r_true, dt, alpha):
    #     '''One-step prediction loss (MSE for state + reward)
    #     '''
    #     s_pred, r_pred = self.predict_state_reward(s_t, a_t, dt)
    #     # Canonical latent encoding
    #     _, _, u = self.autoencoder.encode(s_t)
    #     # Reconstruction from latent canonical encoding
    #     s_recon = self.autoencoder.decode(u)

    #     # Reconstruction loss
    #     loss_recon = F.mse_loss(s_recon, s_t)          
    #     # Next-state prediction loss
    #     loss_state = F.mse_loss(s_pred, s_tp1_true)
    #     # Reward prediction loss
    #     loss_reward = F.mse_loss(r_pred, r_true)

    #     # Combined training loss
    #     # As a convex combination of the state loss and the reward loss
    #     total_loss = alpha * (loss_recon + loss_state) + (1 - alpha) * loss_reward
    #     return total_loss, loss_recon, loss_state, loss_reward

    def compute_loss(self, s_t, a_t, s_tp1_true, r_true, dt, alpha, num_rollouts=5):
        """Compute the training loss for the stochastic Hamiltonian SDE dynamics model.

        This method evaluates how well the model predicts the next state and reward
        given the current state and action, while accounting for the stochasticity
        in the dynamics. To reduce the variance introduced by random Brownian noise,
        it performs multiple stochastic rollouts and averages the predictions.
        """
        rollout_preds = []
        reward_preds = []

        # Multi-rollout averaging
        for _ in range(num_rollouts):
            s_pred, r_pred = self.predict_state_reward(s_t, a_t, dt)
            rollout_preds.append(s_pred)
            reward_preds.append(r_pred)

        s_pred_mean = torch.stack(rollout_preds, dim=0).mean(dim=0)
        r_pred_mean = torch.stack(reward_preds, dim=0).mean(dim=0)

        # Canonical latent encoding
        _, _, u = self.autoencoder.encode(s_t)
        # Reconstruction from latent canonical encoding
        s_recon = self.autoencoder.decode(u)

        # State reconstruction loss (from autoencoder)
        loss_recon = F.mse_loss(s_recon, s_t)
        # Next-state prediction loss
        loss_state = F.mse_loss(s_pred_mean, s_tp1_true)
        # Reward prediction loss
        loss_reward = F.mse_loss(r_pred_mean, r_true)

        # Combined training loss
        # As a convex combination of the state loss and the reward loss
        total_loss = alpha * (loss_recon + loss_state) + (1 - alpha) * loss_reward
        return total_loss, loss_recon, loss_state, loss_reward

class NODATrainer:
    def __init__(self, model, data, batch_size, lr, dt, alpha, holdout_ratio=0.15, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.dt = dt
        self.alpha = alpha
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Prepare data for batching (convert numpy arrays to torch tensors)
        obss, actions, next_obss, rewards = self.model.format_samples_for_training(data)
        
        # self.dataset =  TensorDataset(torch.tensor(obss, dtype=torch.float32),
        #                                 torch.tensor(actions, dtype=torch.float32),
        #                                 torch.tensor(next_obss, dtype=torch.float32),
        #                                 torch.tensor(rewards, dtype=torch.float32))

        data_size = obss.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size

        # train_dataset, holdout_dataset = random_split(self.dataset, [train_size, holdout_size])
        # Seed for reproducibility
        # g = torch.Generator().manual_seed(42)
        # train_dataset, holdout_dataset = random_split(self.dataset, [train_size, holdout_size], generator=g)

        indices = np.arange(data_size)
        np.random.shuffle(indices)
        train_idx, holdout_idx = indices[:train_size], indices[train_size:]

        # Initialize scalers 
        # # StandardScaler for normalizing inputs
        self.obs_scaler = StandardScaler(name="obs") 
        self.act_scaler = StandardScaler(name="act") 
        # Already applied Min-Max scaling on reward in buffer
        # self.rew_scaler = StandardScaler() 

        # Fit on train split 
        self.obs_scaler.fit(obss[train_idx]) 
        self.act_scaler.fit(actions[train_idx]) 
        # self.rew_scaler.fit(rewards[train_idx]) 

        # Transform both train + holdout 
        obss = self.obs_scaler.transform(obss) 
        actions = self.act_scaler.transform(actions) 
        next_obss = self.obs_scaler.transform(next_obss) # same obs scaler 
        # rewards = self.rew_scaler.transform(rewards)

        train_dataset = TensorDataset(
            torch.tensor(obss[train_idx], dtype=torch.float32),
            torch.tensor(actions[train_idx], dtype=torch.float32),
            torch.tensor(next_obss[train_idx], dtype=torch.float32),
            torch.tensor(rewards[train_idx], dtype=torch.float32),
        )

        holdout_dataset = TensorDataset(
            torch.tensor(obss[holdout_idx], dtype=torch.float32),
            torch.tensor(actions[holdout_idx], dtype=torch.float32),
            torch.tensor(next_obss[holdout_idx], dtype=torch.float32),
            torch.tensor(rewards[holdout_idx], dtype=torch.float32),
        )

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.holdout_loader = DataLoader(holdout_dataset, batch_size=64, shuffle=False)

    def train_one_epoch(self):
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
                obss, actions, next_obss, rewards, self.dt, self.alpha
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

    # def train_one_epoch(self):
    #     total_loss = 0
    #     total_recon_loss = 0
    #     total_state_loss = 0
    #     total_reward_loss = 0

    #     # for name, param in self.model.named_parameters():
    #     #     print(name, param.data.mean().item(), param.grad is not None)

    #     for batch in self.train_loader:
    #         obss, actions, next_obss, rewards = batch

    #         obss, actions, next_obss, rewards = obss.to(self.device), actions.to(self.device), next_obss.to(self.device), rewards.to(self.device)

    #         # Zero the gradients
    #         self.optimizer.zero_grad()

    #         # Forward pass: compute the total loss (state + reward prediction loss)
    #         loss, loss_recon, loss_state, loss_reward = self.model.compute_loss(obss, actions, next_obss, rewards, self.dt, self.alpha)
            
    #         # Backpropagation and optimization
    #         loss.backward()
    #         self.optimizer.step()

    #         total_loss += loss.item()
    #         total_recon_loss += loss_recon.item()
    #         total_state_loss += loss_state.item()
    #         total_reward_loss += loss_reward.item()
        
    #     mean_loss = total_loss / len(self.dataloader)
    #     mean_recon_loss = total_recon_loss / len(self.dataloader)
    #     mean_state_loss = total_state_loss / len(self.dataloader)
    #     mean_reward_loss = total_reward_loss / len(self.dataloader)

    #     return mean_loss, mean_recon_loss, mean_state_loss, mean_reward_loss

    def evaluate_holdout(self):
        """Evaluate model on holdout/validation set."""
        self.model.eval()
        total_loss, total_recon, total_state, total_reward = 0, 0, 0, 0

        for batch in self.holdout_loader:
            obss, actions, next_obss, rewards = [x.to(self.device) for x in batch]

            loss, loss_recon, loss_state, loss_reward = self.model.compute_loss(
                    obss, actions, next_obss, rewards, self.dt, self.alpha)

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

    def train(self, 
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
            train_loss, train_recon, train_state, train_reward = self.train_one_epoch()
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

def evaluate_multistep_rollout(model, data, dt, horizon=50, device="cpu", num_rollouts=100):
    """
    Evaluate multi-step rollout prediction error of the dynamics model.

    Args:
        model: Trained dynamics model.
        horizon: Number of steps to rollout (e.g., 50, 100).
        dt: Integration timestep used during rollout.
        num_rollouts: Number of random rollouts sampled from dataset for evaluation.

    Returns:
        mse_rollout: Mean Squared Error across horizon steps.
        rollout_preds: Predicted rollout trajectories.
        rollout_truth: Ground-truth rollout trajectories.
    """
    model.eval()

    obss = torch.tensor(data["observations"], dtype=torch.float32).to(device)
    actions = torch.tensor(data["actions"], dtype=torch.float32).to(device)
    next_obss = torch.tensor(data["next_observations"], dtype=torch.float32).to(device)

    data_size = obss.shape[0]

    rollout_preds_all = []
    rollout_truth_all = []
    
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
            s_pred, _ = model.predict_state_reward(s_pred, a_seq[t].unsqueeze(0), dt)
            rollout_pred.append(s_pred.squeeze(0))  # remove batch dim

        rollout_pred = torch.stack(rollout_pred)   # [horizon, state_dim]

        rollout_preds_all.append(rollout_pred)
        rollout_truth_all.append(rollout_truth)

    with torch.no_grad():
        rollout_preds_all = torch.stack(rollout_preds_all)   # [num_rollouts, horizon, state_dim]
        rollout_truth_all = torch.stack(rollout_truth_all)   # [num_rollouts, horizon, state_dim]
        mse_rollout = F.mse_loss(rollout_preds_all, rollout_truth_all)

    return mse_rollout.item(), rollout_preds_all, rollout_truth_all

def train_dynamics_model():
    import argparse
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="aliengo")
    parser.add_argument("--seed", type=int, default=42)
    # Auto-Encoder
    parser.add_argument("--input_dim", type=int, default=58)    # state dim
    parser.add_argument("--action_dim", type=int, default=12)
    parser.add_argument("--latent_dim", type=int, default=2*12) # 2*K canonical states (q, p), K is DoF
    # NODA Trainer
    parser.add_argument("--lr", type=float, default=3e-4)       # learning rate
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.02)         # from control_dt (0.02 => 50 Hz)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_load_path", type=str, default="mbrl_dynamics_net/dataset/PreprocessedDataset/train")
    parser.add_argument("--preprocess", type=bool, default=True)
    parser.add_argument('--run_name', type=str, default=f"NODA_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help='used for logging to distingush different runs')

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
            "class": "NODA",
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
        tensorboard_writer = SummaryWriter(log_dir=log_dirs)

    model = NODA(args.input_dim, args.latent_dim, args.action_dim, device=args.device)
    noda_trainer = NODATrainer(model, data, 
                          batch_size=args.batch_size, 
                          lr=args.lr, 
                          dt=args.dt, 
                          alpha=args.alpha, 
                          device=args.device)

    if use_wandb:
        noda_trainer.train(
                        num_epochs=args.num_epochs, 
                        wandb=wandb, 
                        save_path=log_dirs)
    else:
        noda_trainer.train(
                        num_epochs=args.num_epochs, 
                        tensorboard_writer=tensorboard_writer,
                        save_path=log_dirs)
        # Ensures all logs are written
        tensorboard_writer.flush()   
        tensorboard_writer.close()

    mse_rollout, preds, truth = evaluate_multistep_rollout(
        model,
        data, 
        horizon=20,
        dt=args.dt,
        device=args.device,
        num_rollouts=50
    )

    print(f"Multi-step rollout MSE (20 steps): {mse_rollout:.6f}")

    # Encode state
    # q, p, u = autoencoder.encode(s_t)
     
    # actions           (12,)
    # observations      (58,)
    # next_observations (58,)
    # terminals         (1,)
    # rewards           (1,)

if __name__ == '__main__':
    train_dynamics_model()
