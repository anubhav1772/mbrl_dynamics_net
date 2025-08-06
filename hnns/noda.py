import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchdiffeq import odeint
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
# Add the parent folder of `mbrl_dynamics_net` to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import Dict, List, Union, Tuple, Optional, Callable
from mbrl_dynamics_net.utils.buffer import OfflineDatasetLoader

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

class HamiltonianODE(nn.Module):
    def __init__(self, latent_dim, action_dim) -> None:
        super(HamiltonianODE, self).__init__()
        self.latent_dim = latent_dim
        assert latent_dim % 2 == 0, "latent_dim must be even"
        self.K = latent_dim // 2

        # Learn Hamiltonian: H(q, p) -> scalar
        self.H = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)  # scalar output
        )

        # External force model Q(a)
        self.force = nn.Sequential(
            nn.Linear(action_dim, self.K),
            nn.Tanh(),
            nn.Linear(self.K, self.K)
        )

    def forward(self, t, u_a):
        # u_a = concat([u, a]) where u = [q, p]
        u, a = torch.split(u_a, [self.latent_dim, u_a.shape[-1] - self.latent_dim], dim=-1)  # u: (64, 24), a: (64, 12)
        q, p = torch.chunk(u, 2, dim=-1)  # q: (64, 12), p: (64, 12)

        # Ensure q and p require gradients
        q.requires_grad_(True)
        p.requires_grad_(True)

        # Compute Hamiltonian input
        H_in = torch.cat([q, p], dim=-1)  # H_in: (64, 24)

        # Compute the Hamiltonian for the full batch
        H_scalar = self.H(H_in).squeeze()  # H_scalar: (64,)
        # Sum or average over the batch to make it a scalar (choose one)
        H_scalar = H_scalar.mean()
        # Compute gradients with respect to q and p
        grads = torch.autograd.grad(H_scalar, (q, p), retain_graph=True, create_graph=True)

        # Extract gradients for each sample in the batch
        dq_dt = grads[1]  # dq_dt: (64, 12)
        dp_dt = -grads[0] + self.force(a)  # dp_dt: (64, 12)

        # Combine dq_dt and dp_dt to form the derivative of the state (du_dt)
        du_dt = torch.cat([dq_dt, dp_dt], dim=-1)  # du_dt: (64, 24)
        return torch.cat([du_dt, a], dim=-1)       # (64, 36)

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
        self.ode_func = HamiltonianODE(latent_dim, action_dim).to(device)
        self.reward_decoder = RewardDecoder(latent_dim, action_dim).to(device)
        # self.latent_dim = latent_dim
        # self.action_dim = action_dim

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"].reshape(-1, 1)
        return obss, actions, next_obss, rewards

    def predict_state_reward(self, s_t, a_t, t_span=[0, 1]):
        '''Predict next state and reward given current state and action.
        '''
        q, p, u = self.autoencoder.encode(s_t)

        # Predict reward r using q,p and action a
        r_pred = self.reward_decoder(q, p, a_t)

        u_a = torch.cat([u, a_t], dim=-1)
        t_span = torch.tensor(t_span, dtype=torch.float32).to(self.device)
        u_a_traj = odeint(self.ode_func, u_a, t_span, method='rk4')

        # Extract final state (t = 1)
        u_a_next = u_a_traj[-1]             # Shape: (batch_size, latent_dim + action_dim)

        # Separate latent state
        u_next = u_a_next[:, :u.shape[1]]  # Extract u (latent state) part (drop action) 

        # u_next = u_a_traj[-1][:, :self.latent_dim] 

        s_t_plus1_pred = self.autoencoder.decode(u_next)
        
        return s_t_plus1_pred, r_pred

    def compute_loss(self, s_t, a_t, s_tp1_true, r_true, alpha=0.5):
        '''One-step prediction loss (MSE for state + reward)
        '''
        s_pred, r_pred = self.predict_state_reward(s_t, a_t)
        # Canonical latent encoding
        _, _, u = self.autoencoder.encode(s_t)
        # Reconstruction from latent canonical encoding
        s_recon = self.autoencoder.decode(u)

        # Reconstruction loss
        loss_recon = F.mse_loss(s_recon, s_t)          
        # Next-state prediction loss
        loss_state = F.mse_loss(s_pred, s_tp1_true)
        # Reward prediction loss
        loss_reward = F.mse_loss(r_pred, r_true)

        # Total loss function for NODA
        # As a convex combination of the state loss and the reward loss
        total_loss = alpha * (loss_recon + loss_state) + (1 - alpha) * loss_reward
        total_loss = total_loss.clone().detach().requires_grad_(True)
        return total_loss, loss_recon.item(), loss_state.item(), loss_reward.item()

class NODATrainer:
    def __init__(self, model, data, batch_size=64, lr=1e-4, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Prepare data for batching (convert numpy arrays to torch tensors)
        obss, actions, next_obss, rewards = self.model.format_samples_for_training(data)
        
        # Create a DataLoader for batching
        self.dataset =  TensorDataset(torch.tensor(obss, dtype=torch.float32),
                                        torch.tensor(actions, dtype=torch.float32),
                                        torch.tensor(next_obss, dtype=torch.float32),
                                        torch.tensor(rewards, dtype=torch.float32))
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def train_one_epoch(self):
        total_loss_ = 0
        total_recon_loss = 0
        total_state_loss = 0
        total_reward_loss = 0

        for batch in self.dataloader:
            obss, actions, next_obss, rewards = batch

            obss, actions, next_obss, rewards = obss.to(self.device), actions.to(self.device), next_obss.to(self.device), rewards.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass: compute the total loss (state + reward prediction loss)
            total_loss, loss_recon, loss_state, loss_reward = self.model.compute_loss(obss, actions, next_obss, rewards)
            
            # Backpropagation and optimization
            total_loss.backward()
            self.optimizer.step()

            total_loss_ += total_loss.item()
            total_recon_loss += loss_recon
            total_state_loss += loss_state
            total_reward_loss += loss_reward
        
        mean_loss = total_loss_ / len(self.dataloader)
        mean_recon_loss = total_recon_loss / len(self.dataloader)
        mean_state_loss = total_state_loss / len(self.dataloader)
        mean_reward_loss = total_reward_loss / len(self.dataloader)

        return mean_loss, mean_recon_loss, mean_state_loss, mean_reward_loss

    def train(self, num_epochs=1000):
        for epoch in range(num_epochs):
            mean_loss, mean_recon_loss, mean_state_loss, mean_reward_loss = self.train_one_epoch()

            # Log training progress (could be to TensorBoard or standard print)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {mean_loss:.4f}, Recon Loss: {mean_recon_loss:.4f}, State Loss: {mean_state_loss:.4f}, Reward Loss: {mean_reward_loss:.4f}")

# Initialize the AutoEncoder
input_dim = 58      # State dim
latent_dim = 2*12   # 2*K canonical states (q, p), K is DoF
action_dim = 12     
device = "cuda" if torch.cuda.is_available() else "cpu"
# autoencoder = AutoEncoder(input_dim, latent_dim)

# Aliengo Offline Data
data_load_path = 'mbrl_dynamics_net/dataset/PreprocessedDataset/train'
data = OfflineDatasetLoader().get_dataset(data_load_path, preprocess=True)
for key, value in data.items():
    print(f"{key}: {np.array(value).shape}")

model = NODA(input_dim, latent_dim, action_dim, device=device)
trainer = NODATrainer(model, data, batch_size=64, lr=2e-4, device=device)
trainer.train(num_epochs=10)

# Encode state
# q, p, u = autoencoder.encode(s_t)
 
# actions           (12,)
# observations      (58,)
# next_observations (58,)
# terminals         (1,)
# rewards           (1,)

# - `u` is latent state [q, p] of shape (batch_size, latent_dim)
# - `a` is action of shape (batch_size, action_dim)
# - `ode_func` is an instance of your HamiltonianODE or ODENetwork

# ode_func = HamiltonianODE(latent_dim, action_dim).to(u.device)

# # Concatenate u and a for input to ode_func
# u_a = torch.cat([u, a], dim=-1)

# # Choose time span
# t_span = torch.tensor([0, 1], dtype=torch.float32).to(u.device)  # From t=0 to t=1

# # Integrate using odeint
# # Output shape: [2, batch_size, latent_dim] — one for t=0 and one for t=1
# u_a_traj = odeint(ode_func, u_a, t_span, method='rk4')  # Or use 'dopri5'

# # Decode to predict next state
# s_t_plus1_pred = autoencoder.decode(u_next)

