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
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import wandb

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
    def __init__(self, model, data, batch_size, lr, dt, alpha, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.dt = dt
        self.alpha = alpha
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

        # for name, param in self.model.named_parameters():
        #     print(name, param.data.mean().item(), param.grad is not None)

        for batch in self.dataloader:
            obss, actions, next_obss, rewards = batch

            obss, actions, next_obss, rewards = obss.to(self.device), actions.to(self.device), next_obss.to(self.device), rewards.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass: compute the total loss (state + reward prediction loss)
            total_loss, loss_recon, loss_state, loss_reward = self.model.compute_loss(obss, actions, next_obss, rewards, self.dt, self.alpha)
            
            # Backpropagation and optimization
            total_loss.backward()
            self.optimizer.step()

            total_loss_ += total_loss.item()
            total_recon_loss += loss_recon.item()
            total_state_loss += loss_state.item()
            total_reward_loss += loss_reward.item()
        
        mean_loss = total_loss_ / len(self.dataloader)
        mean_recon_loss = total_recon_loss / len(self.dataloader)
        mean_state_loss = total_state_loss / len(self.dataloader)
        mean_reward_loss = total_reward_loss / len(self.dataloader)

        return mean_loss, mean_recon_loss, mean_state_loss, mean_reward_loss

    def train(self, num_epochs=1000, wandb = None, tensorboard_writer = None):
        for epoch in range(num_epochs):
            mean_total_loss, mean_recon_loss, mean_state_loss, mean_reward_loss = self.train_one_epoch()

            # Log training progress
            if wandb is not None:
                wandb.log({'timestep': epoch,
                    'loss/dynamics_train_loss': mean_total_loss,
                })
            else:
                tensorboard_writer.add_scalar("loss/dynamics_train_loss", mean_total_loss, epoch)
            print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {mean_total_loss:.4f}, Recon Loss: {mean_recon_loss:.4f}, State Loss: {mean_state_loss:.4f}, Reward Loss: {mean_reward_loss:.4f}")

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
    parser.add_argument("--num_epochs", type=int, default=100)
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
        log_dirs = make_log_dirs(args.task, 'test/dynamics', args.seed, vars(args), run_name=args.run_name)
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
        noda_trainer.train(num_epochs=args.num_epochs, wandb=wandb)
    else:
        noda_trainer.train(num_epochs=args.num_epochs, tensorboard_writer=tensorboard_writer)
        tensorboard_writer.close()

    # Encode state
    # q, p, u = autoencoder.encode(s_t)
     
    # actions           (12,)
    # observations      (58,)
    # next_observations (58,)
    # terminals         (1,)
    # rewards           (1,)

if __name__ == '__main__':
    train_dynamics_model()
