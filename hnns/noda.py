import torch
import torch.nn as nn
from torchdiffeq import odeint

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):  
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
    def __init__(self, latent_dim, action_dim):
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
        u, a = torch.split(u_a, [self.latent_dim, u_a.shape[-1] - self.latent_dim], dim=-1)
        q, p = torch.chunk(u, 2, dim=-1)
        q.requires_grad_(True)
        p.requires_grad_(True)

        # Compute Hamiltonian
        H_in = torch.cat([q, p], dim=-1)

        du_list = []
        for i in range(q.shape[0]):
            H_scalar = self.H(H_in[i:i+1]).squeeze()
            grads = torch.autograd.grad(H_scalar, (q[i], p[i]), retain_graph=True, create_graph=True)
            dq_dt_i = grads[1]
            dp_dt_i = -grads[0] + self.force(a[i:i+1]).squeeze(0)
            du_i = torch.cat([dq_dt_i, dp_dt_i], dim=-1)
            du_list.append(du_i)

        du_dt = torch.stack(du_list, dim=0)
        return du_dt

# Initialize the AutoEncoder
input_dim = 76      # State dim
latent_dim = 2*12   # 2*K canonical states (q, p), K is DoF
action_dim = 12     
autoencoder = AutoEncoder(input_dim, latent_dim)

# Encode state
q, p, u = autoencoder.encode(s_t)
 
# Aliengo Offline Data
data_load_path = 'mbrl_dynamics_net/dataset/PreprocessedDataset/train'
data = OfflineDatasetLoader().get_dataset(data_load_path, preprocess=True)
for key, value in data.items():
    print(f"{key}: {np.array(value).shape}")

# actions           (12,)
# observations      (58,)
# next_observations (58,)
# terminals         (1,)
# rewards           (1,)

# - `u` is latent state [q, p] of shape (batch_size, latent_dim)
# - `a` is action of shape (batch_size, action_dim)
# - `ode_func` is an instance of your HamiltonianODE or ODENetwork

ode_func = HamiltonianODE(latent_dim, action_dim).to(u.device)

# Concatenate u and a for input to ode_func
u_a = torch.cat([u, a], dim=-1)

# Choose time span
t_span = torch.tensor([0, 1], dtype=torch.float32).to(u.device)  # From t=0 to t=1

# Integrate using odeint
# Output shape: [2, batch_size, latent_dim] — one for t=0 and one for t=1
u_a_traj = odeint(ode_func, u_a, t_span, method='rk4')  # Or use 'dopri5'

# Extract final state (t = 1)
u_a_next = u_a_traj[-1]  # Shape: (batch_size, latent_dim + action_dim)

# Separate latent state
u_next = u_a_next[:, :u.shape[1]]  # Extract u (latent state) part (drop action)

# Decode to predict next state
s_t_plus1_pred = autoencoder.decode(u_next)

