import torch
import torch.nn as nn

class HamiltonianODE(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(HamiltonianODE, self).__init__()
        self.latent_dim = latent_dim
        assert latent_dim % 2 == 0, "latent_dim must be even"
        self.K = latent_dim // 2

        # Learn Hamiltonian: H(q, p) → scalar
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
        H = self.H(H_in).sum()  # sum over batch for autograd

        # Compute gradients
        grad_q, grad_p = torch.autograd.grad(H, (q, p), create_graph=True)

        # Compute Q_k(a)
        Q = self.force(a)

        # Final derivatives
        dq_dt = grad_p
        dp_dt = -grad_q + Q
        du_dt = torch.cat([dq_dt, dp_dt], dim=-1)
        return du_dt