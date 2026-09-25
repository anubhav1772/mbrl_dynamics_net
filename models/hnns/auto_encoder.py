import torch
import torch.nn as nn

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

# Initialize the AutoEncoder
input_dim = 76      # State dime
latent_dim = 2*12   # 2*K canonical states (q, p), K is DoF
autoencoder = AutoEncoder(input_dim, latent_dim)

