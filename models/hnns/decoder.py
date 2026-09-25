class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        # Decoder: Map the evolved canonical states back to the original state space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, u):
        # Decode the canonical states to the next state
        s_next = self.decoder(u)
        return s_next
