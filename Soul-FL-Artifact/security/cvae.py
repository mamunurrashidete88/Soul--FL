
import torch
from torch import nn
from torch.nn import functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim=128, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, anchor, sample=True):
        h = F.relu(self.encoder(torch.cat([x, anchor], 1)))
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        recon = self.output(F.relu(self.decoder(torch.cat([z, anchor], 1))))
        return recon, mu, logvar

    def loss(self, x, anchor, kl=0.1):
        recon, mu, logvar = self.forward(x, anchor)
        err = (recon - x).square().sum(dim=1).mean()
        kld = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
        return err + kl * kld

    @torch.no_grad()
    def score(self, x, anchor):
        self.eval()
        recon, _, _ = self.forward(x, anchor, sample=False)
        return (recon - x).square().sum(dim=1)
