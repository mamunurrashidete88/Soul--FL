from dataclasses import dataclass
import math

SEEDS = (11, 23, 37, 41, 53)

@dataclass(frozen=True)
class Settings:
    dataset: str = 'cifar10'
    clients: int = 100
    attackers: int = 40
    invitations: int = 10
    rounds: int = 200
    pivot: int = 51
    calibration_rounds: int = 10
    local_epochs: int = 5
    batch_size: int = 32
    client_lr: float = 0.01
    alpha: float = 0.5
    projected_dim: int = 1024
    cvae_hidden: int = 512
    cvae_latent: int = 128
    cvae_kl: float = 0.1
    cvae_lr: float = 0.001
    score_window: int = 5
    threshold_quantile: float = 0.95
    initial_balance: float = 100.0
    credit: float = 10.0
    gate: float = 2.0
    decay: float = 0.05
    seed: int = 11

    @property
    def decay_factor(self):
        return math.exp(-self.decay)

    @property
    def anchor_dim(self):
        return 192 if self.dataset == 'cifar10' else 64 if self.dataset == 'femnist' else 19

DEFAULT = Settings()
