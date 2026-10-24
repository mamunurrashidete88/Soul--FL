
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader

def head_delta(local_model, global_model):
    return torch.cat([(before.detach().cpu().float() - after.detach().cpu().float()).flatten()
                      for before, after in zip(global_model.head_parameters(), local_model.head_parameters())]).numpy()

class Client:
    def __init__(self, key, loader, model_factory, device='cpu', attacker=False, pivot=51,
                 source=2, target=8, epochs=5, lr=0.01):
        self.key, self.loader, self.factory, self.device = key, loader, model_factory, device
        self.attacker, self.pivot, self.source, self.target = attacker, pivot, source, target
        self.epochs, self.lr = epochs, lr

    def train(self, global_model, round_number, calibration=False):
        local = self.factory().to(self.device)
        local.load_state_dict(global_model.state_dict())
        local.train()
        opt = torch.optim.SGD(local.parameters(), lr=self.lr)
        total_steps = len(self.loader) * self.epochs

        if calibration and total_steps < 12:
            raise ValueError('Calibration needs >=12 local optimizer steps for 11 distinct snapshots')
        checkpoints = set(round(v * total_steps / 12) for v in range(1, 12)) if calibration else set()
        snapshots = []
        step = 0
        for _ in range(self.epochs):
            for images, labels in self.loader:
                images, labels = images.to(self.device), labels.to(self.device)
                if self.attacker and round_number >= self.pivot:
                    labels = labels.clone()
                    labels[labels == self.source] = self.target
                opt.zero_grad()
                loss = nn.functional.cross_entropy(local(images), labels)
                loss.backward()
                opt.step()
                step += 1
                if step in checkpoints:
                    snapshots.append(head_delta(local, global_model))
        if calibration and len(snapshots) != 11:
            raise RuntimeError('Calibration snapshots were not distinct')
        original = global_model.state_dict()
        trained = local.state_dict()
        delta = {name: (original[name].detach().cpu() - trained[name].detach().cpu())
                 for name in original if original[name].is_floating_point()}
        return delta, head_delta(local, global_model), snapshots
