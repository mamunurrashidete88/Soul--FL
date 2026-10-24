import json
from pathlib import Path
import numpy as np

def dirichlet_partition(labels, clients=100, alpha=0.5, min_size=64, seed=11, trials=500):
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        pieces = [[] for _ in range(clients)]
        for label in np.unique(labels):
            indices = rng.permutation(np.flatnonzero(labels == label))
            sizes = rng.multinomial(len(indices), rng.dirichlet(np.full(clients, alpha)))
            for i, split in enumerate(np.split(indices, np.cumsum(sizes)[:-1])):
                pieces[i].extend(split.tolist())
        if min(map(len, pieces)) >= min_size:
            return [rng.permutation(part).tolist() for part in pieces]
    raise RuntimeError('Could not satisfy minimum size; change seed, alpha, or population')

def load_femnist_leaf(json_dir):

    source = Path(json_dir)
    rows = []
    for file in sorted(source.rglob('*.json')):
        if file.parent.name not in ('train', 'test'):
            continue
        block = json.loads(file.read_text())
        for user in block.get('users', []):
            data = block['user_data'][user]
            for image, label in zip(data['x'], data['y']):
                try:
                    digit = int(label)
                except (ValueError, TypeError):
                    continue
                if 0 <= digit <= 9:
                    raw = np.asarray(image).reshape(28, 28)
                    if raw.max() <= 1:
                        raw = raw * 255
                    rows.append((file.parent.name, np.rint(raw).clip(0, 255).astype(np.uint8), digit))
    if not rows:
        raise ValueError('No LEAF FEMNIST digits found in train/test JSON folders')
    result = {}
    for split in ('train', 'test'):
        matching = [(x, y) for s, x, y in rows if s == split]
        result[split] = (np.stack([x for x, _ in matching])[:, None],
                         np.asarray([y for _, y in matching], dtype=np.int64))
    return result

def load_images(name, root, seed=11):
    if name == 'femnist':
        return load_femnist_leaf(root)
    if name != 'cifar10':
        raise ValueError('Only cifar10 and LEAF femnist are supported')
    from torchvision.datasets import CIFAR10
    train = CIFAR10(root, train=True, download=True)
    test = CIFAR10(root, train=False, download=True)
    return {'train': (np.asarray(train.data).transpose(0, 3, 1, 2), np.asarray(train.targets)),
            'test': (np.asarray(test.data).transpose(0, 3, 1, 2), np.asarray(test.targets))}

def make_loader(raw, labels, indices, batch=32, shuffle=True):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    x = torch.tensor(raw[indices].astype(np.float32) / 255)
    y = torch.tensor(labels[indices], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=shuffle)
