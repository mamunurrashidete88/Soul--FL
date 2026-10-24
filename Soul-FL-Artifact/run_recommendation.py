"""MovieLens-1M transfer: paired key sweep with local training.

"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from core.server import Server
from security.blockchain_sim import VoucherLedger
from security.zk import anchor_commitment

SEEDS = (11, 23, 37, 41, 53)

def read_movielens(root):
    root = Path(root)
    ratings = defaultdict(list)
    with (root / 'ratings.dat').open(encoding='latin-1') as f:
        for line in f:
            user, movie, rating, timestamp = map(int, line.strip().split('::'))
            ratings[user].append((timestamp, movie, rating))
    genres = {}
    with (root / 'movies.dat').open(encoding='latin-1') as f:
        for line in f:
            movie, _, genre = line.strip().split('::', 2)
            genres[int(movie)] = genre.split('|')
    movies = sorted(genres)
    all_genres = sorted({x for v in genres.values() for x in v})
    lookup = {movie: i for i, movie in enumerate(movies)}
    return ratings, genres, all_genres, lookup

def select_users(ratings, seed):
    valid = [u for u, records in ratings.items() if int(0.7 * len(records)) >= 64]
    rng = np.random.default_rng(seed)
    order = rng.permutation(sorted(valid))
    if len(order) < 61:
        raise ValueError('Need at least 61 users with 64 training ratings')
    return order[:60].tolist(), int(order[60])

def split_user(records):
    ordered = sorted(records, key=lambda v: (v[0], v[1]))
    a, b = int(0.7 * len(ordered)), int(0.8 * len(ordered))
    return [ordered[:a], ordered[a:b], ordered[b:]]

def positive(records, lookup):
    return [lookup[m] for _, m, r in records if r >= 4 and m in lookup]

def genre_anchor(training_records, genres, categories, rng):
    counts = [len(training_records)] + [sum(g in genres[m] for _, m, _ in training_records) for g in categories]
    return np.asarray(counts, dtype=np.float32) + rng.normal(0, 1, len(counts))

class SignedSketch:
    def __init__(self, raw_dim, buckets=1024, repetitions=3, seed=11):
        rng = np.random.default_rng(seed)
        self.locations = rng.integers(0, buckets, size=(repetitions, raw_dim), dtype=np.int32)
        self.signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(repetitions, raw_dim))
        self.buckets = buckets
        self.scale = repetitions ** -0.5

    def __call__(self, update):
        values = update.detach().cpu().numpy().reshape(-1)
        result = np.zeros(self.buckets, dtype=np.float32)
        for loc, sign in zip(self.locations, self.signs):
            result += np.bincount(loc, weights=values * sign, minlength=self.buckets).astype(np.float32)
        return result * self.scale

class ItemModel(torch.nn.Module):
    def __init__(self, items, width=64):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(items, width) * 0.01)

class RecommenderClient:
    def __init__(self, positives, item_count, seed, observed=None):
        self.positives = set(positives)
        self.observed = set(observed if observed is not None else positives)
        self.item_count = item_count
        generator = torch.Generator().manual_seed(seed)
        self.user = torch.randn(64, generator=generator) * 0.01

    def benign_update(self, global_items, seed, epochs=2):
        rng = np.random.default_rng(seed)
        if not self.positives:
            return torch.zeros_like(global_items)
        items = torch.nn.Parameter(global_items.detach().clone())
        user = torch.nn.Parameter(self.user.detach().clone())
        optimizer = torch.optim.Adam([items, user], lr=5e-3)
        pairs = [(item, self._negative(rng)) for item in sorted(self.positives) for _ in range(4)]
        for _ in range(epochs):
            for group in np.array_split(rng.permutation(len(pairs)), max(1, int(np.ceil(len(pairs) / 128)))):
                pos = torch.tensor([pairs[i][0] for i in group], dtype=torch.long)
                neg = torch.tensor([pairs[i][1] for i in group], dtype=torch.long)
                optimizer.zero_grad()
                diff = (items[pos] * user).sum(1) - (items[neg] * user).sum(1)
                torch.nn.functional.softplus(-diff).mean().backward()
                optimizer.step()
        self.user = user.detach().clone()
        return (global_items - items.detach())

    def _negative(self, rng):
        while True:
            item = int(rng.integers(self.item_count))
            if item not in self.observed:
                return item

def optimized_fake_update(global_items, target, norm_cap, seed):
    
    rng = np.random.default_rng(seed)
    items = torch.nn.Parameter(global_items.detach().clone())
    user = torch.nn.Parameter(global_items[target].detach().clone())
    optimizer = torch.optim.Adam([items, user], lr=5e-3)
    for _ in range(20):
        neg_ids = torch.tensor(rng.integers(len(items), size=16), dtype=torch.long)
        optimizer.zero_grad()
        margin = items[target] @ user - (items[neg_ids] * user).sum(1)
        torch.nn.functional.softplus(-margin).mean().backward()
        optimizer.step()
    update = global_items - items.detach()
    size = float(update.norm())
    return update * min(1., norm_cap / max(size, 1e-12))

def evaluate(model, users, split, lookup, target):
    with torch.no_grad():
        item_vectors = model.weight.detach().cpu()
        ranking, hits, exposed = [], [], []
        for user_id, client in users.items():
            train, validation, test = split[user_id]
            blocked = {lookup[m] for _, m, _ in train + validation if m in lookup}
            relevant = set(positive(test, lookup)) - blocked
            if relevant:
                scores = item_vectors @ client.user.cpu()
                scores[list(blocked)] = -float('inf')
                top = torch.topk(scores, min(10, len(scores) - len(blocked))).indices.tolist()
                gains = sum(1 / np.log2(rank + 2) for rank, item in enumerate(top) if item in relevant)
                ideal = sum(1 / np.log2(rank + 2) for rank in range(min(len(relevant), 10)))
                ranking.append(gains / ideal)
                hits.append(float(bool(relevant.intersection(top))))
            if target not in {lookup[m] for _, m, _ in train + validation + test if m in lookup}:
                scores = item_vectors @ client.user.cpu()
                scores[list(blocked)] = -float('inf')
                exposed.append(float(target in torch.topk(scores, 10).indices.tolist()))
        return float(np.mean(ranking)), float(np.mean(hits)), float(np.mean(exposed))

def run(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    ratings, genres, categories, lookup = read_movielens(args.data)
    honest_ids, source_id = select_users(ratings, args.seed)
    splits = {u: split_user(ratings[u]) for u in honest_ids + [source_id]}
    source_records = splits[source_id][0]
    source_observed = {movie for _, movie, _ in source_records}
    candidates = sorted(movie for movie, idx in lookup.items() if movie not in source_observed and
                        sum(movie in {m for _, m, r in splits[u][0] if r >= 4} for u in honest_ids) >= 10)
    if not candidates:
        raise ValueError('No target meets the stated positive-interaction threshold')
    target = lookup[int(rng.choice(candidates))]
    model = ItemModel(len(lookup))
    sketch = SignedSketch(model.weight.numel(), seed=args.seed)
    server = Server(model, 19, components=32, snapshots_per_submission=0)
    ledger = VoucherLedger()
    honest = {f'h-{u}': RecommenderClient(positive(splits[u][0], lookup), len(lookup), args.seed + int(u),
               [lookup[m] for _, m, _ in splits[u][0] if m in lookup]) for u in honest_ids}
    source = positive(source_records, lookup)
    attackers = {f'a-{i}': RecommenderClient(source, len(lookup), args.seed + 100000 + i,
                  [lookup[m] for _, m, _ in source_records if m in lookup]) for i in range(args.keys)}
    all_clients = {**honest, **attackers}
    for key in all_clients:
        own_records = splits[int(key[2:])][0] if key.startswith('h-') else source_records
        anchor = genre_anchor(own_records, genres, categories, rng)
        commitment = anchor_commitment(key, anchor)
        # Recommendation enrollment.
        ledger.enroll_assumed(key, commitment)
        server.register_anchor(key, anchor)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    benign_norms = []
    absence = {}
    if args.intermit:
        selected_honest = np.random.default_rng(args.seed + 40000).choice(sorted(honest), 12, replace=False)
        absence = {key: int(np.random.default_rng(args.seed + 50000 + j).integers(51, 102))
                   for j, key in enumerate(selected_honest)}
    with output.open('w', newline='') as f:
        columns = ['seed','round','keys','method','attack','ndcg10','hr10','target10','attacker_weight','honest_eligible',
                   'attacker_submitted','attacker_accepted','attacker_vouchers']
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for t in range(1, args.rounds + 1):
            
            universe = sorted(honest) + [f'a-{i}' for i in range(40)]
            priority = np.random.default_rng(args.seed * 100000 + t).permutation(universe)
            available = lambda key: key not in absence or not (absence[key] <= t < absence[key] + 60)
            qualifying = lambda key: (args.method != 'soul' or ledger.is_eligible(key))
            selected = [k for k in priority if k in all_clients and available(k) and qualifying(k)][:10]
            access = np.mean([qualifying(k) for k in honest])
            server.start_round(t)
            admitted = []
            for key in selected:
                client = all_clients[key]
                if key in attackers and t >= 51 and args.attack == 'injection':
                    client.positives.add(target)
                    client.observed.add(target)
                if key in attackers and t >= 51 and args.attack == 'optimized':
                    cap = float(np.quantile(benign_norms, 0.95)) if benign_norms else 0.0
                    update = optimized_fake_update(model.weight.detach(), target, cap, args.seed * 100000 + t * 100 + int(key[2:]))
                else:
                    update = client.benign_update(model.weight.detach(), args.seed * 100000 + t * 100 + len(key))
                if key in attackers and t <= 50:
                    benign_norms.append(float(update.norm()))
                if key in attackers and t >= 51 and benign_norms:
                    cap = float(np.quantile(benign_norms, 0.95))
                    update = update * min(1., cap / max(float(update.norm()), 1e-12))
                projection = sketch(update)
                screened, _ = server.screen(key, projection, [], t)
                accepted = args.method == 'fedavg' or screened
                if accepted:
                    admitted.append((key, {'weight': update}, ledger.balance(key) if args.method == 'soul' else ledger.initial))
            weights = server.aggregate([row[1] for row in admitted], [row[2] for row in admitted])
            attacker_weight = sum(w for w, row in zip(weights, admitted) if row[0] in attackers)
            if args.method == 'soul':
                for key, _, _ in admitted:
                    voucher, signature = ledger.voucher(key)
                    ledger.apply(voucher, signature)
            server.finish_round(t)
            ndcg = hr = exposure = ''
            if t % 5 == 0:
                ndcg, hr, exposure = evaluate(model, honest, splits, lookup, target)
            writer.writerow({'seed': args.seed,'round':t,'keys':args.keys,'method':args.method,'attack':args.attack,
                'ndcg10':ndcg,'hr10':hr,'target10':exposure,'attacker_weight':attacker_weight,'honest_eligible':access,
                'attacker_submitted':sum(k in attackers for k in selected),
                'attacker_accepted':sum(row[0] in attackers for row in admitted),
                'attacker_vouchers':sum(row[0] in attackers for row in admitted) if args.method == 'soul' else 0})
            f.flush()
            ledger.advance()
            print(f'round={t} admitted={len(admitted)} attacker_weight={attacker_weight:.3f}')
    print(f'Wrote fresh outcomes to {output}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='MovieLens-1M directory containing ratings.dat and movies.dat')
    p.add_argument('--seed', type=int, choices=SEEDS, default=11)
    p.add_argument('--keys', type=int, choices=[1,5,10,20,40], default=40)
    p.add_argument('--attack', choices=['injection','optimized'], default='injection')
    p.add_argument('--method', choices=['fedavg','screen-fixed','soul'], default='soul')
    p.add_argument('--intermit', action='store_true', help='20 percent of honest users absent for 60 rounds')
    p.add_argument('--rounds', type=int, default=200)
    p.add_argument('--output', default='experiments/logs/recommendation.csv')
    run(p.parse_args())
