
import argparse
import csv
import random
from pathlib import Path
import numpy as np
import torch
from config import Settings
from core.models import get_model
from core.client import Client
from core.server import Server
from data_loader import load_images, dirichlet_partition, make_loader
from security.zk import image_histogram, perturb, verify_image_relation, anchor_commitment
from security.blockchain_sim import VoucherLedger
from security.trust_engine import aggregation_weights

def run(args):
    cfg = Settings(dataset=args.dataset, seed=args.seed, clients=args.clients,
                   attackers=args.attackers, rounds=args.rounds, projected_dim=args.components,
                   invitations=args.invitations)
    if cfg.dataset not in ('cifar10', 'femnist'):
        raise ValueError('Choose cifar10 or femnist')
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data = load_images(cfg.dataset, args.data)
    train_x, train_y = data['train']
    test_x, test_y = data['test']
    # Three batches per local epoch ensure distinct calibration snapshots.
    if args.reused_data_keys:
        pieces = dirichlet_partition(train_y, 61, cfg.alpha, min_size=96, seed=cfg.seed)
        cohort = [(f'h-{i}', pieces[i], False) for i in range(60)] + [
            (f'a-{i}', pieces[60], True) for i in range(args.reused_data_keys)]
    else:
        pieces = dirichlet_partition(train_y, cfg.clients, cfg.alpha, min_size=96, seed=cfg.seed)
        cohort = [(f'key-{i}', indices, i < cfg.attackers) for i, indices in enumerate(pieces)]
    model_factory = lambda: get_model(cfg.dataset)
    server = Server(model_factory(), cfg.anchor_dim, args.device, cfg.projected_dim,
                    cfg.cvae_hidden, cfg.cvae_latent, cfg.cvae_kl, cfg.cvae_lr)
    ledger = VoucherLedger(initial=cfg.initial_balance, credit=cfg.credit, gate=cfg.gate)
    clients = {}
    for key, indices, is_attacker in cohort:
        raw = train_x[indices]
        anchor, noise = perturb(image_histogram(raw), rng)
        commitment = anchor_commitment(key, anchor)
        checked = verify_image_relation(raw, anchor, noise, key, commitment)
        ledger.enroll(key, commitment, relation_checked=checked)
        server.register_anchor(key, anchor)
        clients[key] = Client(key, make_loader(train_x, train_y, indices, cfg.batch_size),
                              model_factory, args.device, attacker=is_attacker,
                              pivot=cfg.pivot, epochs=cfg.local_epochs, lr=cfg.client_lr)
    test_loader = make_loader(test_x, test_y, np.arange(len(test_y)), 512, False)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['seed','round','keys','run_mode','accuracy','asr','attacker_eligible',
            'attacker_invitations','attacker_submissions','attacker_accepted','attacker_vouchers',
            'attacker_weight','honest_eligible'])
        writer.writeheader()
        for t in range(1, cfg.rounds + 1):
            available = [key for key in clients if ledger.is_eligible(key)]
            attacker_eligible_count = sum(ledger.is_eligible(key) for key in clients if clients[key].attacker)
            if args.reused_data_keys:
                
                universe = [f'h-{j}' for j in range(60)] + [f'a-{j}' for j in range(100)]
                priority = np.random.default_rng(cfg.seed * 100000 + t).permutation(universe)
                eligible_set = set(available)
                selected = [key for key in priority if key in eligible_set][:cfg.invitations]
            else:
                selected = rng.choice(available, min(cfg.invitations, len(available)), replace=False).tolist() if available else []
            honest = [key for key in clients if not clients[key].attacker]
            honest_access = sum(ledger.is_eligible(key) for key in honest) / len(honest)
            server.start_round(t)
            admitted = []
            attacker_accepted = 0
            for key in selected:
                delta, head, snapshots = clients[key].train(server.model, t, t <= cfg.calibration_rounds)
                passed, score = server.screen(key, head, snapshots, t)
                if passed:
                    admitted.append((key, delta, ledger.balance(key)))
                    attacker_accepted += int(clients[key].attacker)
            balances = [entry[2] for entry in admitted]
            weights = server.aggregate([entry[1] for entry in admitted], balances)
            operator_weight = sum(w for w, entry in zip(weights, admitted) if clients[entry[0]].attacker)
            applied = 0
            for key, _, _ in admitted:
                voucher, signature = ledger.voucher(key)
                ledger.apply(voucher, signature)
                applied += int(clients[key].attacker)
            server.finish_round(t)
            accuracy, asr = server.evaluate(test_loader)
            writer.writerow({'seed': cfg.seed, 'round': t,
                'keys': args.reused_data_keys if args.reused_data_keys else cfg.attackers,
                'run_mode': 'reused-data' if args.reused_data_keys else 'fixed-cohort',
                'accuracy': accuracy, 'asr': asr,
                'attacker_eligible': attacker_eligible_count,
                'attacker_invitations': sum(clients[key].attacker for key in selected),
                'attacker_submissions': sum(clients[key].attacker for key in selected),
                'attacker_accepted': attacker_accepted, 'attacker_vouchers': applied,
                'attacker_weight': operator_weight, 'honest_eligible': honest_access})
            handle.flush()
            ledger.advance()
            print(f'round={t} accuracy={accuracy:.2f} asr={asr:.2f} attacker_weight={operator_weight:.3f}')
    print(f'Run log written to {path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cifar10','femnist'], default='cifar10')
    parser.add_argument('--data', default='data', help='CIFAR cache or LEAF FEMNIST directory')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--clients', type=int, default=100)
    parser.add_argument('--attackers', type=int, default=40)
    parser.add_argument('--reused-data-keys', type=int, choices=[1,10,40,100],
                        help='Copy one qualifying attacker dataset across K separately enrolled keys with 60 honest clients')
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--invitations', type=int, default=10)
    parser.add_argument('--components', type=int, default=1024)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='experiments/logs/image_run.csv')
    run(parser.parse_args())
