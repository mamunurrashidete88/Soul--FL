
import argparse
import csv
import json
import statistics
from pathlib import Path

WINDOWS = ((51, 60), (61, 100), (101, 200))

def summarize(path, start, end):
    with Path(path).open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if start <= int(row['round']) <= end]
    if len(rows) != end - start + 1:
        raise ValueError(f'{path}: expected every round {start}--{end}, found {len(rows)}')
    def total(column):
        return sum(float(row[column]) for row in rows)
    submitted = total('attacker_submissions')
    eligible = total('attacker_eligible')
    if not submitted or not eligible:
        raise ValueError(f'{path}: zero denominator for acceptance or voucher frequency')
    return {
        'accept_pct': 100 * total('attacker_accepted') / submitted,
        'voucher_pct': 100 * total('attacker_vouchers') / eligible,
        'attacker_weight_pct': 100 * total('attacker_weight') / len(rows),
        'honest_access_pct': 100 * total('honest_eligible') / len(rows),
    }

def main(paths):
    if len(paths) < 2:
        raise ValueError('Provide at least two independently generated run logs')
    results = {}
    for start, end in WINDOWS:
        per_seed = [summarize(path, start, end) for path in paths]
        results[f'{start}-{end}'] = {
            key: {'mean': statistics.mean(row[key] for row in per_seed),
                  'sample_sd': statistics.stdev(row[key] for row in per_seed)}
            for key in per_seed[0]
        }
    print(json.dumps({'runs': len(paths), 'windows': results}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+')
    main(parser.parse_args().logs)
