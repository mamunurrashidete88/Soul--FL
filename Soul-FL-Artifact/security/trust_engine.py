SCALE = 1_000_000
DECAY_NUM = 951_229  # floor(exp(-0.05) * SCALE)

def decay_units(balance: int, elapsed: int, factor: int = DECAY_NUM) -> int:
    if balance < 0 or elapsed < 0:
        raise ValueError('Balance and elapsed rounds must be nonnegative')
    for _ in range(elapsed):
        balance = balance * factor // SCALE
    return balance

def effective_balance(balance: int, last_round: int, current_round: int) -> int:
    if current_round < last_round:
        raise ValueError('Round cannot move backwards')
    return decay_units(balance, current_round - last_round)

def eligible(balance_units: int, gate: float = 2.0) -> bool:
    return balance_units > round(gate * SCALE)

def aggregation_weights(balances):
    import math
    raw = [math.log1p(b / SCALE) for b in balances]
    denominator = sum(raw)
    if denominator <= 0:
        raise ValueError('Admitted balances must be positive')
    return [v / denominator for v in raw]
