"""Weighted model-update aggregation with weights frozen before credit."""
from security.trust_engine import aggregation_weights

def weighted_update(model, admitted, balances):
    """admitted contains (global - local) deltas for floating model state tensors."""
    if not admitted:
        return []
    weights = aggregation_weights(balances)
    import torch
    state = model.state_dict()
    with torch.no_grad():
        for key, value in state.items():
            if value.is_floating_point():
                delta = sum(weight * update[key].to(value.device) for weight, update in zip(weights, admitted))
                value.sub_(delta)
        model.load_state_dict(state)
    return weights
