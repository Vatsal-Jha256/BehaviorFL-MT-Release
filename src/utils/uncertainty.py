"""Normalized inverse-loss weighting (NILW).

The rule is w_i = (1 / L_i) / sum_j(1 / L_j). This differs from the
temperature-softmax UW-SO weighting rule.
"""

from typing import Sequence

import torch


# NILW

def compute_nilw_weights(
    task_losses: Sequence,
) -> torch.Tensor:
    """NILW task weights: w_i = (1/L_i) / sum_j(1/L_j).

    Lower loss receives higher weight. Pass only active non-zero losses.
    This is not the temperature-softmax UW-SO formula.
    """
    device = None
    vals = []
    for l in task_losses:
        if isinstance(l, torch.Tensor):
            v = max(l.detach().item(), 1e-8)
            if device is None:
                device = l.device
        else:
            v = max(float(l), 1e-8)
        vals.append(1.0 / v)

    device = device or torch.device("cpu")
    inv = torch.tensor(vals, dtype=torch.float32, device=device)
    return inv / inv.sum()


def compute_uwso_weights(
    task_losses: Sequence,
) -> torch.Tensor:
    """Backward-compatible alias for the historical function name.

    Deprecated: use compute_nilw_weights. The implementation is NILW, not UW-SO.
    """
    return compute_nilw_weights(task_losses)
