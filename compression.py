import copy

import torch
from torch import nn
import torch.nn.utils.prune as prune


def apply_pruning(model, amount: float = 0.5):
    """Return a copy of ``model`` with structured pruning applied."""
    pruned = copy.deepcopy(model)
    for layer in [pruned.seq[0], pruned.seq[2]]:
        prune.ln_structured(layer, name="weight", amount=amount, n=2, dim=0)
        prune.remove(layer, "weight")
    return pruned


def apply_low_rank(model, rank: int = 32):
    """Approximate linear layers with a low-rank factorization."""
    low_rank = copy.deepcopy(model)
    new_layers: list[nn.Module] = []
    for layer in low_rank.seq:
        if isinstance(layer, nn.Linear):
            W = layer.weight.data
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = min(rank, S.size(0))
            U_r = U[:, :r]
            S_r = S[:r]
            Vh_r = Vh[:r, :]
            B = S_r.unsqueeze(1) * Vh_r
            first = nn.Linear(Vh_r.shape[1], r, bias=False)
            first.weight.data = B
            second = nn.Linear(r, U_r.shape[0], bias=True)
            second.weight.data = U_r
            second.bias.data = layer.bias.data
            new_layers.extend([first, second])
        else:
            new_layers.append(layer)
    low_rank.seq = nn.Sequential(*new_layers)
    return low_rank


def apply_operator_fusion(model, path: str = "model_fused.pt"):
    """Export a TorchScript version with operator fusion."""
    fused = copy.deepcopy(model).eval()
    scripted = torch.jit.script(fused)
    optimized = torch.jit.optimize_for_inference(scripted)
    torch.jit.save(optimized, path)
    return optimized
