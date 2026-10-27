"""PCGrad gradient projection (Yu et al., NeurIPS 2020).

For each task, project its gradient away from conflicting task gradients,
then sum the projected gradients.
https://arxiv.org/abs/2001.06782
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_gradient_projection(
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    threshold: float = 0.0,
) -> torch.Tensor:
    """PCGrad: project grad_a orthogonal to grad_b when cos(g_a,g_b) < -threshold."""
    orig_shape = grad_a.shape
    ga = grad_a.flatten().float()
    gb = grad_b.flatten().float()

    ga_norm = ga.norm()
    gb_norm = gb.norm()

    # If either gradient is zero, nothing to project.
    if ga_norm < 1e-12 or gb_norm < 1e-12:
        return grad_a

    cosine_sim = torch.dot(ga / ga_norm, gb / gb_norm)

    # PCGrad: project ONLY when gradients conflict (cos < -threshold).
    # Equivalently: remove the component of g_a in the direction of g_b
    # when that component is negative (pulling task A away from task B's update).
    if cosine_sim < -threshold:
        # g_a_proj = g_a - (g_a · g_b / ||g_b||²) * g_b
        projection_coeff = torch.dot(ga, gb) / (gb_norm ** 2 + 1e-12)
        ga_projected = ga - projection_coeff * gb
        return ga_projected.reshape(orig_shape).to(grad_a.dtype)
    else:
        return grad_a


def apply_pcgrad_to_gradients(
    gradients: List[torch.Tensor],
    threshold: float = 0.0,
) -> List[torch.Tensor]:
    """Apply PCGrad to a list of per-task gradients (Yu et al., NeurIPS 2020).

    Each input must be a complete flattened gradient over the same shared
    parameter vector.  Projections are cumulative in a randomized task order,
    matching Algorithm 1.  The caller combines the resulting task gradients.
    """
    if len(gradients) < 2:
        return gradients

    originals = [g.detach().clone() for g in gradients]
    projected = []
    for i, gi in enumerate(gradients):
        gi_out = gi.detach().clone()
        for j in torch.randperm(len(originals), device=gi.device).tolist():
            if i != j:
                gi_out = compute_gradient_projection(
                    gi_out, originals[j], threshold
                )
        projected.append(gi_out)

    return projected


class PCGradModule:
    """PCGrad wrapper for multi-task learning (Yu et al., NeurIPS 2020)."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.gradient_cache = {}

    def compute_pcgrad_loss(
        self,
        losses: List[torch.Tensor],
        model: nn.Module,
        retain_graph: bool = True,
    ) -> tuple:
        """Return (mean_loss, pcgrad-projected per-task gradients)."""
        params = list(model.parameters())
        task_grads = []
        for loss in losses:
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=True,
            )
            # Replace None grads with zeros
            grads = tuple(
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, params)
            )
            task_grads.append(grads)

        projected = self._apply_pcgrad(task_grads)
        total_loss = sum(losses) / len(losses)
        return total_loss, projected

    def _apply_pcgrad(
        self,
        gradients: List[Tuple[torch.Tensor, ...]],
    ) -> List[Tuple[torch.Tensor, ...]]:
        """Apply PCGrad over the complete parameter-gradient vector."""
        if len(gradients) < 2:
            return gradients
        shapes = [g.shape for g in gradients[0]]
        sizes = [g.numel() for g in gradients[0]]
        vectors = [torch.cat([g.reshape(-1) for g in task]) for task in gradients]
        projected_vectors = apply_pcgrad_to_gradients(vectors, self.threshold)
        projected_tasks = []
        for vector in projected_vectors:
            offset = 0
            task_parts = []
            for shape, size in zip(shapes, sizes):
                task_parts.append(vector[offset:offset + size].reshape(shape))
                offset += size
            projected_tasks.append(tuple(task_parts))
        return projected_tasks
