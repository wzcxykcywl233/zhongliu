"""Utilities for an isolated CoTracker3 confidence-head ablation."""

from __future__ import annotations

from collections.abc import Iterable

import torch


CONFIDENCE_ROW = 1


def configure_confidence_head_only(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Freeze the model and expose only the confidence row of ``vis_conf_head``.

    CoTracker3's separate visibility/confidence head has two output rows.  A
    tensor can only be trainable as a whole, so hooks mask the visibility row
    while the optimizer is restricted to the head tensors.  The caller must
    use zero weight decay, otherwise AdamW would still alter the masked row.
    """

    updateformer = getattr(model, "updateformer", None)
    head = getattr(updateformer, "vis_conf_head", None)
    if not isinstance(head, torch.nn.Linear):
        raise ValueError(
            "confidence-head-only tuning requires updateformer.vis_conf_head"
        )
    if head.out_features != 2:
        raise ValueError(
            "expected vis_conf_head outputs [visibility, confidence], "
            f"got {head.out_features} rows"
        )
    if head.bias is None:
        raise ValueError("confidence-head-only tuning requires a bias term")

    for parameter in model.parameters():
        parameter.requires_grad = False

    head.weight.requires_grad = True
    head.bias.requires_grad = True

    weight_mask = torch.zeros_like(head.weight)
    weight_mask[CONFIDENCE_ROW] = 1
    bias_mask = torch.zeros_like(head.bias)
    bias_mask[CONFIDENCE_ROW] = 1

    handles = (
        head.weight.register_hook(lambda gradient: gradient * weight_mask),
        head.bias.register_hook(lambda gradient: gradient * bias_mask),
    )
    # Retain the handles for the model lifetime and make repeated setup fail
    # loudly instead of stacking hooks during an accidental second call.
    if hasattr(model, "_confidence_head_only_hook_handles"):
        for handle in handles:
            handle.remove()
        raise RuntimeError("confidence-head-only gradient hooks already configured")
    model._confidence_head_only_hook_handles = handles
    return [head.weight, head.bias]


def trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    """Return stable names for experiment metadata and tests."""

    return [name for name, value in model.named_parameters() if value.requires_grad]


def trainable_parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)
