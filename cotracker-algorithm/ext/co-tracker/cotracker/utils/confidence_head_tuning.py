"""Utilities for an isolated CoTracker3 confidence-head ablation."""

from __future__ import annotations

from collections.abc import Iterable

import torch


CONFIDENCE_ROW = 1


def _confidence_head(model: torch.nn.Module) -> torch.nn.Linear:
    updateformer = getattr(model, "updateformer", None)
    head = getattr(updateformer, "vis_conf_head", None)
    if not isinstance(head, torch.nn.Linear):
        raise ValueError(
            "confidence tuning requires updateformer.vis_conf_head"
        )
    if head.out_features != 2:
        raise ValueError(
            "expected vis_conf_head outputs [visibility, confidence], "
            f"got {head.out_features} rows"
        )
    if head.bias is None:
        raise ValueError("confidence tuning requires a bias term")
    return head


def _mask_visibility_row(
    model: torch.nn.Module, head: torch.nn.Linear
) -> list[torch.nn.Parameter]:
    """Mask row zero and retain the hook handles for the model lifetime."""

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


def configure_confidence_head_only(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Freeze the model and expose only the confidence row of ``vis_conf_head``.

    CoTracker3's separate visibility/confidence head has two output rows.  A
    tensor can only be trainable as a whole, so hooks mask the visibility row
    while the optimizer is restricted to the head tensors.  The caller must
    use zero weight decay, otherwise AdamW would still alter the masked row.
    """

    head = _confidence_head(model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return _mask_visibility_row(model, head)


def configure_confidence_updateformer(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Train UpdateFormer representations with confidence-only supervision.

    The coordinate projection and visibility output row stay frozen.  Shared
    UpdateFormer blocks are trainable, so coordinate predictions can still
    change through their inputs even though ``flow_head`` itself is immutable.
    The returned head parameters require a zero-weight-decay optimizer group.
    """

    head = _confidence_head(model)
    for parameter in model.parameters():
        parameter.requires_grad = False

    shared_parameters = []
    for name, parameter in model.updateformer.named_parameters():
        if name.startswith("flow_head.") or name.startswith("vis_conf_head."):
            continue
        parameter.requires_grad = True
        shared_parameters.append(parameter)
    if not shared_parameters:
        raise ValueError("UpdateFormer exposes no shared trainable parameters")

    head_parameters = _mask_visibility_row(model, head)
    return shared_parameters, head_parameters


def trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    """Return stable names for experiment metadata and tests."""

    return [name for name, value in model.named_parameters() if value.requires_grad]


def trainable_parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)
