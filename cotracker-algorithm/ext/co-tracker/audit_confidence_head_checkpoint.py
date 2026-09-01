"""Verify that a checkpoint changes only CoTracker3's confidence-head row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ALLOWED_WEIGHT = "updateformer.vis_conf_head.weight"
ALLOWED_BIAS = "updateformer.vis_conf_head.bias"
CONFIDENCE_ROW = 1


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and "model" in value:
        value = value["model"]
    if not isinstance(value, dict):
        raise TypeError(f"{path} does not contain a state dict")
    if value and next(iter(value)).startswith("module."):
        value = {key.removeprefix("module."): tensor for key, tensor in value.items()}
    return value


def audit(
    base_path: Path, tuned_path: Path, scope: str = "head"
) -> dict[str, object]:
    if scope not in ("head", "updateformer"):
        raise ValueError(f"unsupported audit scope: {scope}")
    base = _state_dict(base_path)
    tuned = _state_dict(tuned_path)
    if set(base) != set(tuned):
        missing = sorted(set(base) - set(tuned))
        extra = sorted(set(tuned) - set(base))
        raise AssertionError(f"state-dict keys differ; missing={missing}, extra={extra}")

    changed: list[str] = []
    changed_elements = 0
    maximum_delta = 0.0
    for key, base_tensor in base.items():
        tuned_tensor = tuned[key]
        if base_tensor.shape != tuned_tensor.shape:
            raise AssertionError(f"shape changed for {key}")
        unequal = base_tensor != tuned_tensor
        if not bool(unequal.any()):
            continue
        is_head = key in (ALLOWED_WEIGHT, ALLOWED_BIAS)
        is_shared_updateformer = (
            scope == "updateformer"
            and key.startswith("updateformer.")
            and not key.startswith("updateformer.flow_head.")
            and not key.startswith("updateformer.vis_conf_head.")
        )
        if not is_head and not is_shared_updateformer:
            raise AssertionError(f"forbidden parameter changed: {key}")
        if is_head:
            forbidden = unequal.clone()
            forbidden[CONFIDENCE_ROW] = False
            if bool(forbidden.any()):
                raise AssertionError(f"visibility row changed: {key}")
        changed.append(key)
        changed_elements += int(unequal.sum().item())
        maximum_delta = max(
            maximum_delta,
            float((tuned_tensor - base_tensor).abs().max().item()),
        )

    if not changed:
        raise AssertionError("confidence row did not change")
    if not any(key in (ALLOWED_WEIGHT, ALLOWED_BIAS) for key in changed):
        raise AssertionError("confidence row did not change")
    if scope == "updateformer" and not any(
        key not in (ALLOWED_WEIGHT, ALLOWED_BIAS) for key in changed
    ):
        raise AssertionError("shared UpdateFormer parameters did not change")
    invariant = (
        "Only row 1 of updateformer.vis_conf_head may change"
        if scope == "head"
        else (
            "Only shared updateformer parameters and row 1 of "
            "updateformer.vis_conf_head may change; flow_head stays frozen"
        )
    )
    return {
        "passed": True,
        "scope": scope,
        "base_checkpoint": str(base_path),
        "tuned_checkpoint": str(tuned_path),
        "changed_tensors": changed,
        "changed_elements": changed_elements,
        "maximum_absolute_delta": maximum_delta,
        "invariant": invariant,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=["head", "updateformer"], default="head")
    args = parser.parse_args()
    result = audit(args.base, args.tuned, scope=args.scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
