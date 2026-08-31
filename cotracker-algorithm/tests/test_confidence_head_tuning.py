import sys
import unittest
from pathlib import Path

import torch


EXT_ROOT = Path(__file__).resolve().parents[1] / "ext" / "co-tracker"
sys.path.insert(0, str(EXT_ROOT))

from cotracker.utils.confidence_head_tuning import (  # noqa: E402
    configure_confidence_head_only,
    trainable_parameter_names,
)


class _TinyUpdateFormer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.vis_conf_head = torch.nn.Linear(3, 2)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(3, 3)
        self.updateformer = _TinyUpdateFormer()


class ConfidenceHeadTuningTests(unittest.TestCase):
    def test_only_head_tensors_are_exposed_and_visibility_gradients_are_zero(self):
        model = _TinyModel()
        parameters = configure_confidence_head_only(model)

        self.assertEqual(
            trainable_parameter_names(model),
            [
                "updateformer.vis_conf_head.weight",
                "updateformer.vis_conf_head.bias",
            ],
        )
        loss = model.updateformer.vis_conf_head(torch.ones(1, 3)).sum()
        loss.backward()

        self.assertTrue(torch.equal(parameters[0].grad[0], torch.zeros(3)))
        self.assertEqual(float(parameters[1].grad[0]), 0.0)
        self.assertTrue(torch.equal(parameters[0].grad[1], torch.ones(3)))
        self.assertEqual(float(parameters[1].grad[1]), 1.0)
        self.assertTrue(all(value.grad is None for value in model.encoder.parameters()))

    def test_zero_weight_decay_optimizer_preserves_visibility_row(self):
        model = _TinyModel()
        parameters = configure_confidence_head_only(model)
        before_weight = model.updateformer.vis_conf_head.weight.detach().clone()
        before_bias = model.updateformer.vis_conf_head.bias.detach().clone()
        optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.0)

        loss = model.updateformer.vis_conf_head(torch.ones(1, 3)).sum()
        loss.backward()
        optimizer.step()

        self.assertTrue(
            torch.equal(model.updateformer.vis_conf_head.weight[0], before_weight[0])
        )
        self.assertEqual(
            float(model.updateformer.vis_conf_head.bias[0].detach()),
            float(before_bias[0]),
        )
        self.assertFalse(
            torch.equal(model.updateformer.vis_conf_head.weight[1], before_weight[1])
        )


if __name__ == "__main__":
    unittest.main()
