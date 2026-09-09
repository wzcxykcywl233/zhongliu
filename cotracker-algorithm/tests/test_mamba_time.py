import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


PATH = (
    Path(__file__).resolve().parents[1]
    / "ext"
    / "co-tracker"
    / "cotracker"
    / "models"
    / "core"
    / "cotracker"
    / "mamba_time.py"
)
spec = importlib.util.spec_from_file_location("mamba_time_standalone", PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class MambaTimeTests(unittest.TestCase):
    def test_time_block_preserves_shape_and_backpropagates(self) -> None:
        block = module.MambaTimeBlock(hidden_size=12, state_dim=4)
        inputs = torch.randn(3, 7, 12, requires_grad=True)
        output = block(inputs)
        self.assertEqual(output.shape, inputs.shape)
        output.square().mean().backward()
        self.assertIsNotNone(inputs.grad)

    def test_trajectory_refiner_starts_as_identity(self) -> None:
        refiner = module.MambaTrajectoryRefiner(
            model_dim=12, state_dim=4, blocks=1
        )
        trajectories = torch.rand(1, 6, 5, 2) * 7
        reliability = torch.ones(1, 6, 5)
        refined, residual = refiner(
            trajectories, reliability, reliability, (8, 8)
        )
        self.assertTrue(torch.equal(refined, trajectories))
        self.assertTrue(torch.count_nonzero(residual) == 0)

    def test_trajectory_refiner_is_bounded(self) -> None:
        refiner = module.MambaTrajectoryRefiner(
            model_dim=12, state_dim=4, blocks=1, max_residual_pixels=3.0
        )
        torch.nn.init.constant_(refiner.output_proj.bias, 20.0)
        trajectories = torch.full((1, 4, 3, 2), 5.0)
        reliability = torch.ones(1, 4, 3)
        refined, residual = refiner(
            trajectories, reliability, reliability, (16, 16)
        )
        self.assertLessEqual(float(residual.detach().abs().max()), 3.00001)
        self.assertTrue(torch.all(refined >= 0))
        self.assertTrue(torch.all(refined <= 15))

    def test_adapter_freezes_the_base_model(self) -> None:
        model = nn.Linear(3, 3)
        trainable = module.attach_trajectory_mamba_refiner(model)
        self.assertTrue(hasattr(model, "trajectory_mamba_refiner"))
        self.assertFalse(model.weight.requires_grad)
        self.assertTrue(trainable)
        self.assertTrue(all(parameter.requires_grad for parameter in trainable))

    def test_replacement_freezes_everything_except_new_time_blocks(self) -> None:
        class DummyUpdateFormer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hidden_size = 12
                self.time_blocks = nn.ModuleList([nn.Linear(12, 12) for _ in range(3)])
                self.space_block = nn.Linear(12, 12)

        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.updateformer = DummyUpdateFormer()
                self.encoder = nn.Linear(3, 3)

        model = DummyModel()
        trainable = module.replace_updateformer_time_attention(
            model, state_dim=4
        )
        self.assertEqual(len(model.updateformer.time_blocks), 3)
        self.assertTrue(
            all(
                isinstance(block, module.MambaTimeBlock)
                for block in model.updateformer.time_blocks
            )
        )
        self.assertFalse(model.encoder.weight.requires_grad)
        self.assertFalse(model.updateformer.space_block.weight.requires_grad)
        self.assertTrue(trainable)
        self.assertTrue(all(parameter.requires_grad for parameter in trainable))


if __name__ == "__main__":
    unittest.main()
