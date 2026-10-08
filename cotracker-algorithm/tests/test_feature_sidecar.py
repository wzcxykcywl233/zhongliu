import unittest
import torch
from cotracker.models.core.cotracker.feature_sidecar import (
    FeatureSidecar, attach_sidecar, tensor_hash, KINDS,
    validate_initialization_outputs, attach_sidecar_verified,
    set_sidecar_training_mode, assert_sidecar_training_mode,
)
from cotracker.models.core.cotracker.cotracker import EfficientUpdateFormer
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline


class FeatureSidecarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_remote_freeze_transition_does_not_relax_sidecar_identity(self):
        original = (torch.tensor([1.]), torch.tensor([.9]), torch.tensor([.9]))
        # Regression for the three transition differences in the remote log.
        frozen = tuple(x - d for x,d in zip(original,(5.7220458984375e-5,4.231929779052734e-6,1.1324882507324219e-6)))
        report = validate_initialization_outputs(original, frozen, tuple(x.clone() for x in frozen))
        self.assertEqual(report['initialization_reference'], 'frozen-pretrained')
        self.assertEqual(report['initialization_max_abs_difference'], [0.,0.,0.])
        changed = list(frozen)
        changed[1] = changed[1] + 1e-7
        with self.assertRaisesRegex(ValueError, 'zero initialization changed'):
            validate_initialization_outputs(original, frozen, changed)
        changed[1] = torch.tensor([float('nan')])
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            validate_initialization_outputs(original, frozen, changed)

    def test_real_full_model_verified_initialization(self):
        model = CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60)
        old = tensor_hash(model.state_dict())
        report = attach_sidecar_verified(model, 'mlp')
        self.assertEqual(report['base_tensor_sha256'],old)
        self.assertEqual(report['initialization_max_abs_difference'],[0.,0.,0.])

    def test_first_training_batch_in_mixed_mode_for_every_branch(self):
        # Exercise the real offline train_data path, not just a standalone MLP.
        for kind in KINDS:
            torch.manual_seed(21)
            model = CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60)
            attach_sidecar(model,kind)
            before_hash = tensor_hash(model.state_dict())
            set_sidecar_training_mode(model)
            video = torch.rand(1,6,3,64,64) * 255
            queries = torch.tensor([[[0.,16.,16.],[2.,32.,32.],[0.,48.,48.]]])
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
            out = model(video=video,queries=queries,iters=2,is_train=True)
            self.assertIsNotNone(out[3])
            target = out[0].detach() + 1.
            loss = sum((prediction-target).square().mean() for prediction in out[3][0][0])
            optimizer.zero_grad()
            loss.backward()
            self.assertGreater(model.updateformer.feature_sidecar.output.weight.grad.abs().sum(),0)
            self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if '.feature_sidecar.' not in n))
            optimizer.step()
            assert_sidecar_training_mode(model)
            self.assertEqual(before_hash,tensor_hash(model.state_dict()))
            # Evaluator switches all modules to eval; training must restore ONLY
            # the branch afterwards, including at epoch boundaries.
            model.eval()
            with self.assertRaisesRegex(RuntimeError,'base eval and branch train'):
                assert_sidecar_training_mode(model)
            set_sidecar_training_mode(model)
            model.fnet.train()
            with self.assertRaisesRegex(RuntimeError,'frozen module entered train'):
                assert_sidecar_training_mode(model)
            set_sidecar_training_mode(model)
            model.updateformer.flow_head.weight.requires_grad_(True)
            with self.assertRaisesRegex(RuntimeError,'unexpected trainable/frozen'):
                assert_sidecar_training_mode(model)

    def test_identity_gradient_and_no_point_leak(self):
        for kind in KINDS:
            torch.manual_seed(8)
            module = FeatureSidecar(kind, input_dim=12, hidden_dim=16, width=8, depth=1)
            hidden = torch.randn(1, 2, 13, 16)
            rich = torch.randn(1, 2, 13, 12)
            self.assertTrue(torch.equal(module(hidden,rich),hidden))
            module(hidden,rich).square().sum().backward()
            self.assertGreater(module.output.weight.grad.abs().sum(),0)
            self.assertTrue(all(p.grad is not None for p in module.parameters()))
            with torch.no_grad():
                module.output.weight.normal_(std=.1)
                a = module(hidden,rich)
                rich[:,1] *= 100
                b = module(hidden,rich)
            self.assertTrue(torch.equal(a[:,0],b[:,0]))

    def test_long_vs_reset_same_weights(self):
        long = FeatureSidecar('mamba',input_dim=4,hidden_dim=8,width=8,depth=1)
        short = FeatureSidecar('mamba_short',input_dim=4,hidden_dim=8,width=8,depth=1)
        with torch.no_grad(): long.output.weight.fill_(.1)
        short.load_state_dict(long.state_dict())
        h, x = torch.randn(1,1,22,8), torch.randn(1,1,22,4)
        self.assertFalse(torch.equal(long(h,x),short(h,x)))

    def test_real_updateformer_identity_and_frozen_hash(self):
        model = CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60).eval()
        x = torch.randn(1,3,4,model.updateformer.input_transform.in_features)
        with torch.no_grad(): a = model.updateformer(x)
        old = tensor_hash(model.state_dict())
        proof = attach_sidecar(model, 'mamba')
        with torch.no_grad(): b = model.updateformer(x)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(old,proof['base_tensor_sha256'])
        self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters() if '.feature_sidecar.' not in n))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        optimizer.zero_grad()
        model.updateformer(x)[...,:2].square().sum().backward()
        optimizer.step()
        self.assertEqual(old,tensor_hash(model.state_dict()))
        self.assertFalse(torch.equal(a[...,:2],model.updateformer(x)[...,:2]))
        # No direct branch injection into frozen visibility/confidence head.
        self.assertTrue(torch.equal(a[...,2:],model.updateformer(x)[...,2:]))


if __name__ == '__main__':
    unittest.main()
