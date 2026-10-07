import unittest
import torch
from cotracker.models.core.cotracker.feature_sidecar import FeatureSidecar, attach_sidecar, tensor_hash, KINDS
from cotracker.models.core.cotracker.cotracker import EfficientUpdateFormer
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline


class FeatureSidecarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

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
