import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from cotracker.models.core.cotracker.selective_scan_parallel import parallel_scan
from cotracker.models.core.cotracker.feature_sidecar import SelectiveScan, FeatureSidecar, attach_sidecar, configure_sidecar_runtime, tensor_hash
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline


def sequential(a,b):
    h, result = torch.zeros_like(b[:,0]), []
    for t in range(a.shape[1]):
        h = a[:,t]*h + b[:,t]
        result.append(h)
    return torch.stack(result,1)


class SidecarSpeedSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_scan_forward_and_backward_non_power_two(self):
        for t in (1,2,7,13,60):
            torch.manual_seed(3)
            a = torch.rand(2,t,3,4,dtype=torch.float64,requires_grad=True)
            b = torch.randn_like(a,requires_grad=True)
            weight = torch.randn_like(a)
            y = sequential(a,b)
            ga,gb = torch.autograd.grad((y*weight).sum(),(a,b))
            z = parallel_scan(a,b)
            pa,pb = torch.autograd.grad((z*weight).sum(),(a,b))
            torch.testing.assert_close(y,z,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(ga,pa,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(gb,pb,rtol=1e-12,atol=1e-12)

    def test_gradcheck_underflow_and_point_independence(self):
        a = torch.rand(1,5,2,2,dtype=torch.float64,requires_grad=True)
        b = torch.randn_like(a,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(parallel_scan,(a,b)))
        a = torch.full((2,63,3,4),1e-30)
        b = torch.randn_like(a)
        torch.testing.assert_close(parallel_scan(a,b),sequential(a,b))
        before = parallel_scan(a,b)[0].clone()
        b[1] *= 100
        self.assertTrue(torch.equal(before,parallel_scan(a,b)[0]))

    def test_real_selective_scan_parameter_gradients(self):
        torch.manual_seed(42)
        reference = SelectiveScan(width=8,state=4)
        fast = copy.deepcopy(reference)
        fast.scan_backend,fast.point_chunk_size = 'parallel',2
        x = torch.randn(5,17,8,requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        ref = reference(x)
        par = fast(y)
        torch.testing.assert_close(ref,par,rtol=2e-5,atol=2e-6)
        ref.square().mean().backward()
        par.square().mean().backward()
        torch.testing.assert_close(x.grad,y.grad,rtol=3e-5,atol=2e-6)
        for p,q in zip(reference.parameters(),fast.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=2e-6)

    def test_checkpoint_recomputation_preserves_all_gradients(self):
        for kind in ('mlp','mamba_short','mamba'):
            torch.manual_seed(12)
            reference = FeatureSidecar(kind,input_dim=12,hidden_dim=16,width=8,depth=2).train()
            with torch.no_grad(): reference.output.weight.normal_(std=.02)
            fast = copy.deepcopy(reference)
            wrapper = SimpleNamespace(updateformer=SimpleNamespace(feature_sidecar=fast))
            configure_sidecar_runtime(wrapper,backend='parallel',point_chunk_size=2,checkpoint_blocks=True)
            hidden,rich = torch.randn(1,3,23,16),torch.randn(1,3,23,12)
            a,b = reference.residual(hidden,rich),fast.residual(hidden,rich)
            torch.testing.assert_close(a,b,atol=2e-6,rtol=3e-5)
            a.square().mean().backward()
            b.square().mean().backward()
            for p,q in zip(reference.parameters(),fast.parameters()):
                self.assertIsNotNone(p.grad)
                torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=3e-5)

    def test_real_model_runtime_default_gain_zero_and_pixel_cap(self):
        torch.manual_seed(4)
        model = CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60).eval()
        for p in model.parameters(): p.requires_grad_(False)
        x = torch.randn(1,3,5,model.updateformer.input_transform.in_features)
        with torch.no_grad(): base = model.updateformer(x)
        attach_sidecar(model,'mamba')
        old = tensor_hash(model.state_dict())
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(base,model.updateformer(x)))
            model.updateformer.feature_sidecar.output.weight.normal_(std=.02)
            reference = model.updateformer(x)
            configure_sidecar_runtime(model,backend='parallel',point_chunk_size=2)
            parallel = model.updateformer(x)
            torch.testing.assert_close(reference,parallel,rtol=2e-5,atol=2e-6)
            configure_sidecar_runtime(model,gain=0)
            self.assertTrue(torch.equal(base,model.updateformer(x)))
            configure_sidecar_runtime(model,gain=.5,max_correction_pixels=.5)
            capped = model.updateformer(x)
            norms = torch.linalg.vector_norm(capped[...,:2]-base[...,:2],dim=-1)*model.stride
            self.assertLessEqual(float(norms.max()),.500001)
            self.assertTrue(torch.equal(base[...,2:],capped[...,2:]))
        self.assertEqual(old,tensor_hash(model.state_dict()))
        for kwargs in ({'gain':float('nan')},{'gain':2},{'max_correction_pixels':-1},{'backend':'unknown'},{'point_chunk_size':-2}):
            with self.assertRaises(ValueError): configure_sidecar_runtime(model,**kwargs)

    def test_runtime_file_is_loaded_without_new_checkpoint_tensors(self):
        runtime = {'backend':'parallel','point_chunk_size':16,'gain':.25,
                   'max_correction_pixels':.5,'checkpoint_blocks':True}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'runtime.json'
            path.write_text(json.dumps(runtime),encoding='utf-8-sig')
            model = CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60)
            with patch.dict(os.environ,{'TRACKRAD_SIDECAR_RUNTIME':str(path)}):
                report=attach_sidecar(model,'mamba')
            self.assertEqual(model.updateformer.feature_sidecar.runtime,runtime)
            self.assertEqual(report['execution_runtime'],runtime)
            self.assertFalse(any('runtime' in name or 'scan_backend' in name for name in model.state_dict()))
            path.write_text(json.dumps({**runtime,'unknown':1}),encoding='utf-8')
            other = CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60)
            with patch.dict(os.environ,{'TRACKRAD_SIDECAR_RUNTIME':str(path)}):
                with self.assertRaisesRegex(ValueError,'unexpected sidecar runtime'):
                    attach_sidecar(other,'mamba')


if __name__ == '__main__': unittest.main()
