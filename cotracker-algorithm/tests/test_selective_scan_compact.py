import copy
import math
from pathlib import Path
import sys
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ext'/'co-tracker'))
from cotracker.models.core.cotracker.selective_scan_compact import compact_selective_scan
from cotracker.models.core.cotracker.feature_sidecar import SelectiveScan, FeatureSidecar, configure_sidecar_runtime, attach_sidecar, tensor_hash
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from types import SimpleNamespace


def reference(u,d,a,b,c,skip):
    h=torch.zeros((u.shape[0],u.shape[2],a.shape[1]),device=u.device,dtype=u.dtype)
    result=[]
    for t in range(u.shape[1]):
        h=torch.exp(d[:,t,:,None]*a)*h+d[:,t,:,None]*b[:,t,None,:]*u[:,t,:,None]
        result.append((h*c[:,t,None,:]).sum(-1)+skip*u[:,t])
    return torch.stack(result,1)


def inputs(t,dtype=torch.float64):
    return tuple(x.requires_grad_() for x in (torch.randn(2,t,3,dtype=dtype),torch.rand(2,t,3,dtype=dtype),
        -torch.rand(3,4,dtype=dtype),torch.randn(2,t,4,dtype=dtype),torch.randn(2,t,4,dtype=dtype),torch.randn(3,dtype=dtype)))


class CompactScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_same_forward_all_six_gradients_and_cross_checkpoint_state(self):
        for t in (1,7,8,9,17,60):
            torch.manual_seed(12)
            args=inputs(t)
            weight=torch.randn(2,t,3,dtype=torch.float64)
            expected=reference(*args)
            g=torch.autograd.grad((expected*weight).sum(),args)
            actual=compact_selective_scan(*args)
            q=torch.autograd.grad((actual*weight).sum(),args)
            self.assertTrue(torch.equal(expected,actual))
            for a,b in zip(g,q): torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)

    def test_gradcheck_noncontiguous_and_underflow_no_decay_division(self):
        torch.manual_seed(9)
        self.assertTrue(torch.autograd.gradcheck(compact_selective_scan,inputs(9)))
        args=list(inputs(17))
        args[0]=args[0].flip(1).transpose(0,1).contiguous().transpose(0,1).detach().requires_grad_()
        self.assertFalse(args[0].is_contiguous())
        args[2]=torch.full_like(args[2],-1000.).requires_grad_()
        expected=reference(*args);actual=compact_selective_scan(*args)
        torch.testing.assert_close(expected,actual)
        g=torch.autograd.grad(expected.sum(),args);q=torch.autograd.grad(actual.sum(),args)
        for a,b in zip(g,q): torch.testing.assert_close(a,b)

    def test_saved_state_scales_with_sparse_anchors_not_expanded_time_graph(self):
        args=inputs(60,torch.float32)
        saved=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda x:(saved.append(x.shape) or x),lambda x:x):
            compact_selective_scan(*args).sum().backward()
        self.assertEqual(len(saved),7)
        self.assertEqual(saved[-1],torch.Size((math.ceil(60/8),2,3,4)))
        self.assertLess(sum(math.prod(s) for s in saved),2*60*3*4*2)

    def test_real_scan_module_parameter_and_input_gradients_fp32(self):
        for t in (17,60):
            torch.manual_seed(8)
            ref=SelectiveScan(width=8,state=4);new=copy.deepcopy(ref);new.scan_backend='compact'
            x=torch.randn(5,t,8,requires_grad=True);y=x.detach().clone().requires_grad_()
            a,b=ref(x),new(y)
            self.assertTrue(torch.equal(a,b))
            target=torch.randn_like(a)
            ((a-target)**2).mean().backward();((b-target)**2).mean().backward()
            torch.testing.assert_close(x.grad,y.grad,rtol=2e-4,atol=3e-6)
            for p,q in zip(ref.parameters(),new.parameters()): torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=3e-6)

    def test_four_iterations_short_long_branches_and_optimizer_update(self):
        for kind in ('mlp','mamba_short','mamba'):
            torch.manual_seed(18)
            ref=FeatureSidecar(kind,input_dim=8,hidden_dim=12,width=8,depth=3).train()
            with torch.no_grad(): ref.output.weight.normal_(std=.01)
            new=copy.deepcopy(ref)
            configure_sidecar_runtime(SimpleNamespace(updateformer=SimpleNamespace(feature_sidecar=new)),backend='compact')
            h,r=torch.randn(1,3,60,12),torch.randn(1,3,60,8)
            before=set(new.state_dict())
            a=[ref.residual(h,r) for _ in range(4)];b=[new.residual(h,r) for _ in range(4)]
            for x,y in zip(a,b): self.assertTrue(torch.equal(x,y))
            sum((x-.01).square().mean() for x in a).backward()
            sum((x-.01).square().mean() for x in b).backward()
            for p,q in zip(ref.parameters(),new.parameters()):
                self.assertIsNotNone(q.grad)
                torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=3e-6)
            torch.optim.AdamW(ref.parameters(),lr=5e-5).step();torch.optim.AdamW(new.parameters(),lr=5e-5).step()
            for p,q in zip(ref.parameters(),new.parameters()): torch.testing.assert_close(p,q,rtol=2e-4,atol=3e-6)
            self.assertEqual(before,set(new.state_dict()))

    def test_full_model_outputs_and_frozen_parameters_preserved(self):
        torch.manual_seed(4)
        model=CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60).eval()
        attach_sidecar(model,'mamba');model.eval()
        with torch.no_grad(): model.updateformer.feature_sidecar.output.weight.normal_(std=.001)
        checksum=tensor_hash(model.state_dict())
        video=torch.rand(1,3,3,64,64)*255
        queries=torch.tensor([[[0.,16.,16.],[1.,32.,32.]]])
        with torch.no_grad():
            expected=model(video=video,queries=queries,iters=2)
            configure_sidecar_runtime(model,backend='compact')
            actual=model(video=video,queries=queries,iters=2)
        for x,y in zip(expected[:3],actual[:3]): torch.testing.assert_close(x,y,rtol=3e-5,atol=3e-6)
        self.assertEqual(checksum,tensor_hash(model.state_dict()))
        with self.assertRaises(ValueError): configure_sidecar_runtime(model,backend='compact',point_chunk_size=16)


if __name__=='__main__': unittest.main()
