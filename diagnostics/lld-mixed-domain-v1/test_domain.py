import copy
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
import protocol as P
import domain as D
import train_domain as T
import evaluation as E

torch.set_num_threads(2)


class DomainTests(unittest.TestCase):
    def test_grl_forward_and_scaled_opposite_gradient(self):
        x=torch.tensor([1.,-2.],requires_grad=True)
        y=D.reversal(x,.01)
        self.assertTrue(torch.equal(x,y))
        (y*torch.tensor([2.,3.])).sum().backward()
        torch.testing.assert_close(x.grad,torch.tensor([-.02,-.03]))

    def test_zero_detaches(self):
        x=torch.ones(2,requires_grad=True)
        self.assertFalse(D.reversal(x,0).requires_grad)
        for v in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): D.reversal(x,v)

    def test_schedule_bounded_native_steps(self):
        values=[D.schedule(i,.01) for i in range(400)]
        self.assertEqual(values[0],0)
        self.assertEqual(values[250],.01)
        self.assertEqual(values[399],.01)
        self.assertEqual(values,sorted(values))
        with self.assertRaises(ValueError): D.schedule(-1,.01)

    def test_head_gradient_not_reversed_or_scaled(self):
        torch.manual_seed(1); h=D.DomainHead(8)
        x=torch.randn(1,8,requires_grad=True)
        loss,_=D.domain_loss(h,x,'lld_synthetic',1.)
        gx,gh=torch.autograd.grad(loss,(x,h[-1].weight))
        loss2,_=D.domain_loss(h,x,'lld_synthetic',.01)
        gx2,gh2=torch.autograd.grad(loss2,(x,h[-1].weight))
        torch.testing.assert_close(gx2,gx*.01)
        torch.testing.assert_close(gh2,gh)
        for domain in ('bad','test-38'):
            with self.assertRaises(ValueError): D.domain_loss(h,x,domain,.01)

    def test_detached_head_independent_clip_student_exact(self):
        torch.manual_seed(2); base=nn.Linear(8,8); probe=copy.deepcopy(base)
        head=D.DomainHead(8)
        a=torch.optim.AdamW(base.parameters(),lr=5e-5)
        b=torch.optim.AdamW(probe.parameters(),lr=5e-5)
        c=torch.optim.AdamW(head.parameters(),lr=1e-4)
        initial=copy.deepcopy(head.state_dict())
        for i in range(6):
            x=torch.randn(1,8)
            a.zero_grad(); b.zero_grad(); c.zero_grad()
            y=base(x); z=probe(x)
            aux,_=D.domain_loss(head,z,'lld_synthetic' if i%2 else 'trackrad',0)
            T.optimize(y.square().sum(),y.new_zeros(()),list(base.parameters()),[],a,None)
            T.optimize(z.square().sum(),aux*10000,list(probe.parameters()),list(head.parameters()),b,c)
        for k,v in base.state_dict().items(): self.assertTrue(torch.equal(v,probe.state_dict()[k]))
        self.assertTrue(any(not torch.equal(v,head.state_dict()[k]) for k,v in initial.items()))

    def test_tap_fnet_chunks_query_excluded(self):
        class Model(nn.Module):
            def __init__(self): super().__init__(); self.fnet=nn.Identity()
        m=Model(); tap=D.Tap(m,'fnet'); tap.reset(5)
        x=torch.arange(5.).view(5,1,1,1).expand(5,3,2,2).requires_grad_()
        m.fnet(x[:2]); m.fnet(x[2:])
        torch.testing.assert_close(tap.embedding(),torch.full((1,3),2.5))
        tap.embedding().sum().backward()
        self.assertEqual(x.grad[0].abs().sum(),0)
        self.assertGreater(x.grad[1:].abs().sum(),0)
        tap.close(); self.assertFalse(m.fnet._forward_hooks)

    def test_tap_sidecar_last_iteration_native_only(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__(); self.updateformer=nn.Module(); self.updateformer.feature_sidecar=nn.Module()
                self.updateformer.feature_sidecar.blocks=nn.ModuleList([nn.Identity()])
        m=Model(); tap=D.Tap(m,'sidecar'); tap.reset(3)
        block=m.updateformer.feature_sidecar.blocks[-1]
        block(torch.ones(2,3,8)); x=torch.full((2,3,8),2.); x[:,0]=100
        block(x)
        torch.testing.assert_close(tap.embedding(),torch.full((1,8),2.))
        with self.assertRaises(ValueError): block(torch.ones(2,4,8))
        tap.reset(2)
        with self.assertRaises(ValueError): tap.embedding()
        tap.close()

    def test_real_tracker_task_and_domain_recipients(self):
        from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
        from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar,configure_sidecar_runtime
        torch.manual_seed(9)
        m=CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60)
        attach_sidecar(m,'mamba'); configure_sidecar_runtime(m,backend='compact')
        q=torch.tensor([[[0.,20.,20.],[0.,30.,30.]]]); v=torch.rand(1,3,3,64,64)*255
        for arm,kind in [('S010','sidecar'),('F010','fnet')]:
            D.configure(m,arm); tap=D.Tap(m,kind); tap.reset(3)
            out=m(video=v,queries=q,iters=2,is_train=True)
            # The real existing trajectory objective reaches partially unfrozen fnet.
            target=dict(tracks=out[0].detach()+1,visibility=torch.ones_like(out[1]))
            task=T.T.S.training_loss(out,target)
            side=[p for n,p in m.named_parameters() if n.startswith(P.PREFIX)]
            enc=[p for n,p in m.named_parameters() if n.startswith(P.ENCODER_PREFIXES)]
            recipient=side if kind=='sidecar' else enc
            self.assertGreater(D.gradient_norm(task,recipient),0)
            head=D.DomainHead(tap.embedding().shape[-1])
            loss,_=D.domain_loss(head,tap.embedding(),'lld_synthetic',.01)
            self.assertGreater(D.gradient_norm(loss,recipient),0)
            if kind=='fnet': self.assertEqual(D.gradient_norm(loss,side),0.)
            (task+loss).backward()
            for name,p in m.named_parameters():
                allow=name.startswith(P.PREFIX) or (arm.startswith('F') and name.startswith(P.ENCODER_PREFIXES))
                self.assertEqual(p.requires_grad,allow)
                if not allow: self.assertIsNone(p.grad)
            m.zero_grad(set_to_none=True); tap.close()

    def test_receipt_detects_tamper(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root=Path(d); P.C.save(root/'plan.json',{'x':1}); P.C.save(root/'f.json',{'x':2})
            P.receipt(root,root/'proof.json',[root/'f.json'])
            self.assertTrue(P.cached(root,root/'proof.json'))
            P.C.save(root/'f.json',{'x':3})
            with self.assertRaises(ValueError): P.cached(root,root/'proof.json')

    def metrics(self,seeds):
        values={}
        for a in ['pretrained']+list(P.ARMS):
            for s in seeds: values[a,s]={'DSC':.90,'HD95':4.,'MASD':1.4,'CD':1.5,'D98':.94}
        return values

    def test_selection_no_domain_accuracy_or_test(self):
        m=self.metrics([0]); m['S001',0]['DSC']+=.001; m['S010',0]['DSC']+=.002
        m['S010',0]['D98']-=.01
        # F001 improves upon degraded F0 but not M0, so it cannot promote.
        m['F0',0]['DSC']-=.004; m['F001',0]['DSC']-=.002
        d=E.selection(m,[0],list(P.ARMS))
        self.assertEqual(d['promoted'],['S001'])
        self.assertEqual(d['selection_split'],'validation-10')

    def test_full_two_seed_consistency(self):
        m=self.metrics([0,1,2]); arms=['M0','F0','S001']
        for s,delta in enumerate([.001,-.0001,-.0001]): m['S001',s]['DSC']+=delta
        self.assertEqual(E.selection(m,[0,1,2],arms)['promoted'],[])
        m['S001',1]['DSC']+=.0003
        self.assertEqual(E.selection(m,[0,1,2],arms)['promoted'],['S001'])

    def test_test_split_cannot_enter_smoke(self):
        with patch.object(E,'test_gate',return_value=['M0']):
            with self.assertRaises(ValueError): E.allowed(Path('.'),'smoke','test-38','M0',0)
        with self.assertRaises(ValueError): E.allowed(Path('.'),'pilot','train-40','M0',0)
        with self.assertRaises(ValueError): E.allowed(Path('.'),'smoke','validation-10','pretrained',0)

    def test_missing_cached_labels_never_generate(self):
        with patch.object(P.C,'cached',return_value=None), patch.object(T.T,'labels') as labels:
            with self.assertRaises(ValueError): T.old_labels(Path('.'),{},Path('.'),{},0,0,{})
            labels.assert_not_called()

    def test_full_roster_requires_validation(self):
        with patch.object(P,'decision',return_value={'promoted':[]}): self.assertEqual(P.arms_for(Path('.'),'full'),[])
        with patch.object(P,'decision',return_value={'promoted':['F001']}):
            self.assertEqual(P.arms_for(Path('.'),'full'),['M0','F0','F001'])
        with patch.object(P,'decision',return_value={'promoted':[], 'partial_no_adversarial_qualifies_vs_M0':True}):
            self.assertEqual(P.arms_for(Path('.'),'full'),['M0','F0'])

    def test_smoke_gate_required_before_pilot(self):
        with patch.object(P,'cached',return_value=None):
            with self.assertRaises(ValueError): P.gate(Path('.'),'pilot','S001',0)
        P.gate(Path('.'),'smoke','S001',0)
        with self.assertRaises(ValueError): P.gate(Path('.'),'smoke','S001',1)

    def test_test_roster_must_be_frozen_full_roster(self):
        with patch.object(P,'decision',return_value={'test_arms':['M0','S001']}), patch.object(P,'arms_for',return_value=['M0','F0','S001']):
            with self.assertRaises(ValueError): P.test_gate(Path('.'))

    def test_invalid_nonfinite_objective_no_step(self):
        from unittest.mock import Mock
        p=nn.Parameter(torch.ones(1)); opt=Mock()
        with self.assertRaises(ValueError): T.optimize(p.sum()*float('nan'),p.new_zeros(()),[p],[],opt,None)
        opt.step.assert_not_called()

    def test_resume_optimizer_and_classifier_equivalence(self):
        def create():
            torch.manual_seed(7); m=nn.Linear(8,8); h=D.DomainHead(8)
            a=torch.optim.AdamW(m.parameters(),lr=5e-5); b=torch.optim.AdamW(h.parameters(),lr=1e-4)
            return m,h,a,b
        def step(objects,i):
            m,h,a,b=objects; torch.manual_seed(90+i); x=torch.randn(1,8)
            a.zero_grad(); b.zero_grad(); z=m(x)
            loss,_=D.domain_loss(h,z,'lld_synthetic' if i%2 else 'trackrad',D.schedule(i,.01))
            T.optimize(z.square().mean(),loss,list(m.parameters()),list(h.parameters()),a,b)
        original=create()
        for i in range(3): step(original,i)
        saved=copy.deepcopy([x.state_dict() for x in original])
        resumed=create()
        for obj,state in zip(resumed,saved): obj.load_state_dict(state)
        for i in range(3,6): step(original,i); step(resumed,i)
        for before,after in zip(original[:2],resumed[:2]):
            for k,v in before.state_dict().items(): self.assertTrue(torch.equal(v,after.state_dict()[k]))

    def test_sources_only_new_and_repairs(self):
        paths=P.sources()
        self.assertTrue(paths)
        self.assertFalse(any(p.startswith('cotracker-algorithm/') for p in paths))
        self.assertIn('diagnostics/lld-mixed-domain-v1/domain.py',paths)


if __name__=='__main__': unittest.main()
