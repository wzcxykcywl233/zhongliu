import copy
from contextlib import nullcontext
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import binding as B
import runtime as R
import diagnose as D
import instrumented_train as IT
from engine import execute
from domain import configure, Tap, DomainHead, domain_loss, schedule
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar, configure_sidecar_runtime

torch.set_num_threads(2)


def model(arm='F010'):
    torch.manual_seed(34)
    m = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
    attach_sidecar(m, 'mamba'); configure_sidecar_runtime(m, backend='compact')
    # Nonzero branch exercises all of its parameter gradients, not only its output.
    with torch.no_grad():
        m.updateformer.feature_sidecar.output.weight.uniform_(-.001, .001)
    configure(m, arm)
    return m


def sample():
    torch.manual_seed(52)
    return dict(video=torch.rand(3, 3, 64, 64)*255,
                queries=torch.tensor([[[0.,20.,20.],[0.,30.,30.],[0.,40.,32.]]]),
                domain='trackrad', draw=123)


class RuntimeTests(unittest.TestCase):
    def test_real_forward_gradients_and_updated_weights_match(self):
        a = model(); b = copy.deepcopy(a); R.install(b)
        s = sample(); evidence = []
        for m in (a, b):
            tap = Tap(m,'fnet'); tap.reset(3)
            torch.manual_seed(93); head = DomainHead(m.latent_dim)
            params = [p for p in m.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=5e-5)
            out = m(video=s['video'][None],queries=s['queries'],iters=2,is_train=True)
            self.assertEqual(len(tap.values),1)
            target = dict(tracks=out[0].detach()+1,visibility=torch.ones_like(out[1]))
            task = B.BASE.T.S.training_loss(out,target)
            aux,_ = domain_loss(head,tap.embedding(),'trackrad',.01)
            # autograd.grad + retained graph is the exact smoke audit requirement.
            enc = [p for n,p in m.named_parameters() if n.startswith(B.P.ENCODER_PREFIXES)]
            self.assertGreater(B.BASE.gradient_norm(task,enc),0)
            self.assertGreater(B.BASE.gradient_norm(aux,enc),0)
            (task+aux).backward()
            self.assertEqual(len(tap.values),1, 'checkpoint must not duplicate domain hook captures')
            grads = {n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
            self.assertTrue(all(p.grad is None for p in m.parameters() if not p.requires_grad))
            opt.step()
            evidence.append(dict(outputs=[x.detach() for x in out[:3]],gradients=grads,
                classifier_gradients={n:p.grad.clone() for n,p in head.named_parameters()},
                student=copy.deepcopy(m.state_dict()),classifier=copy.deepcopy(head.state_dict())))
            tap.close()
        D.compare(*evidence)
        self.assertEqual(a.state_dict().keys(),b.state_dict().keys())

    def test_inference_and_parameter_names_unchanged(self):
        a = model(); b = copy.deepcopy(a); R.install(b); s=sample()
        self.assertEqual(a.state_dict().keys(),b.state_dict().keys())
        with torch.no_grad():
            x = a(video=s['video'][None],queries=s['queries'],iters=2)
            y = b(video=s['video'][None],queries=s['queries'],iters=2)
        for u,v in zip(x[:3],y[:3]): self.assertTrue(torch.equal(u,v))
        with self.assertRaises(ValueError): R.install(b)

    def test_saved_tensor_volume_reduced(self):
        s=sample(); values=[]
        # A three-query toy is dominated by the unchanged virtual-token branch;
        # use 33 queries here (remote probes keep all original 128 queries).
        s['queries']=s['queries'].repeat(1,11,1)
        for lowmem in (False,True):
            m=model('F0')
            if lowmem: R.install(m)
            # Deduplicate aliased storage and exclude weights: repeated saved
            # parameter references are not activation allocations.
            weights={p.untyped_storage().data_ptr() for p in m.parameters()}
            saved={}
            def pack(t):
                storage=t.untyped_storage()
                if storage.data_ptr() not in weights: saved[storage.data_ptr()]=storage.nbytes()
                return t
            with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):
                out=m(video=s['video'][None],queries=s['queries'],iters=2,is_train=True)
            values.append(sum(saved.values())); del out,m
        self.assertLess(values[1],values[0]*.85)

    def test_source_guard_fails_closed(self):
        def wrong(self,x): return x
        with self.assertRaises(ValueError): R.transformed_forward(wrong)
        class Bad(torch.nn.Module): pass
        with self.assertRaises(ValueError): R.install(Bad())

    def test_engine_matches_original_optimizer_math(self):
        s=sample()
        for arm in ('M0','S010','F0','Fprobe','F010'):
            original=model(arm); replacement=copy.deepcopy(original)
            packets=[]
            for use_engine,m in [(False,original),(True,replacement)]:
                side=[p for n,p in m.named_parameters() if n.startswith(B.P.PREFIX)]
                enc=[p for n,p in m.named_parameters() if n.startswith(B.P.ENCODER_PREFIXES) and p.requires_grad]
                params=side+enc; groups=[dict(params=side,lr=5e-5)]
                if enc: groups.append(dict(params=enc,lr=5e-6))
                opt=torch.optim.AdamW(groups,weight_decay=1e-5)
                kind,maximum=B.P.ARMS[arm]; head=tap=disc=None; classifier=[]
                if kind!='none':
                    torch.manual_seed(86); head=DomainHead(256 if kind=='sidecar' else m.latent_dim)
                    classifier=list(head.parameters()); disc=torch.optim.AdamW(classifier,lr=1e-4,weight_decay=1e-5)
                    tap=Tap(m,kind)
                target=dict(tracks=s['queries'][:,None,:,1:].expand(1,3,3,2)+1,
                            visibility=torch.ones(1,3,3))
                if use_engine:
                    class Phases:
                        def part(self,name): return nullcontext()
                    with patch.object(torch.Tensor,'cuda',lambda x:x):
                        execute(m,s,target,2,arm,params,classifier,opt,disc,head,tap,Phases(),True)
                else:
                    B.BASE.T.seed_all(s['draw']+47)
                    if tap: tap.reset(3)
                    out=m(video=s['video'][None],queries=s['queries'],iters=4,is_train=True)
                    task=B.BASE.T.S.training_loss(out,target); aux=task.new_zeros(())
                    if head: aux,_=domain_loss(head,tap.embedding(),s['domain'],schedule(2,maximum))
                    if enc: B.BASE.gradient_norm(task,enc)
                    if head and maximum>0: B.BASE.gradient_norm(aux,side if kind=='sidecar' else enc)
                    B.BASE.optimize(task,aux,params,classifier,opt,disc)
                packets.append(copy.deepcopy(m.state_dict()))
                if tap: tap.close()
            for name in packets[0]:
                self.assertTrue(torch.equal(packets[0][name],packets[1][name]),f'{arm}/{name}')

    def test_numerical_failure_is_not_accepted(self):
        a=dict(outputs=[torch.ones(2)],gradients={},classifier_gradients={},student={},classifier={})
        b=copy.deepcopy(a); b['outputs'][0][0]+=.01
        with self.assertRaises(AssertionError): D.compare(a,b)
        b=copy.deepcopy(a); b['gradients']['unexpected']=torch.ones(1)
        with self.assertRaises(ValueError): D.compare(a,b)

    def test_runtime_failure_cannot_be_skipped_by_resume(self):
        IT.validate_runtime_records([])
        IT.validate_runtime_records([{'seconds':1.},{'seconds':180.}])
        for value in (0.,-1.,180.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError): IT.validate_runtime_records([{'seconds':value}])

    def test_promotion_requires_memory_speed_and_equivalence(self):
        a=[dict(step=i,domain='trackrad' if i%2==0 else 'lld_synthetic',peak_allocated=100.,device_total=130.,measured_seconds=50.) for i in B.STEPS]
        b=[dict(r,peak_allocated=70.,measured_seconds=30.,equivalence={'outputs_max_abs_difference':0.}) for r in a]
        self.assertTrue(D.assess(a,b)['passed'])
        for field,value in [('peak_allocated',99.),('measured_seconds',181.),('equivalence',{})]:
            bad=copy.deepcopy(b); bad[2][field]=value
            self.assertFalse(D.assess(a,bad)['passed'])
        bad=copy.deepcopy(b); bad[0]['measured_seconds']=float('nan')
        with self.assertRaises(ValueError): D.assess(a,bad)

    def test_job_marker_and_payload_tampering(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root=Path(d); B.P.C.save(root/'manifest.json',{'a':1})
            folder=B.job(root,'original','F0'); B.P.C.save(folder/'steps.json',[])
            B.P.C.save(folder/'verified.json',dict(complete=True,manifest=B.P.C.sha(root/'manifest.json'),mode='original',arm='F0',files={'steps.json':B.P.C.sha(folder/'steps.json')}))
            self.assertTrue(B.verified_job(root,'original','F0'))
            B.P.C.save(folder/'steps.json',[1])
            with self.assertRaises(ValueError): B.verified_job(root,'original','F0')


if __name__=='__main__': unittest.main()
