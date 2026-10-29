import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import worker as fix
import train
from cotracker.models.core.cotracker.cotracker import CoTracker2
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline

torch.set_num_threads(1)


def model(slot):
    # Exercise the actual forward/window scheduling, with cheap deterministic feature/window stubs.
    cls = (CoTracker2, CoTrackerThreeOnline)[slot]
    m = cls.__new__(cls)
    torch.nn.Module.__init__(m)
    m.stride=4; m.window_len=16; m.latent_dim=128; m.corr_levels=1; m.corr_radius=0
    m.fnet=lambda x: torch.ones(x.shape[0],128,x.shape[-2]//4,x.shape[-1]//4)
    def feat(f,t,q,**kw):
        value=torch.ones(1,1,q.shape[1],128)
        return value if slot==0 else (value,value)
    m.get_track_feat=feat
    m.calls=[]
    def window(**kw):
        m.calls.append(kw['coords'].shape[1])
        p=kw['coords']*4+.125
        v=torch.full(p.shape[:-1],10.)
        return ([p],v) if slot==0 else ([p],[v],[v-.5])
    m.forward_window=window
    return m.eval().requires_grad_(False)


def inputs(t):
    v=torch.arange(t*3*16*16,dtype=torch.float32).reshape(1,t,3,16,16)%255
    q=torch.tensor([[[0.,4.,5.],[0.,8.,9.]]])
    return v,q


def teacher(slot,kind,m,v,q,draw,arm,pool):
    out=m(v,q,iters=4)
    return out[0],out[1],torch.ones_like(out[1]) if slot==0 else out[2],{'unchanged':True}


class ShortTeacherTests(unittest.TestCase):
    def test_failed_step_is_on3(self):
        draw=fix.core.RULE['seed_base']+61*101
        self.assertEqual(draw,20267190)
        self.assertEqual(int(np.random.default_rng(draw+41).integers(3)),1)

    def test_actual_old_forward_skips_short_windows_both_families(self):
        for slot in (0,1):
            for t in (2,6,8):
                m=model(slot); v,q=inputs(t); p,vis,*_=m(v,q)
                self.assertEqual(m.calls,[])
                self.assertTrue(bool((p==0).all()))
                self.assertEqual(vis.unique().tolist(),[.5])

    def test_fixed_native_shapes_and_single_window_both_families(self):
        for slot in (0,1):
            for t in (2,6,8):
                m=model(slot); v,q=inputs(t); before_v=v.clone(); before_q=q.clone()
                old=vars(m).copy()
                p,vis,c,audit=fix.teacher_call(teacher,slot,'unused',m,v,q,None,'control',None)
                self.assertEqual(m.calls,[16]); self.assertEqual(p.shape,(1,t,2,2))
                self.assertTrue(bool((vis>.9).all()))
                self.assertEqual(audit['short_window_repair']['native_output_frames'],t)
                self.assertNotIn('forward',vars(m)); self.assertIs(m.forward_window,old['forward_window'])
                self.assertTrue(torch.equal(v,before_v)); self.assertTrue(torch.equal(q,before_q))
                self.assertTrue(torch.equal(p[0,0],q[0,:,1:]+.125))

    def test_long_sequences_identical_even_with_fixed_forward(self):
        for slot in (0,1):
            for t in (9,12,16,17,31,47,60):
                a,b=model(slot),model(slot); v,q=inputs(t)
                expected=a(v,q)
                actual=fix.fixed_forward(b,slot)(v,q)
                for x,y in zip(expected[:2],actual[:2]): self.assertTrue(torch.equal(x,y))
                self.assertEqual(a.calls,b.calls)
                with patch.object(fix,'fixed_forward',side_effect=AssertionError('must bypass')):
                    fix.teacher_call(teacher,slot,'unused',b,v,q,None,'control',None)

    def test_real_untrained_network_matches_its_existing_internal_padding(self):
        # Real encoders/attention/update layers, CPU random weights; NOT an accuracy test.
        for slot,cls in enumerate((CoTracker2,CoTrackerThreeOnline)):
            torch.manual_seed(7)
            m=cls(window_len=16,stride=4,model_resolution=(64,64)).eval().requires_grad_(False)
            v=torch.rand(1,6,3,64,64)*255
            q=torch.tensor([[[0.,24.,25.],[0.,40.,39.]]])
            padded=torch.cat((v,v[:,-1:].expand(-1,10,-1,-1,-1)),1)
            with torch.no_grad():
                expected=m(padded,q,iters=1)
                actual=fix.fixed_forward(m,slot)(v,q,iters=1)
            count=2 if slot==0 else 3
            for i,(a,b) in enumerate(zip(actual[:count],expected[:count])):
                self.assertTrue(bool(torch.isfinite(a).all()))
                # Sigmoid on differently sized buffers may differ by one float32 ULP.
                self.assertTrue(torch.equal(a,b[:,:6]) if i==0 else torch.allclose(a,b[:,:6],atol=1e-7,rtol=1e-6))
            self.assertEqual(actual[0].shape,(1,6,2,2))

    def test_actual_protocol_accepts_native_short_teacher(self):
        protocol=train.S.load_protocol(fix.core.CHECKOUT)
        for slot in (0,1):
            m=model(slot); v,q=inputs(6); support=torch.zeros(1,25,3)
            out=fix.teacher_call(protocol.teacher_forward,slot,train.S.TEACHER_TYPES[slot],m,v,q,support,'control',protocol.baseline())
            self.assertEqual(out[0].shape,(1,6,2,2))
            self.assertEqual(out[3]['query_time_counts'],[2,0,0,0,0,0])
            self.assertEqual(out[3]['visible_gt09_fraction'],1.)

    def test_offline_and_long_do_not_change_output_or_call(self):
        token=object()
        m=types.SimpleNamespace(window_len=60)
        with patch.object(fix,'fixed_forward',side_effect=AssertionError('no patch')):
            self.assertIs(fix.teacher_call(lambda *a:token,2,'offline',m,*inputs(6),None,None,None),token)

    def test_hooks_restored_on_failure(self):
        for slot in (0,1):
            m=model(slot); before=m.forward_window
            with self.assertRaisesRegex(RuntimeError,'probe failure'):
                fix.teacher_call(lambda *a: (_ for _ in ()).throw(RuntimeError('probe failure')),
                    slot,'unused',m,*inputs(6),None,None,None)
            self.assertNotIn('forward',vars(m)); self.assertIs(m.forward_window,before)

    def test_bad_model_source_and_single_frame_fail_closed(self):
        m=model(1)
        with patch.dict(fix.TEACHER_SOURCE,{'cotracker3_online.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'version bound'): fix.fixed_forward(m,1)
        with self.assertRaisesRegex(ValueError,'at least two'):
            fix.teacher_call(teacher,1,'unused',m,*inputs(1),None,None,None)
        m.train()
        with self.assertRaisesRegex(ValueError,'frozen teacher'):
            fix.teacher_call(teacher,1,'unused',m,*inputs(6),None,None,None)

    def test_short_output_validation(self):
        def zero(*a):
            v=a[4]; return torch.zeros(1,6,2,2),torch.ones(1,6,2),torch.ones(1,6,2),{}
        with self.assertRaisesRegex(ValueError,'execute one window'):
            fix.teacher_call(zero,1,'unused',model(1),*inputs(6),None,None,None)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP'))
        self.root=Path(self.temp.name)
        fix.core.save(self.root/'manifest.json',{'frozen':'original'})
        self.guard=patch.object(fix.core,'checked',return_value={}); self.guard.start()
        self.binding=fix.bind(self.root)
    def tearDown(self):
        self.guard.stop(); self.temp.cleanup()

    def test_binding_preserves_original_and_rejects_changed_source(self):
        original=(self.root/'manifest.json').read_bytes()
        self.assertEqual(fix.bind(self.root),self.binding)
        self.assertEqual((self.root/'manifest.json').read_bytes(),original)
        value=fix.core.read(self.binding); value['before']='tampered'; fix.core.save(self.binding,value)
        with self.assertRaisesRegex(ValueError,'Frozen artifact changed'): fix.bind(self.root)

    def test_resume_compatibility_only_for_unaffected_legacy_steps(self):
        old={'records':[{'slot':1,'sampling':{'native_frames':list(range(60))}}]}
        fix.validate_resume(old,self.binding)
        short=copy.deepcopy(old); short['records'][0]['sampling']['native_frames']=list(range(6))
        with self.assertRaisesRegex(ValueError,'unverified short'): fix.validate_resume(short,self.binding)
        short['short_teacher_repair_sha256']=fix.core.sha(self.binding)
        fix.validate_resume(short,self.binding)
        short['short_teacher_repair_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'another teacher repair'): fix.validate_resume(short,self.binding)

    def test_train_wiring_preserves_legacy_and_binds_new_artifacts(self):
        originals=(train.labels,train.S.load_protocol,train.receipt,train.S.atomic_torch)
        def fake_train(*args):
            folder=self.root/'pilot/train/seed_0/M0'
            train.S.atomic_torch(folder/'resume.pth',{'records':[]})
            train.receipt(self.root,folder/'verified.json',[folder/'resume.pth'],steps=0)
        with patch.object(train,'train',side_effect=fake_train):
            fix.train_fixed(self.root,None,None,None,None,'pilot','M0',0,180)
        folder=self.root/'pilot/train/seed_0/M0'
        saved=torch.load(folder/'resume.pth',weights_only=True)
        self.assertEqual(saved['short_teacher_repair_sha256'],fix.core.sha(self.binding))
        self.assertIn(fix.BINDING,fix.core.cached(self.root,folder/'verified.json')['files'])
        self.assertEqual(originals,(train.labels,train.S.load_protocol,train.receipt,train.S.atomic_torch))

    def test_failed_label_keeps_threshold_and_records_details(self):
        originals=(train.labels,train.S.load_protocol,train.receipt,train.S.atomic_torch)
        sample=dict(draw=20267190,domain='acdc',sampling={'patient':'patient092','native_frames':list(range(6))},
            video=inputs(6)[0][0])
        protocol=types.SimpleNamespace(teacher_forward=lambda *args: (
            torch.ones(1,6,2,2),torch.full((1,6,2),.8),torch.ones(1,6,2),{}))
        def bad_labels(*args):
            # Model path bypassed here: verify the original policy's low-visibility rejection is retained.
            p=train.S.load_protocol(None)
            p.teacher_forward(2,'offline',types.SimpleNamespace(window_len=60),*inputs(6),None,None,None)
            raise ValueError('Invalid teacher labels')
        def fake_train(*args): train.labels(self.root,{},None,sample,0,61,{})
        with patch.object(train,'labels',side_effect=bad_labels), patch.object(train.S,'load_protocol',return_value=protocol), patch.object(train,'train',side_effect=fake_train):
            with self.assertRaisesRegex(ValueError,'Invalid teacher labels'):
                fix.train_fixed(self.root,None,None,None,None,'pilot','M0',0,180)
        report=fix.core.read(self.root/'label-failures/seed_0_00061.json')
        self.assertEqual(report['teacher_slot'],1)
        self.assertEqual(report['native_frames'],6)
        self.assertEqual(report['output_diagnostic']['visible_gt09_count'],0)
        self.assertEqual(originals,(train.labels,train.S.load_protocol,train.receipt,train.S.atomic_torch))

    def test_legacy_short_label_cache_cannot_be_silently_reused(self):
        path=self.root/'labels/seed_0/00061.pth'
        train.S.atomic_torch(path,{'tracks':torch.zeros(1,6,2,2)})
        fix.core.receipt(self.root,path.with_suffix('.json'),[path],
            binding={'slot':1,'sampling':{'native_frames':list(range(6))}})
        sample=dict(draw=20267190)
        def fake_train(*args): train.labels(self.root,{},None,sample,0,61,{})
        with patch.object(train,'train',side_effect=fake_train):
            with self.assertRaisesRegex(ValueError,'lack repair provenance'):
                fix.train_fixed(self.root,None,None,None,None,'pilot','M0',0,180)


if __name__=='__main__': unittest.main()
