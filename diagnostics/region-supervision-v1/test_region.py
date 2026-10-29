"""CPU tests use synthetic data only; not a substitute for the remote GPU smoke."""
import ast
import base64
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from core import *
import core
import data
import region
import reports
import train
torch.set_num_threads(1)

class TempTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP'))
        self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

class Contracts(TempTest):
    def test_python310_syntax(self):
        for p in PACKAGE.glob('*.py'): ast.parse(p.read_text(encoding='utf-8-sig'),feature_version=(3,10))

    def test_path_guard(self):
        for x in ['../x','/x','C:/x','a\\b','.']:
            with self.assertRaises(ValueError): safe(self.root,x)
        self.assertEqual(safe(self.root,'a/b'),self.root/'a/b')

    def test_frozen_and_receipt(self):
        save(self.root/'manifest.json',{'one':1}); freeze(self.root/'manifest.json',{'one':1})
        with self.assertRaises(ValueError): freeze(self.root/'manifest.json',{'one':2})
        save(self.root/'x.json',{'a':2}); receipt(self.root,self.root/'proof.json',[self.root/'x.json'])
        self.assertTrue(cached(self.root,self.root/'proof.json'))
        save(self.root/'x.json',{'a':3})
        with self.assertRaises(ValueError): cached(self.root,self.root/'proof.json')

    def test_inventory(self):
        p={'Dataset':[dict(Split='train-40',Case='A_001',File=r'C:\d\A_001\images\A_001_frames.mha',SHA256='A'*64)]}
        self.assertEqual(list(inventory(p,'train-40')),['A_001/images/A_001_frames.mha'])
        p['Dataset']*=2
        with self.assertRaises(ValueError): inventory(p,'train-40')

    def test_selection_guards(self):
        def r(d=.001,d98=0,hd=-.01): return dict(DSC=d,D98=d98,HD95=hd,MASD=0,CD=0)
        self.assertTrue(qualifies([r(),r(),r(-.0001)]))
        for rows in [[r(0)], [r(d98=-.003)], [r(hd=.03)], [r(),r(-.0001),r(-.0001)], [r(.02),r(.01),r(-.003)]]:
            self.assertFalse(qualifies(rows))

    def test_stage_and_g1_gate(self):
        train.gate(self.root,'smoke','G1')
        with self.assertRaises(ValueError): train.gate(self.root,'pilot','M0')
        with patch.object(train,'cached',return_value={'complete':True}):
            save(self.root/'reports/pilot/validation-10/selection.json',{'region_gate':False})
            with self.assertRaises(ValueError): train.gate(self.root,'full','G1')
            train.gate(self.root,'full','D1')
            save(self.root/'reports/pilot/validation-10/selection.json',{'region_gate':True})
            train.gate(self.root,'full','G1')

    def test_official_metrics_and_transport(self):
        obj=dict(results=[dict(case_id='A',**{v:.5 for v in FIELDS.values()})],
                 aggregates=dict(total_time=1,**{v:.5 for v in FIELDS.values()}))
        save(self.root/'manifest.json',{'x':1}); folder=self.root/'eval'
        save(folder/'metrics.json',obj); save(folder/'prediction.json',{})
        receipt(self.root,folder/'inference-verified.json',[folder/'prediction.json'],ids=['A'])
        raw=(folder/'metrics.json').read_bytes()
        (folder/'evaluation.log').write_text('REGION_METRICS_BASE64='+base64.b64encode(raw).decode()+'\n',encoding='utf-16')
        with patch.object(reports,'checked',return_value={}): reports.record_metrics(self.root,folder)
        self.assertTrue(cached(self.root,folder/'metrics-verified.json'))
        with self.assertRaises(ValueError): reports.official(folder/'metrics.json',['B'])
        obj['results'][0][FIELDS['DSC']]=float('nan'); save(folder/'metrics.json',obj)
        with self.assertRaises(ValueError): reports.official(folder/'metrics.json',['A'])

    def test_report_full_pairing_and_fixed_selection(self):
        save(self.root/'manifest.json',{'x':1})
        arms=['M0','W1','D1']; seeds=[0,1,2]
        for arm,view in [('pretrained','points'),('M0','points'),('W1','points'),('D1','points'),('D1','region')]:
            for seed in ([0] if arm=='pretrained' else seeds):
                folder=self.root/'full/validation-10'/f'seed_{seed}'/arm/view
                dsc=.90+{'pretrained':.005,'M0':0,'W1':.001,'D1':.002}[arm]+(0.001 if view=='region' else 0)
                values={v:(dsc if k=='DSC' else .9 if k=='D98' else 1.) for k,v in FIELDS.items()}
                save(folder/'metrics.json',dict(results=[dict(case_id='A',**values)],aggregates=dict(total_time=1.,**values)))
        with patch.object(reports,'checked',return_value={'ids':{'validation-10':['A']}}),patch.object(reports,'training_audit',return_value=(arms,seeds,[])),patch.object(reports,'cached',return_value={'ids':['A']}),patch.object(reports,'receipt'):
            reports.report(self.root,'full','validation-10')
            decision=read(self.root/'reports/full/validation-10/selection.json')
            self.assertEqual(decision['selected'],['D1','region']); self.assertTrue(decision['region_gate'])
            self.assertFalse(decision['selected_qualifies_vs_pretrained'])
            self.assertEqual(decision['test_outputs'],[['M0','points'],['W1','points'],['D1','points'],['D1','region']])
            reports.report(self.root,'full','validation-10')  # immutable repeat, no tuple/list mismatch

    def test_test_gate_stops_before_any_metrics(self):
        with patch.object(reports,'checked',return_value={}),patch.object(reports,'training_audit'),patch.object(reports,'cached',return_value=None):
            with self.assertRaises(ValueError): reports.test_gate(self.root)

    def test_audit_rejects_excluded_patient_even_if_hashes_match(self):
        save(self.root/'manifest.json',{'x':1})
        plan={'ids':{'train-40':['A']},'eligible':['patient001']}
        for arm in ARMS:
            folder=self.root/'smoke/train/seed_0'/arm
            rows=[dict(step=i,video='v',queries='q',sampling={'patient':'A' if i%2==0 else 'patient001'},
                       domain='trackrad' if i%2==0 else 'acdc',slot=i//2,draw=i,
                       label_tracks='p',label_visibility='v',label_confidence='c') for i in range(6)]
            save(folder/'steps.json',rows)
            save(folder/'proof.json',dict(frozen_backbone_exact=True,initial_sidecar={'a':'b'},
                initial_head={'h':'i'} if arm in ('D1','G1') else {},domain_counts={'acdc':3,'trackrad':3}))
            (folder/'final.pth').write_bytes(b'fixture-not-a-model')
            receipt(self.root,folder/'verified.json',[folder/'steps.json',folder/'proof.json',folder/'final.pth'],
                    phase='smoke',arm=arm,seed=0,steps=6,reload_gpu_forward=True)
        with patch.object(reports,'checked',return_value=plan):
            reports.training_audit(self.root,'smoke')
            folder=self.root/'smoke/train/seed_0/M0'
            rows=read(folder/'steps.json'); rows[1]['sampling']['patient']='patient076'; save(folder/'steps.json',rows)
            receipt(self.root,folder/'verified.json',[folder/'steps.json',folder/'proof.json',folder/'final.pth'],
                    phase='smoke',arm='M0',seed=0,steps=6,reload_gpu_forward=True)
            with self.assertRaisesRegex(ValueError,'split/patient'): reports.training_audit(self.root,'smoke')

class DataContract(TempTest):
    def sitk(self,lookup):
        return types.SimpleNamespace(ReadImage=lambda p:lookup[Path(p).name],GetArrayFromImage=lambda a:a)

    def test_sparse_native_endpoints(self):
        movie=np.random.default_rng(1).random((12,2,16,20)).astype('float32')
        ed=np.zeros((2,16,20),np.uint8); ed[0,4:10,4:10]=1
        es=np.roll(ed,2,axis=2)
        row=dict(patient='p',cine='training/p/p_4d.nii.gz',annotation_frames_zero_based={'ED':1,'ES':7})
        fake=self.sitk({'p_4d.nii.gz':movie,'p_frame02_gt.nii.gz':ed,'p_frame08_gt.nii.gz':es})
        with patch.dict(sys.modules,{'SimpleITK':fake}):
            s=data.external_sample(row,self.root,42); reverse=data.external_sample(row,self.root,43)
        self.assertEqual(s['sampling']['native_frames'],list(range(1,8)))
        self.assertEqual(reverse['sampling']['native_frames'],list(range(7,0,-1)))
        self.assertEqual(s['annotated'].tolist(),[True,False,False,False,False,False,True])
        self.assertEqual(s['video'].shape,(7,3,384,512)); self.assertEqual(s['video'].dtype,torch.float32)
        self.assertEqual(float(s['masks'][1:-1].sum()),0)
        self.assertTrue(torch.isfinite(s['video']).all()); self.assertLessEqual(float(s['video'].max()),255)

    def test_trackrad_orientation(self):
        # W,H,T, horizontal ramp with an asymmetric target; no transpose ambiguity.
        w,h,t=20,16,4
        image=np.broadcast_to(np.arange(w,dtype='float32')[:,None,None],(w,h,t)).copy()
        masks=np.zeros((w,h,t),np.uint8); masks[4:10,3:9]=1
        fake=self.sitk({'A_frames.mha':image,'A_labels.mha':masks})
        with patch.dict(sys.modules,{'SimpleITK':fake}): s=data.trackrad_sample('A',self.root,1)
        self.assertEqual(s['sampling']['native_frames'],[0,1,2,3])
        self.assertTrue(torch.allclose(s['video'][0,0,10],s['video'][0,0,100],atol=1e-4))
        self.assertGreater(float(s['video'][0,0,100,-1]),float(s['video'][0,0,100,0]))

    def test_mixed_schedule_excludes_bad_cases(self):
        plan={'ids':{'train-40':['A','B']},'eligible':['patient001','patient002']}
        seen=[]
        def fake(name,root,draw):
            if isinstance(name,dict): name=name['patient']
            seen.append(name)
            masks=torch.zeros(3,1,24,24); masks[:,:,5:19,5:19]=1
            return dict(video=torch.rand(3,3,24,24),masks=masks,annotated=torch.ones(3,dtype=torch.bool),sampling={'patient':name})
        audit={'patients':[{'patient':n} for n in plan['eligible']]}
        # Scheduling test does not require local skimage; real contour extraction is GPU-smoke tested.
        sampler=types.SimpleNamespace(convert_mask_to_points=lambda mask,n_border_points:torch.zeros(1,n_border_points,3))
        with patch.object(data,'external_sample',side_effect=fake),patch.object(data,'trackrad_sample',side_effect=fake),patch.dict(sys.modules,{'resources.seg_to_tap':sampler}):
            rows=[data.sample(plan,audit,self.root,self.root,0,i) for i in range(6)]
        self.assertEqual([r['domain'] for r in rows],['trackrad','acdc']*3)
        self.assertFalse(set(seen)&{'patient076','patient082','patient074'})
        self.assertEqual(rows[0]['queries'].shape,(1,128,3))

class RegionLosses(unittest.TestCase):
    def masks(self):
        m=torch.zeros(3,1,24,32); m[:,:,5:17,8:22]=1
        return m

    def test_missing_and_query_mask_not_targets(self):
        m=self.masks(); ann=torch.tensor([True,False,True]); x=torch.randn(3,1,12,16,requires_grad=True)
        a=region.region_loss(x,m,ann); m2=m.clone(); m2[0]=1-m[0]; m2[1]=1-m[1]
        self.assertTrue(torch.equal(a,region.region_loss(x,m2,ann)))
        a.backward(); self.assertEqual(float(x.grad[:2].abs().sum()),0)
        self.assertGreater(float(x.grad[2].abs().sum()),0)

    def test_boundary_gradient_and_gating(self):
        masks=self.masks(); tracks=torch.tensor([[[10.,10.]],[[10.,10.]],[[40.,10.]]],requires_grad=True)
        ann=torch.tensor([True,False,True]); vis=torch.ones(3,1,dtype=torch.bool)
        loss=region.boundary_loss(tracks,masks,ann,vis); loss.backward()
        self.assertGreater(float(tracks.grad[2,0,0]),0)
        self.assertEqual(float(tracks.grad[:2].abs().sum()),0)
        vis[2]=False
        self.assertEqual(float(region.boundary_loss(tracks,masks,ann,vis).detach()),0)

    def test_translation_prior_and_chunk_reference(self):
        first=self.masks()[0]; q=torch.tensor([[8.,5.],[21.,5.],[21.,16.],[8.,16.]])
        tracks=torch.stack((q,q+torch.tensor([4.,0.])))
        prior,_=region.prior_and_heat(first,tracks,(24,32))
        chunk,_=region.prior_and_heat(first,tracks[1:],(24,32),q)
        self.assertTrue(torch.allclose(prior[1:],chunk))
        self.assertLess(float((prior[1,0,:,4:]-first[0,:,:-4]).abs().max()),1e-5)

    def test_head_and_track_gradients(self):
        head=region.RegionHead(); first=self.masks()[0]; feat=torch.randn(3,128,6,8)
        tracks=torch.tensor([[[8.,5.],[21.,5.],[21.,16.],[8.,16.]]]*3,requires_grad=True)
        out=head(feat,first,tracks,feat[:1]); loss=region.region_loss(out,self.masks(),torch.ones(3,dtype=torch.bool))
        loss.backward()
        self.assertTrue(torch.isfinite(tracks.grad).all())
        self.assertGreater(float(tracks.grad[1:].abs().sum()),0)
        self.assertGreater(float(head.decode[-1].weight.grad.abs().sum()),0)

    def test_real_backbone_gradient_paths(self):
        from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
        from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar,configure_sidecar_runtime,set_sidecar_training_mode
        torch.manual_seed(5)
        model=CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60)
        attach_sidecar(model,'mamba'); configure_sidecar_runtime(model,backend='compact'); set_sidecar_training_mode(model)
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        video=torch.rand(1,2,3,64,64)*255; q=torch.tensor([[[0.,20.,20.],[0.,40.,20.],[0.,40.,40.],[0.,20.,40.]]])
        out=model(video=video,queries=q,iters=2,is_train=True)
        head=region.RegionHead(); feat=train.features(model,video[0])
        mask=torch.zeros(2,1,64,64); mask[:,:,20:42,20:42]=1
        logits=head(feat,mask[0],out[0][0],feat[:1],q[0,:,1:])
        loss=region.region_loss(logits,mask,torch.ones(2,dtype=torch.bool))
        loss.backward()
        grads=[n for n,p in model.named_parameters() if p.grad is not None and p.grad.abs().sum()>0]
        self.assertTrue(grads); self.assertTrue(all(n.startswith(train.PREFIX) for n in grads))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
        optimizer.step()
        self.assertTrue(all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if n in frozen))

if __name__=='__main__': unittest.main()
