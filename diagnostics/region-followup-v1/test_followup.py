import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import common as F
import evaluate as E
import diagnose as D
import train_g1 as G


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP'))
        self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_only_requested_endpoints(self):
        self.assertEqual(E.endpoint('G1',2)[1:], ('validation-10','region'))
        self.assertEqual(E.endpoint('D1',0)[1:], ('test-38','points'))
        with self.assertRaises(ValueError): E.endpoint('new_winner',0)
        self.assertFalse(F.RULE['g1_test'])
        self.assertFalse(F.RULE['diagnostic_selection'])

    def test_receipt_tamper(self):
        F.C.save(self.root/'followup-plan.json', {'test':1})
        p=self.root/'data.json'; F.C.save(p, {'value':2})
        marker=self.root/'done.json'; F.receipt(self.root,marker,[p])
        self.assertTrue(F.cached(self.root,marker)['complete'])
        F.C.save(p, {'value':3})
        with self.assertRaises(ValueError): F.cached(self.root,marker)

    def test_override_scoped_and_bound(self):
        F.C.save(self.root/'followup-plan.json', {'test':1})
        F.C.save(self.root/'g1/authorization.json',dict(explicit_override=True,
            followup_plan_sha256=F.C.sha(self.root/'followup-plan.json')))
        with patch.object(F,'checked'):
            F.authorize_g1(self.root,self.root/'prior',self.root/'g1','full','G1')
            for phase,arm in [('pilot','G1'),('full','D1'),('smoke','G1')]:
                with self.assertRaises(ValueError): F.authorize_g1(self.root,self.root/'prior',self.root/'g1',phase,arm)
            F.C.save(self.root/'followup-plan.json', {'test':2})
            with self.assertRaises(ValueError): F.authorize_g1(self.root,self.root/'prior',self.root/'g1','full','G1')

    def paired_fixture(self):
        rows=[dict(step=i,video='v',queries='q',sampling={'patient':'A'},domain='acdc',slot=0,draw=i,
            label_tracks='p',label_visibility='v',label_confidence='c',point=0.) for i in range(2000)]
        proof=dict(frozen_backbone_exact=True,initial_sidecar={'x':'y'},initial_head={'h':'z'},domain_counts={'a':1000})
        return rows,proof

    def test_paired_g1_not_just_counts(self):
        rows,proof=self.paired_fixture(); reference=copy.deepcopy(rows)
        F.compare_g1(rows,reference,proof,proof)
        for key,value in [('video','different'),('queries','different'),('label_visibility','different'),('point',.1)]:
            modified=copy.deepcopy(rows); modified[125][key]=value
            with self.assertRaises(ValueError): F.compare_g1(modified,reference,proof,proof)
        with self.assertRaises(ValueError): F.compare_g1(rows[:-1],reference,proof,proof)
        with self.assertRaises(ValueError): F.compare_g1(rows,reference,{**proof,'initial_head':{}},proof)

    def test_frame_diagnostics_query_excluded(self):
        gt=np.zeros((3,5,4)); output=np.zeros_like(gt)
        output[:,:,0]=1 # incorrect query must not count
        gt[1,2,1]=1; output[1,3,1]=1
        gt[1,2,2]=1 # empty prediction, nonempty GT
        rows=D.frame_rows(output,gt,0,'X','raw')
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]['CentroidDistancePixels'],1.)
        self.assertEqual(rows[1]['DiagnosticDice'],0.)
        self.assertIsNone(rows[1]['CentroidDistancePixels'])
        self.assertEqual(rows[2]['DiagnosticDice'],1.)
        self.assertIsNone(rows[2]['AreaRatio'])
        with self.assertRaises(ValueError): D.frame_rows(output,gt[:2],0,'X','raw')

    def test_percentile_matches_training_formula(self):
        a=np.arange(200,dtype=np.float32).reshape(4,5,10)
        lo,hi=np.percentile(a,[1,99]).astype(np.float32)
        np.testing.assert_array_equal(D.normalize(a),np.clip((a-lo)/(hi-lo),0,1)*np.float32(255))
        self.assertEqual(D.normalize(a).dtype,np.float32)
        with self.assertRaises(ValueError): D.normalize(np.ones((2,3,4),np.float32))

    def test_static_no_future_image_or_model(self):
        a=np.zeros((5,3,7),np.float32); q=np.zeros((5,3,1),np.uint8); q[2,1,0]=1
        out,stats=D.decode(None,None,a,q,None,'static')
        self.assertEqual(out.shape,a.shape)
        self.assertTrue(np.all(out[2,1]))
        self.assertEqual(stats,[])

    def test_shape_adapter_still_loads(self):
        fixed=E.inference_module()
        self.assertTrue(callable(fixed.run_algorithm))

    def test_decoder_shapes_and_zero_residual_prior_equivalence(self):
        import types
        import sys
        torch.set_num_threads(2)
        reshape=F.module('followup_test_reshape',F.C.APP/'resources/reshape.py').reshape_video
        resources=types.SimpleNamespace(reshape_video=reshape)
        experiments=types.SimpleNamespace(get_experiment_config=lambda: ('x',types.SimpleNamespace(lock_first_mask=True)))
        def feat(model,video):
            return torch.nn.functional.interpolate(video[:,:1],size=(48,64),mode='bilinear',align_corners=False).expand(-1,128,-1,-1)/255
        array=np.arange(7*9*5,dtype=np.float32).reshape(7,9,5)
        query=np.zeros((7,9,1),np.uint8); query[2:5,3:7]=1
        tracks=torch.tensor([[[180.,170.],[250.,220.]]]*5)
        head=D.RegionHead().eval()
        with patch.dict(sys.modules,resources=resources,experiments=experiments),patch.object(torch.Tensor,'cuda',lambda s:s),patch.object(D,'features',feat):
            raw,stats=D.decode(None,head,array,query,tracks,'raw')
            norm,_=D.decode(None,head,array,query,tracks,'percentile')
            prior,_=D.decode(None,head,array,query,tracks,'prior')
        self.assertEqual(raw.shape,array.shape)
        self.assertEqual(len(stats),5)
        self.assertTrue(all(x['ResidualRms']==0 for x in stats))
        np.testing.assert_array_equal(raw,prior)
        np.testing.assert_array_equal(norm,prior)
        np.testing.assert_array_equal(raw[:,:,0],query[:,:,0]>0)

    def test_g1_reuses_readonly_labels_and_restores_hooks(self):
        # A real cached label follows the exact original loader; no teacher call allowed.
        prior=self.root/'previous'; target=self.root/'g1'
        prior.mkdir(); target.mkdir()
        plan={'test':1}; F.C.save(prior/'manifest.json',plan)
        value={'tracks':torch.zeros(1,2,1,2),'visibility':torch.ones(1,2,1),'confidence':torch.ones(1,2,1)}
        sample={'video':torch.zeros(2,3,4,5),'queries':torch.zeros(1,1,3),'sampling':{'native_frames':[0,1]},'domain':'acdc','draw':100}
        binding=dict(video=G.T.S.tensor_sha(sample['video']),queries=G.T.S.tensor_sha(sample['queries']),
                     sampling=sample['sampling'],domain='acdc',slot=2,draw=100)
        path=prior/'labels/seed_0/00004.pth'; path.parent.mkdir(parents=True)
        torch.save(value,path)
        F.C.receipt(prior,path.with_suffix('.json'),[path],binding=binding)
        ref=[{} for _ in range(5)]
        ref[4]={k:G.T.S.tensor_sha(value[t]) for k,t in [('label_tracks','tracks'),('label_visibility','visibility'),('label_confidence','confidence')]}
        original=G.T.labels; oldgate=G.T.gate; oldreceipt=G.T.receipt
        class Repair:
            @staticmethod
            def short_row(x): return False
            @staticmethod
            def train_fixed(*args):
                actual,b=G.T.labels(target,plan,Path('/weights'),sample,0,4,{})
                torch.testing.assert_close(actual['tracks'],value['tracks'])
                self.assertEqual(b,binding)
                raise RuntimeError('test interrupted after cache read')
        before={p.relative_to(prior).as_posix():F.C.sha(p) for p in prior.rglob('*') if p.is_file()}
        with patch.object(G,'authorize_g1'),patch.object(G,'training_receipt',return_value=(prior,ref,{})),patch.object(G,'module',return_value=Repair):
            with self.assertRaisesRegex(RuntimeError,'test interrupted'):
                G.run(self.root,prior,Path('/w'),Path('/a'),Path('/x'),Path('/d'),0)
        self.assertIs(G.T.labels,original); self.assertIs(G.T.gate,oldgate); self.assertIs(G.T.receipt,oldreceipt)
        self.assertEqual(before,{p.relative_to(prior).as_posix():F.C.sha(p) for p in prior.rglob('*') if p.is_file()})

    def test_old_sources_not_inside_followup(self):
        self.assertFalse(any('region-followup-v1' in x for x in F.C.sources()))

    def test_official_report_uses_aggregate_and_pairs_not_frame_recompute(self):
        import csv
        F.C.save(self.root/'followup-plan.json',{'fixture':1})
        for arm in F.RULE['point_test']:
            for seed in ([0] if arm=='pretrained' else [0,1,2]):
                folder=self.root/E.endpoint(arm,seed)[0]
                n=['pretrained','M0','W1','D1'].index(arm)
                agg={v:.8+n*.01+seed*.001 for v in F.C.FIELDS.values()}
                agg['total_time']=10.
                F.C.save(folder/'metrics.json',dict(aggregates=agg,
                    results=[dict(case_id='A',**{v:.1 for v in F.C.FIELDS.values()})]))
                F.receipt(self.root,folder/'inference-verified.json',[],ids=['A'])
                F.receipt(self.root,folder/'metrics-verified.json',[folder/'metrics.json'])
        with patch.object(E,'checked',return_value={'ids':{'test-38':['A']}}):
            E.report(self.root,self.root/'previous','points')
        out=self.root/'reports/points/test-38'
        with (out/'summary.csv').open(encoding='utf-8-sig',newline='') as f: summary=list(csv.DictReader(f))
        self.assertAlmostEqual(float(summary[-1]['DSC_Mean']),.831)
        self.assertEqual(summary[-1]['N'],'3')
        with (out/'paired-deltas.csv').open(encoding='utf-8-sig',newline='') as f: deltas=list(csv.DictReader(f))
        d=next(x for x in deltas if x['Seed']=='2' and x['Candidate']=='D1/points' and x['Reference']=='W1/points')
        self.assertAlmostEqual(float(d['Delta_DSC']),.01)
        self.assertFalse((out/'selection.json').exists())

    def test_official_transport_bytes_verified(self):
        import base64
        F.C.save(self.root/'followup-plan.json',{'fixture':1})
        folder=self.root/'metrics'
        F.C.save(folder/'metrics.json',dict(aggregates={**{v:1. for v in F.C.FIELDS.values()},'total_time':2},
            results=[dict(case_id='A',**{v:1. for v in F.C.FIELDS.values()})]))
        F.receipt(self.root,folder/'inference-verified.json',[],ids=['A'])
        payload=base64.b64encode((folder/'metrics.json').read_bytes()).decode()
        (folder/'evaluation.log').write_text('REGION_METRICS_BASE64='+payload+'\n',encoding='utf-16')
        with patch.object(E,'checked'):
            E.record(self.root,self.root/'previous','metrics')
            (folder/'metrics.json').write_bytes((folder/'metrics.json').read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'byte mismatch'): E.record(self.root,self.root/'previous','metrics')


if __name__=='__main__': unittest.main()
