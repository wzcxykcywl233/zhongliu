import hashlib
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import infer_fixed as fix
torch.set_num_threads(1)

def load_reshape():
    path=fix.core.APP/'resources/reshape.py'
    spec=importlib.util.spec_from_file_location('real_deployment_reshape',path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.reshape_video

class OutputShape(unittest.TestCase):
    def test_chunked_native_resize_including_239_frames(self):
        reshape=load_reshape()
        for t in (1,2,5,239):
            x=torch.rand(t,1,5,9)>.5
            y=fix.join_region_masks(list(x.split(4)))
            self.assertEqual(y.shape,(1,t,5,9)); self.assertEqual(y.dtype,torch.bool)
            self.assertTrue(torch.equal(y[0],x[:,0]))
            self.assertEqual(reshape(y,(7,11)).shape,(1,t,7,11))
            self.assertEqual(reshape(x[None],(7,11)).shape,(1,t,1,7,11))

    def test_reject_invalid_channel_and_dtype(self):
        for chunks in ([],[torch.ones(1,2,3,4,dtype=torch.bool)],[torch.ones(1,1,3,4)],
                       [torch.ones(0,1,3,4,dtype=torch.bool)],
                       [torch.ones(1,1,3,4,dtype=torch.bool),torch.ones(1,1,4,4,dtype=torch.bool)]):
            with self.assertRaises(ValueError): fix.join_region_masks(chunks)

    def resources(self):
        r=types.ModuleType('resources')
        r.reshape_video=load_reshape()
        r.setup_model=lambda **kwargs:None
        r.convert_mask_to_points=lambda **kwargs:torch.tensor([[[0.,100.,100.],[0.,200.,100.],[0.,200.,200.],[0.,100.,200.]]])
        def track(**kw):
            q=kw['queries']; t=kw['video'].shape[1]
            return types.SimpleNamespace(trajectories=q[:,:,1:][:,None].expand(-1,t,-1,-1).clone(),
                visibility=torch.ones(1,t,4),confidence=torch.ones(1,t,4))
        r.hierarchical_forward_pass=track
        r.convert_point_trajectory_to_mask_sequence=lambda p,video_shape,**kw:torch.zeros(1,p.shape[1],*video_shape,dtype=torch.bool)
        return r

    def integration(self,fixed,view,t=5,raise_in_head=False,lock=False):
        m=fix.load_fixed()
        if not fixed:
            # Restore precisely the original broken return for a negative-control regression.
            text=(fix.ORIGINAL/'infer.py').read_text(encoding='utf-8-sig')
            exec(compile(text,str(fix.ORIGINAL/'infer.py'),'exec'),m.__dict__)
        r=self.resources(); setup=r.setup_model; convert=r.convert_point_trajectory_to_mask_sequence
        x=np.random.default_rng(1).random((11,7,t)).astype('float32')*255
        q=np.zeros((11,7,1),np.uint8); q[2:8,2:5]=1
        def head(feat,first,*args):
            if raise_in_head: raise RuntimeError('synthetic head failure')
            return first[None].expand(len(feat),-1,-1,-1)*4-2
        m.features=lambda model,video:torch.zeros(len(video),128,2,2)
        import experiments
        name,config=experiments.get_experiment_config(fix.core.PROFILE)
        # Also exercise the optional query-mask lock without changing the frozen real profile.
        config=replace(config,lock_first_mask=lock)
        with patch.dict(sys.modules,{'resources':r}),patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self),patch.dict(os.environ,{'TRACKRAD_DIAGNOSTICS':'0','COTRACKER_EXPERIMENT':'baseline'}),patch.object(experiments,'get_experiment_config',return_value=(name,config)):
            try:
                y=m.run_algorithm(torch.nn.Linear(1,1),head,x,q,
                    dict(frame_rate=4.,magnetic_field_strength=1.5,scanned_region='lung'),view)
            finally:
                self.assertIs(r.setup_model,setup); self.assertIs(r.convert_point_trajectory_to_mask_sequence,convert)
                self.assertEqual(os.environ['COTRACKER_EXPERIMENT'],'baseline')
        self.assertEqual(y.shape,x.shape); self.assertEqual(y.dtype,np.bool_)
        if lock: self.assertTrue(np.array_equal(y[:,:,:1],q))
        return y

    def test_real_deployment_chain_region_and_query_lock(self):
        for t in (1,5):
            for lock in (False,True): self.integration(True,'region',t,lock=lock)

    def test_real_deployment_chain_reproduces_old_failure(self):
        with self.assertRaisesRegex(AssertionError,'torch.Size'):
            self.integration(False,'region')

    def test_point_pipeline_unchanged(self):
        self.assertTrue(np.array_equal(self.integration(True,'points'),self.integration(False,'points')))

    def test_hooks_restored_after_head_failure(self):
        with self.assertRaisesRegex(RuntimeError,'synthetic head failure'):
            self.integration(True,'region',raise_in_head=True)

    def test_original_version_is_bound(self):
        self.assertEqual(hashlib.sha256((fix.ORIGINAL/'infer.py').read_text(encoding='utf-8-sig').encode()).hexdigest(),fix.ORIGINAL_INFER_SHA256)
        with patch.object(fix,'ORIGINAL_INFER_SHA256','0'*64):
            with self.assertRaises(ValueError): fix.load_fixed()

    def test_repair_binding_does_not_touch_old_manifest(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root=Path(d); fix.core.save(root/'manifest.json',{'original':1}); before=(root/'manifest.json').read_bytes()
            with patch.object(fix.core,'checked',return_value={}):
                p=fix.bind(root); fix.bind(root)
                self.assertEqual((root/'manifest.json').read_bytes(),before)
                record=fix.core.read(p); record['source']['infer_fixed.py']='0'*64; fix.core.save(p,record)
                with self.assertRaises(ValueError): fix.bind(root)

    def test_point_receipt_function_stays_original(self):
        m=fix.load_fixed(); original=m.receipt
        with patch.object(fix,'load_fixed',return_value=m),patch.object(fix,'bind',return_value=Path('/study')/fix.BINDING),patch.object(m,'infer') as infer:
            fix.run(Path('/study'),Path('/data'),Path('/weights'),'smoke','validation-10',0,'M0','points',True)
            self.assertIs(m.receipt,original); infer.assert_called_once()

    def test_region_receipt_binds_repair_and_rejects_legacy_cache(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root=Path(d); fix.core.save(root/'manifest.json',{'x':1}); fix.core.save(root/fix.BINDING,{'repair':1})
            folder=root/'smoke/validation-10/seed_0/D1/region'
            artifact=folder/'jobs/A/prediction.json'; marker=artifact.parent/'verified.json'
            fix.core.save(artifact,{'a':1})
            m=fix.load_fixed()
            def fake_infer(*args): m.receipt(root,marker,[artifact],case='A')
            with patch.object(fix,'load_fixed',return_value=m),patch.object(fix,'bind',return_value=root/fix.BINDING),patch.object(m,'infer',side_effect=fake_infer):
                fix.run(root,Path('/data'),Path('/weights'),'smoke','validation-10',0,'D1','region',True)
            self.assertEqual(fix.core.cached(root,marker)['files'][fix.BINDING],fix.core.sha(root/fix.BINDING))
            # Well-hashed old region cache still cannot be silently reused without the repair link.
            fix.core.receipt(root,marker,[artifact],case='A')
            with patch.object(fix,'bind',return_value=root/fix.BINDING):
                with self.assertRaisesRegex(ValueError,'repair provenance'):
                    fix.run(root,Path('/data'),Path('/weights'),'smoke','validation-10',0,'D1','region',True)

if __name__=='__main__': unittest.main()
