from dataclasses import fields, replace
import json
from pathlib import Path
import unittest
import torch
from experiments import EXPERIMENTS, ROTATION_BACKCHECK_EXPERIMENTS
from rotation_backcheck import rotate,crop,decide,score_rotation_frames
from test_forward_pass import resource_model
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline


class RotationTests(unittest.TestCase):
    def test_registry_and_exclusions(self):
        off=EXPERIMENTS['rotation_control_i2']
        on=EXPERIMENTS['rotation_history_i2']
        self.assertEqual(off,EXPERIMENTS['memory_control'])
        self.assertEqual([f.name for f in fields(off) if getattr(off,f.name)!=getattr(on,f.name)],
                         ['frame_backcheck','rotation_backcheck'])
        for args in ({'n_iterations':4},{'frame_backcheck':False},{'frame_backcheck_fourway':True}):
            with self.assertRaises(ValueError): replace(on,**args)
        manifest=json.loads((Path(__file__).resolve().parents[1]/'experiments'/'rotation-backcheck-40-10-38.json').read_text())
        self.assertEqual(set(manifest['profiles']),set(ROTATION_BACKCHECK_EXPERIMENTS))

    def test_known_rotation_sign_and_inverse(self):
        delta=torch.tensor([[2.,1.]])
        torch.testing.assert_close(rotate(delta,90),torch.tensor([[-1.,2.]]))
        torch.testing.assert_close(rotate(rotate(delta,20),-20),delta)
        image=torch.arange(64*64).reshape(1,64,64).float()
        centers=torch.tensor([[31.5,31.5]])
        zero,valid=crop(image,centers,0)
        self.assertTrue(valid.all())
        torch.testing.assert_close(zero[0],image,atol=.001,rtol=1e-6)
        clockwise,_=crop(image,centers,90)
        torch.testing.assert_close(clockwise[0],torch.rot90(image,-1,(-2,-1)),atol=.001,rtol=1e-6)

    def test_peak_gate_and_zero_fallback(self):
        scores=torch.tensor([[.1,.2,.3,.8,.2],[.8,.79,.3,.1,.2],[.1,.2,.8,.3,.4]])
        chosen,accept,_,_=decide(scores)
        self.assertEqual(chosen.tolist(),[3,2,2])
        self.assertEqual(accept.tolist(),[True,False,False])

    def test_patch_boundary_reports_missing(self):
        image=torch.ones(1,96,96)
        _,valid=crop(image,torch.tensor([[48.,48.],[0.,0.]]),20)
        self.assertEqual(valid.tolist(),[True,False])

    def test_real_encoder_and_hierarchy_do_not_change_predictions(self):
        old=torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
                with self.subTest(device=device),torch.inference_mode():
                    torch.manual_seed(4)
                    model=CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60).to(device).eval()
                    video=torch.rand(1,3,1,96,96,device=device)*255
                    queries=torch.tensor([[[0.,47.,47.],[0.,49.,49.]]],device=device)
                    args=dict(span=2,support_grid_size=0,n_iterations=2,device=device)
                    original=resource_model.hierarchical_forward_pass(model,video,queries,**args)
                    rows=[]
                    observed=resource_model.hierarchical_forward_pass(model,video,queries,
                        frame_backcheck_rows=rows,rotation_backcheck=True,**args)
                    torch.testing.assert_close(original.trajectories,observed.trajectories,rtol=0,atol=0)
                    torch.testing.assert_close(original.visibility,observed.visibility,rtol=0,atol=0)
                    self.assertEqual([r['frame'] for r in rows],[1,2])
                    self.assertTrue(all('patch_aligned' in r for r in rows))
        finally:
            torch.set_num_threads(old)


if __name__=='__main__': unittest.main()
