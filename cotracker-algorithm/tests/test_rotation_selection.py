import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
spec=importlib.util.spec_from_file_location('rotation_selection_test',ROOT/'experiments'/'diagnose_rotation_selection.py')
diagnostic=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class SyntheticEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor=torch.nn.Parameter(torch.zeros(1))
        self.stride=4
        self.fnet=torch.nn.AvgPool2d(4)


class SelectionTests(unittest.TestCase):
    def test_real_log_audit_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            out=root/'audit'
            out.mkdir()
            for case in range(38):
                for profile in ('rotation_history_i2','rotation_control_i2'):
                    job=root/profile/'checkpoint'/'jobs'/f'C_{case:03d}'
                    image=job/'output'/'images'/'mri-linac-series-targets'/'output.mha'
                    image.parent.mkdir(parents=True)
                    image.write_bytes(b'fixture output')
                    data=dict(profile=profile,prediction=dict(array_sha256='fixture',
                        output_file_sha256=diagnostic.sha256(image)),frame_backcheck=[dict(
                            frame=1,supported_points=10,angle_accepted_points=2,
                            angle_fallback_points=8,patch_zero=.8,patch_aligned=.82,
                            **{'angle_-20_points':1,'angle_20_points':1})])
                    (job/'diagnostics.json').write_text(json.dumps(data),encoding='utf-8')
            result=diagnostic.audit_logs(root,out)
            self.assertEqual(result[0]['Accepted'],76)
            self.assertAlmostEqual(result[0]['AcceptedMeanGain'],.1)
            image.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                diagnostic.audit_logs(root,out)

    def test_reconstruct_accepted_gain(self):
        row=dict(supported_points=10,angle_accepted_points=2,angle_fallback_points=8,
                 patch_aligned=.82,patch_zero=.8)
        total,accepted,supported=diagnostic.accepted_gain(row)
        self.assertAlmostEqual(total/accepted,.1)
        self.assertEqual(supported,10)
        row.update(angle_accepted_points=0,angle_fallback_points=10)
        with self.assertRaises(ValueError):diagnostic.accepted_gain(row)
        row['patch_aligned']=.8
        self.assertEqual(diagnostic.accepted_gain(row),(0.,0,10))

    def test_synthetic_grid_offgrid_bounds_and_resume(self):
        old=torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory)
                model=SyntheticEncoder().eval()
                summaries=diagnostic.synthetic(model,out)
                trials=list(out.glob('synthetic-*.json'))
                self.assertEqual(len(trials),12)
                self.assertEqual(sum(len(diagnostic.read_json(p)) for p in trials),324)
                self.assertEqual(len(summaries),9)
                for row in summaries:
                    self.assertIn(row['Scope'],('in_range_grid','in_range_offgrid','outside_range'))
                initial=(out/'synthetic-angle-trials.csv').read_bytes()
                diagnostic.synthetic(model,out)
                self.assertEqual(initial,(out/'synthetic-angle-trials.csv').read_bytes())
        finally:torch.set_num_threads(old)


if __name__=='__main__':unittest.main()
