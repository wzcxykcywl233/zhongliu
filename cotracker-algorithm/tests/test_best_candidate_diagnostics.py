import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

FILE=Path(__file__).resolve().parents[1]/'experiments'/'diagnose_best_candidates.py'
spec=importlib.util.spec_from_file_location('best_diagnostic',FILE)
d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)


class DiagnosticsTest(unittest.TestCase):
    def test_helper_compatibility(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime=Path(folder)/'runtime'; reference=Path(folder)/'reference'
            runtime.mkdir(); reference.mkdir()
            for name in ('reshape.py','seg_to_tap.py','tap_to_seg.py'):
                (runtime/name).write_bytes(b'x = 1\r\n')
                (reference/name).write_bytes(b'x = 1\n')
            self.assertEqual(len(d.verify_helpers(runtime,reference)),3)
            (runtime/'reshape.py').write_text('x = 2\n')
            with self.assertRaisesRegex(ValueError,'helper mismatch'):
                d.verify_helpers(runtime,reference)

    def test_shift_no_wrap(self):
        a=np.zeros((8,9),bool); a[0,0]=True
        self.assertFalse(d.shift_mask(a,-1,0).any())
        self.assertTrue(d.shift_mask(a,2,3)[2,3])
        self.assertFalse(d.shift_mask(a,9,0).any())

    def test_translation_recovered(self):
        a=np.zeros((30,40),bool); a[9:15,12:20]=True
        b=d.shift_mask(a,3,-4)
        r=d.geometry(b,a)
        self.assertEqual(r['CentroidAlignedDice'],1)
        self.assertEqual(r['CentroidDistancePixels'],5)
        self.assertEqual((r['ShiftY'],r['ShiftX']),(-3,4))
        self.assertGreater(r['CentroidGain'],0)

    def test_shape_not_fixed_by_centroid(self):
        a=np.zeros((21,21),bool); b=a.copy()
        a[7:14,7:14]=True; b[9:12,5:16]=True
        r=d.geometry(b,a)
        self.assertEqual(r['CentroidGain'],0)
        self.assertLess(r['CentroidAlignedDice'],1)

    def test_empty_explicit(self):
        a=np.zeros((4,4),bool); b=a.copy(); b[1,1]=True
        self.assertIsNone(d.geometry(a,a)['FrameDice'])
        self.assertEqual(d.geometry(a,b)['FrameDice'],0)
        self.assertIsNone(d.geometry(a,b)['CentroidGain'])

    def test_bootstrap_case_unit(self):
        r=d.bootstrap([.1,.1,.1],100)
        self.assertAlmostEqual(r['BootstrapLow95'],.1)
        self.assertEqual(r['Cases'],3)
        with self.assertRaises(ValueError): d.bootstrap([np.nan])

    def test_resume_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'frames.csv').write_text('good')
            d.atomic_json(root/'complete.json',{'files':{'frames.csv':d.sha(root/'frames.csv')}})
            self.assertTrue(d.valid_job(root))
            (root/'frames.csv').write_text('bad')
            self.assertFalse(d.valid_job(root))

    def test_single_variable(self):
        a=dict(hierarchical_span=10,n_iterations=2,support_grid_size=0,
               query_memory_mode='topk_confidence_diversity',query_memory_refinement='none',query_state_inheritance='none')
        b={**a,'query_state_inheritance':'vc_decay'}
        d.validate_configs(a,b)
        with self.assertRaises(ValueError): d.validate_configs(a,{**b,'n_iterations':4})

    def test_summary_preserves_official_and_excludes_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); cases=['A','B']
            official={p:{} for p in d.PROFILES}
            for case in cases:
                folder=root/'jobs'/case; folder.mkdir(parents=True)
                for i,p in enumerate(d.PROFILES):
                    official[p][case]={k:.8+i*.01 for k in d.METRICS.values()}
                d.export(folder/'frames.csv',[dict(Case=case,Profile=p,Frame=1,TimeQuartile=1,
                    NominalSpanOffset=1,Region='abdomen',FieldStrength=1.5,FrameDice=.7,
                    CentroidDistancePixels=None,AreaRatio=None,CentroidGain=None) for p in d.PROFILES])
                d.export(folder/'representation.csv',[dict(Case=case,Frame=1,Valid=False,Reason='empty')])
            d.summarize(root,cases,official,'test-38')
            self.assertEqual(d.read(root/'summary.json')['representation_failed_frames'],2)
            self.assertEqual(float(d.csv_rows(root/'official-case-pairs.csv')[0]['Control_DSC']),.8)


if __name__=='__main__': unittest.main()
