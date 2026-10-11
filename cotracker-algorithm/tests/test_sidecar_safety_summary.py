import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'ext' / 'co-tracker'),str(ROOT / 'experiments')]
from summarize_sidecar_speed_safety import choose_candidates, compare_caches, cohort_diagnostics


class SafetySummaryTests(unittest.TestCase):
    def test_cohort_diagnosis_collapses_seeds_not_extra_cases(self):
        rows=[]
        for seed in (0,1,2):
            for case,sign in (('A_001',1),('B_001',-1)):
                rows.append({'Seed':seed,'Arm':'mamba_full','Case':case,
                    'Delta_DSC':sign*.001,'Delta_HD95':-.1,'Delta_MASD':-.01,
                    'Delta_CD':-.01,'Delta_D98':-.002,'Frames':60,
                    'CorrectionMeanInputPixels':.1,'TrackMeanDisplacementInputPixels':.2,
                    'MaskChangedFraction':.01,'ClippedFraction':0})
        summary=cohort_diagnostics(rows)
        total=next(r for r in summary if r['Cohort']=='all')
        self.assertEqual(total['Cases'],2)
        self.assertEqual(total['DSCImprovedCases'],1)
        self.assertEqual(total['DSCWorsenedCases'],1)
        self.assertEqual(total['Delta_DSC_Mean'],0)
        with self.assertRaisesRegex(ValueError,'three paired seeds'):
            cohort_diagnostics(rows[:-1])

    def test_selection_uses_validation_constraints_not_test_ranking(self):
        def row(name,kind,dsc,dd98):
            return {'Arm':name,'Kind':kind,'Delta_DSC_Mean':dsc-.9,'Delta_D98_Mean':dd98,
                    'DSC_Mean':dsc,'HD95_Mean':4.,'CD_Mean':2.,'D98_Mean':.95+dd98}
        rows = [row('mlp_full','mlp',.899,-.002),row('mlp_gain050','mlp',.9001,-.0008),
                row('mamba_full','mamba',.902,-.004),row('mamba_gain025','mamba',.9002,0)]
        self.assertEqual(choose_candidates(rows),['mlp_gain050','mamba_gain025'])
        self.assertEqual(choose_candidates([rows[0],rows[2]]),[])

    def test_correction_trajectory_and_mask_change_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            a,b = Path(temporary)/'a',Path(temporary)/'b'
            for folder in (a,b): (folder/'output/images/mri-linac-series-targets').mkdir(parents=True)
            query=np.array([[[0,1,1],[0,2,2]]],dtype=np.float32)
            reference=np.ones((1,3,2,2),dtype=np.float32)
            changed=reference.copy();changed[:,1:,:,0]+=.5
            for folder,tracks in ((a,changed),(b,reference)):
                np.savez_compressed(folder/'output/sidecar-tracks.npz',queries=query,trajectories=tracks)
                mask=np.zeros((3,4,4),dtype=np.uint8);mask[:,1:3,1:3]=1
                sitk.WriteImage(sitk.GetImageFromArray(mask),str(folder/'output/images/mri-linac-series-targets/output.mha'))
            report=compare_caches(a,b)
            self.assertEqual(report['TrackMeanDisplacementInputPixels'],.5)
            self.assertEqual(report['MaskChangedVoxels'],0)
            np.savez_compressed(a/'output/sidecar-tracks.npz',queries=query+1,trajectories=changed)
            with self.assertRaisesRegex(ValueError,'not paired'): compare_caches(a,b)


if __name__=='__main__': unittest.main()
