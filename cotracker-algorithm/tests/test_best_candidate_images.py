"""Optional real-image-library integration tests; no model or GPU required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('best_image_diag',ROOT/'experiments'/'diagnose_best_candidates.py')
d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
AVAILABLE=all(importlib.util.find_spec(x) is not None for x in ('SimpleITK','torch','cv2','skimage','scipy'))


@unittest.skipUnless(AVAILABLE,'image libraries not installed')
class ImageIntegration(unittest.TestCase):
    def test_roundtrip_rectangle(self):
        mask=np.zeros((75,103),bool); mask[15:55,25:80]=True
        r=d.representation(mask,ROOT/'resources')
        self.assertGreater(r['Contour1000Dice'],.95)
        self.assertGreater(r['ResizeOnlyDice'],.99)
        self.assertEqual(len(r),5)

    def test_non_square_case_overlay_and_sample_coverage(self):
        import SimpleITK as sitk
        from dataclasses import asdict
        import sys
        sys.path.insert(0,str(ROOT))
        from experiments import QUERY_STATE_EXPERIMENTS
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); case='A_001'; dataset=root/'dataset'; previous=root/'previous'
            folder=dataset/case; (folder/'targets').mkdir(parents=True); (folder/'images').mkdir()
            mask=np.zeros((4,75,103),np.uint8); mask[:,15:55,25:80]=1
            def save(path,a): sitk.WriteImage(sitk.GetImageFromArray(a.transpose(2,1,0)),str(path))
            save(folder/'targets'/f'{case}_labels.mha',mask)
            save(folder/'images'/f'{case}_frames.mha',mask*100)
            d.atomic_json(folder/'scanned-region.json','abdomen'); d.atomic_json(folder/'b-field-strength.json',1.5)
            for p in d.PROFILES:
                job=previous/p/'checkpoint'/'jobs'/case
                output=job/'output'/'images'/'mri-linac-series-targets'/'output.mha'
                output.parent.mkdir(parents=True)
                pred=np.stack([d.shift_mask(x,2,1) for x in mask]).astype(np.uint8)
                save(output,pred)
                d.atomic_json(job/'diagnostics.json',{'profile':p,'config':asdict(QUERY_STATE_EXPERIMENTS[p]),
                              'prediction':{'output_file_sha256':d.sha(output)}})
            attempt=root/'attempt'; attempt.mkdir()
            d.analyze_case(case,d.files_for(dataset,previous,case),attempt,ROOT/'resources',2)
            frames=d.csv_rows(attempt/'frames.csv')
            self.assertEqual(len(frames),6)
            self.assertTrue(all(float(r['CentroidAlignedDice'])==1 for r in frames))
            self.assertEqual(len(d.csv_rows(attempt/'representation.csv')),2)
            self.assertGreater(len(list(attempt.glob('frame-*.png'))),0)


if __name__=='__main__': unittest.main()
