from dataclasses import asdict
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import SimpleITK as sitk

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
from experiments import get_experiment_config
from recover_sidecar_output import recover,validate,sha,log_text

PROFILE='hierarchical_full_grid0_iterations2_memory_topk_diverse'
RUNTIME={'backend':'reference','point_chunk_size':0,'gain':.5,'max_correction_pixels':.5,'checkpoint_blocks':False}


class RecoveryTests(unittest.TestCase):
    def fixture(self,root):
        folder=root/'previous/A_010-b2a86cdc';output=folder/'output/images/mri-linac-series-targets/output.mha'
        output.parent.mkdir(parents=True)
        case=root/'case';(case/'images').mkdir(parents=True)
        mask=np.zeros((4,5,3),dtype=np.uint8);mask[1:3,1:4,:]=1
        sitk.WriteImage(sitk.GetImageFromArray(mask),str(output),True)
        sitk.WriteImage(sitk.GetImageFromArray(mask.astype(np.float32)),str(case/'images/A_010_frames.mha'))
        np.savez_compressed(folder/'output/sidecar-tracks.npz',trajectories=np.zeros((1,3,2,2)),queries=np.zeros((1,2,3)),
            visibility=np.ones((1,3,2)),confidence=np.ones((1,3,2)))
        config=asdict(get_experiment_config(PROFILE)[1])
        diagnostic={'profile':PROFILE,'config':config,
            'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'model':{'feature_sidecar':'mamba','sidecar_runtime':RUNTIME,'frozen_pretrained':True,'backbone_architecture':'base'},
            'prediction':{'shape':list(mask.shape),'foreground_voxels':int(mask.sum()),'array_sha256':hashlib.sha256(mask.tobytes()).hexdigest()},
            'mechanism':{'query_points':2}}
        text='Runtime loading: 0.1 s\nTRACKRAD_DIAGNOSTICS_JSON='+json.dumps(diagnostic)+'\nRuntime algorithm: 112.1 s\nRuntime writing: 0.146 s\n'
        (folder/'case.log').write_text(text)
        dest=root/'new';(dest/'output').mkdir(parents=True)
        (dest/'recovery.log').write_text('host recovery log')
        return folder,case,dest,diagnostic

    def test_reuse_bytes_without_inference_or_deleting_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,case,dest,_=self.fixture(root)
            old=sha(folder/'output/images/mri-linac-series-targets/output.mha')
            result=recover(root/'previous',dest,case,'A_010',PROFILE,'mamba',RUNTIME,'a'*64)
            self.assertTrue(result['recovered'])
            self.assertEqual(old,sha(dest/'output/images/mri-linac-series-targets/output.mha'))
            self.assertTrue(folder.is_dir())
            self.assertEqual(old,sha(folder/'output/images/mri-linac-series-targets/output.mha'))
            self.assertEqual((folder/'case.log').read_bytes(),(dest/'case.log').read_bytes())

    def test_runtime_config_mask_and_cache_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,case,dest,d=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'runtime'):
                validate(folder,case,'A_010',PROFILE,'mamba',{**RUNTIME,'gain':.25})
            original=(folder/'case.log').read_text()
            for bad in (original.replace('Runtime writing: 0.146 s',''),original+'Traceback\n',original.replace(d['prediction']['array_sha256'],'0'*64)):
                (folder/'case.log').write_text(bad)
                with self.assertRaises(ValueError): validate(folder,case,'A_010',PROFILE,'mamba',RUNTIME)
            (folder/'case.log').write_text(original)
            np.savez_compressed(folder/'output/sidecar-tracks.npz',trajectories=np.zeros((1,2,2,2)),queries=np.zeros((1,2,3)))
            with self.assertRaisesRegex(ValueError,'cache'): validate(folder,case,'A_010',PROFILE,'mamba',RUNTIME)
            self.assertFalse((dest/'output/images').exists())

    def test_powershell5_encodings_and_runner_timestamp_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,case,dest,_=self.fixture(root)
            text=(folder/'case.log').read_text()
            (folder/'case.log').write_bytes(text.encode('utf-16'))
            validate(folder,case,'A_010',PROFILE,'mamba',RUNTIME)
            end=datetime.fromtimestamp((folder/'case.log').stat().st_mtime,timezone.utc)
            start=end-timedelta(seconds=116)
            log=root/'runner.log'
            host=f'[{start.isoformat()}] Running: {PROFILE}/A_010\n'
            native='Runtime writing: 0.146 s\n'.encode('utf-16-le')
            failure=f'[{end.isoformat()}] FAILED {PROFILE}: {PROFILE}/A_010 did not produce a valid output.mha\n'
            raw=host.encode()+native+failure.encode()
            log.write_bytes(raw)
            self.assertIn('Runtime writing: 0.146 s',log_text(raw))
            result=recover(root/'previous',dest,case,'A_010',PROFILE,'mamba',RUNTIME,'a'*64,log)
            self.assertEqual((datetime.fromisoformat(result['completed_at'])-datetime.fromisoformat(result['started_at'])).total_seconds(),116)
            self.assertTrue(result['runtime_estimated'])
            self.assertEqual((dest/'runner-snapshot.log').read_bytes(),raw)

    def test_ambiguous_or_used_destination_refused_and_no_candidate_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,case,dest,_=self.fixture(root)
            (dest/'unrelated.txt').write_text('preserve')
            with self.assertRaisesRegex(ValueError,'destination'): recover(root/'previous',dest,case,'A_010',PROFILE,'mamba',RUNTIME,'a'*64)
            sibling=root/'previous/A_010-aaaaaaaa/output/images/mri-linac-series-targets';sibling.mkdir(parents=True)
            (sibling/'output.mha').write_bytes(b'not a valid MHA')
            with self.assertRaisesRegex(ValueError,'Multiple'): recover(root/'previous',dest,case,'A_010',PROFILE,'mamba',RUNTIME,'a'*64)
            empty=root/'empty';empty.mkdir();new=root/'new_empty';new.mkdir()
            result=recover(empty,new,case,'A_010',PROFILE,'mamba',RUNTIME,'a'*64)
            self.assertFalse(result['recovered'])


if __name__=='__main__': unittest.main()
