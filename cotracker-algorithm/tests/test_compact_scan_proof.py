import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'ext'/'co-tracker')]
from verify_compact_scan_proof import verify,RUNTIME
from profile_sidecar_training_phases import read_probe_runtime,check_training_headroom


class CompactProofTests(unittest.TestCase):
    def fixture(self,root):
        manifest=[];hashes={}
        for kind in ('mlp','mamba_short','mamba'):
            name=kind+'_compact_0_cp0';folder=root/'isolated'/name;folder.mkdir(parents=True)
            manifest.append({'name':name,'kind':kind,'runtime':RUNTIME})
            checksum={'mlp':'a','mamba_short':'b','mamba':'c'}[kind]*64
            hashes[f'seed_0/{kind}/cotracker_three_final.pth']=checksum
            record={'complete':True,'kind':kind,'runtime':RUNTIME,'fresh_process_per_configuration':True,
                'correctness':{'output_and_all_parameter_gradients_close':True},
                'model_parity':{'complete':True,'max_abs_pvc':[0.,0.,0.]},'source_checkpoint_sha256':checksum,
                'trials':[{'driver_total_mib':24563.5,'driver_free_after_mib':14000}],
                'peak_allocated_mib':7000,'peak_reserved_mib':8000,'median_step_seconds':1.2}
            (folder/'result.json').write_text(json.dumps(record))
        (root/'frozen-study.json').write_text(json.dumps({'CompactScan':True,'ProbeSteps':3}))
        (root/'probe-manifest.json').write_text(json.dumps(manifest))
        (root/'source-training-audit.json').write_text(json.dumps({'final_checkpoints_sha256':hashes}))
        return root/'isolated'/'mamba_compact_0_cp0'/'result.json'

    def test_gpu_evidence_weights_memory_and_timeout_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.fixture(root)
            result=json.loads(path.read_text())
            proof=verify(root)
            self.assertTrue(proof['complete']);self.assertEqual(len(proof['result_sha256']),3)
            for change in ({'complete':False},{'peak_allocated_mib':28335},{'peak_reserved_mib':28554},
                           {'peak_allocated_mib':float('nan')},{'source_checkpoint_sha256':'0'*64},
                           {'model_parity':{'complete':False}},
                           {'trials':[{'driver_total_mib':24563.5,'driver_free_after_mib':0}]}):
                path.write_text(json.dumps({**result,**change}))
                with self.assertRaises(ValueError): verify(root)

    def test_runtime_cannot_alter_training_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runtime.json';path.write_text(json.dumps(RUNTIME),encoding='utf-8-sig')
            self.assertEqual(read_probe_runtime(path),RUNTIME)
            self.assertIsNone(read_probe_runtime(None))
            for change in ({'gain':.5},{'max_correction_pixels':.5},{'checkpoint_blocks':True},{'point_chunk_size':16},{'backend':'parallel'}):
                path.write_text(json.dumps({**RUNTIME,**change}))
                with self.assertRaises(ValueError): read_probe_runtime(path)

    def test_real_probe_aborts_on_returned_memory_pressure(self):
        good={'memory_after':{'peak_allocated_mib':8000,'peak_reserved_mib':9500,'driver_free_mib':13000}}
        check_training_headroom(good)
        for change in ({'peak_allocated_mib':27000},{'peak_reserved_mib':28000},{'driver_free_mib':0}):
            with self.assertRaisesRegex(ValueError,'headroom'):
                check_training_headroom({'memory_after':{**good['memory_after'],**change}})


if __name__=='__main__': unittest.main()
