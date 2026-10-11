"""Source verification adapter: old artifacts are immutable and re-audited."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ext/co-tracker'),str(ROOT/'experiments')]
import audit_sidecar_speed_source as source


class SpeedSourceTests(unittest.TestCase):
    def fixture(self,temporary):
        old,new,repo=(Path(temporary)/p for p in ('old','new','repo'))
        (old/'train/seed_0/mlp').mkdir(parents=True)
        new.mkdir()
        trainer=repo/'cotracker-algorithm/ext/co-tracker/train_on_real_data.py'
        trainer.parent.mkdir(parents=True)
        trainer.write_text('unchanged trainer',encoding='utf-8')
        protocol={'Source':[{'File':str(trainer.relative_to(repo)).replace('\\','/'),
                              'SHA256':source.file_sha(trainer)}],
                  'Seeds':[0,1,2],'Steps':1000,
                  'Weights':[{'Name':'scaled_offline.pth','SHA256':'a'*64}]}
        audit={'complete':True,'final_checkpoints_sha256':{'seed_0/mlp/final.pth':'b'*64}}
        (old/'frozen-run.json').write_text(json.dumps(protocol),encoding='utf-8')
        (old/'train/training-audit.json').write_text(json.dumps(audit),encoding='utf-8')
        (old/'train/seed_0/mlp/initialization-report.json').write_text(
            json.dumps({'sidecar':{'base_tensor_sha256':'matched-base'}}),encoding='utf-8')
        (old/'pretrained-frozen.pth').write_bytes(b'immutable fixture')
        return old,new,repo,trainer,audit

    def test_reaudit_writes_only_new_root_and_unchanged_trainer_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            old,new,repo,trainer,audit=self.fixture(temporary)
            before={str(p):p.read_bytes() for p in old.rglob('*') if p.is_file()}
            with patch.object(source,'audit',return_value=audit) as reaudit, \
                 patch.object(source.torch,'load',return_value={'freeze_base_for_inference':True,
                     'source_checkpoint_sha256':'a'*64,'model':{}}), \
                 patch.object(source,'tensor_hash',return_value='matched-base'):
                source.verify(old,new,repo)
                self.assertEqual(reaudit.call_args.kwargs['output'],new/'rechecked-training-audit.json')
                self.assertTrue((new/'source-training-audit.json').is_file())
                self.assertEqual(before,{str(p):p.read_bytes() for p in old.rglob('*') if p.is_file()})
                trainer.write_text('modified trainer',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Original trainer changed'):
                    source.verify(old,new,repo)
                self.assertEqual(reaudit.call_count,1)

    def test_changed_pair_audit_refused_before_reference_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            old,new,repo,_,audit=self.fixture(temporary)
            with patch.object(source,'audit',return_value={**audit,'complete':False}), \
                 patch.object(source.torch,'load') as load:
                with self.assertRaisesRegex(ValueError,'paired inputs/frozen parameters/checkpoints'):
                    source.verify(old,new,repo)
                load.assert_not_called()


if __name__=='__main__': unittest.main()
