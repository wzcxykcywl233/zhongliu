"""Read-only completed-training verification for the separate speed/safety study."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'ext' / 'co-tracker'),str(Path(__file__).parent)]
from audit_feature_sidecar import audit
from audit_backbone_growth import file_sha
from cotracker.models.core.cotracker.backbone_growth import save_json
from cotracker.models.core.cotracker.feature_sidecar import tensor_hash


def verify(old, new, repo):
    old_protocol = json.loads((old / 'frozen-run.json').read_text(encoding='utf-8-sig'))
    original = json.loads((old / 'train' / 'training-audit.json').read_text())
    # The actual trainer is unchanged: no loss, teacher, loader or schedule edits
    # are needed to profile/reuse these weights or select a scan runtime.
    relative = 'cotracker-algorithm/ext/co-tracker/train_on_real_data.py'
    expected = next(p['SHA256'] for p in old_protocol['Source'] if p['File']==relative)
    if file_sha(repo / relative) != expected.lower(): raise ValueError('Original trainer changed')
    if old_protocol['Seeds'] != [0,1,2] or old_protocol['Steps'] != 1000: raise ValueError('Wrong old training study')
    fresh = audit(old / 'train',[0,1,2],1000,output=new / 'rechecked-training-audit.json')
    if fresh != original: raise ValueError('Original paired inputs/frozen parameters/checkpoints changed')
    reference = torch.load(old / 'pretrained-frozen.pth',map_location='cpu',weights_only=True)
    init = json.loads((old / 'train' / 'seed_0' / 'mlp' / 'initialization-report.json').read_text())
    weight = next(w for w in old_protocol['Weights'] if w['Name']=='scaled_offline.pth')
    if not reference.get('freeze_base_for_inference') or reference['source_checkpoint_sha256'].lower()!=weight['SHA256'].lower() or tensor_hash(reference['model'])!=init['sidecar']['base_tensor_sha256']:
        raise ValueError('Frozen reference differs from original pretrained weights')
    evidence = {'complete':True,'training_read_only':True,'original_frozen_sha256':file_sha(old / 'frozen-run.json'),
                'original_training_audit_sha256':file_sha(old / 'train' / 'training-audit.json'),
                'pretrained_reference_sha256':file_sha(old / 'pretrained-frozen.pth'),
                'final_checkpoints_sha256':fresh['final_checkpoints_sha256']}
    path = new / 'source-training-audit.json'
    if path.exists() and json.loads(path.read_text()) != evidence: raise ValueError('Recovery source changed')
    save_json(path,evidence)
    print('OLD 12 TRAINING ARMS VERIFIED READ-ONLY',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    a = p.parse_args()
    verify(a.old,a.new,a.repo)
