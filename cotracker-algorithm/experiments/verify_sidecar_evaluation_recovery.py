"""Allow only the reviewed import fix; reuse completed training read-only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ext' / 'co-tracker'), str(Path(__file__).parent)]
from audit_feature_sidecar import audit
from audit_backbone_growth import file_sha
from cotracker.models.core.cotracker.feature_sidecar import KINDS, tensor_hash

KNOWN_OLD = {
    'cotracker-algorithm/model.py': '8f7fabb6c8fce301100c3d2daef69e0c448ebf5f35a29023c383ede4b302fdd4',
    'cotracker-algorithm/Dockerfile.inference.reuse': '408e54ff52b8902eff1151b2c0ff7cde3e69ecc372342f6d605393e001407980',
    'cotracker-algorithm/experiments/audit_feature_sidecar.py': '4732ec6e0719eb69249acc14678172d22608ab94c61a49b3703029aad66966c6',
}
ADDED = {
    'cotracker-algorithm/local_cotracker_source.py',
    'cotracker-algorithm/experiments/verify_sidecar_evaluation_recovery.py',
    'cotracker-algorithm/experiments/preflight_sidecar_inference.py',
    'cotracker-algorithm/tests/test_sidecar_inference_recovery.py',
    'scripts/repair_feature_sidecar_inference.ps1',
    'scripts/tests/test_sidecar_inference_recovery.ps1',
}
REVERSE_EDITS = {
    'cotracker-algorithm/model.py': [
        ('from local_cotracker_source import activate_local_cotracker\nactivate_local_cotracker()\n\n', '')],
    'cotracker-algorithm/Dockerfile.inference.reuse': [
        ('ENV PYTHONPATH=/opt/app/ext/co-tracker:/opt/app\n', '')],
    'cotracker-algorithm/experiments/audit_feature_sidecar.py': [
        ('def audit(root, seeds, steps, output=None):', 'def audit(root, seeds, steps):'),
        ("save_json(output if output is not None else root / 'training-audit.json', proof)",
         "save_json(root / 'training-audit.json', proof)"),
        ('    return proof\n', '')],
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_sources(old, rows, repo):
    before = {p['File']:p['SHA256'].lower() for p in old['Source']}
    after = {p['File']:p['SHA256'].lower() for p in rows}
    if len(after) != len(rows) or len(before) != len(old['Source']):
        raise ValueError('duplicate source paths')
    if set(before) - set(after) or (set(after) - set(before)) - ADDED:
        raise ValueError('unexpected added/deleted code; refuse training reuse')
    changed = {p for p in before if before[p] != after[p]}
    if changed - KNOWN_OLD.keys():
        raise ValueError('model/training/protocol code changed: ' + str(sorted(changed)))
    for relative, digest in after.items():
        path = (repo / relative).resolve()
        if repo.resolve() not in path.parents or file_sha(path) != digest:
            raise ValueError('source fingerprint mismatch: ' + relative)
    for relative in changed:
        text = (repo / relative).read_text(encoding='utf-8')
        for new, previous in REVERSE_EDITS[relative]:
            if text.count(new) != 1:
                raise ValueError('import/audit fix is not the reviewed edit: ' + relative)
            text = text.replace(new, previous)
        canonical = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if canonical != KNOWN_OLD[relative]:
            raise ValueError('additional numerical changes in: ' + relative)
        # Git forces LF for .py but Dockerfiles may use Windows CRLF. Permit
        # only these two encodings of the EXACT reviewed original source.
        crlf = hashlib.sha256(text.replace('\n','\r\n').encode('utf-8')).hexdigest()
        if before[relative] not in (canonical, crlf):
            raise ValueError('unrecognized old source: ' + relative)
    if changed != set(KNOWN_OLD):
        raise ValueError('incomplete import recovery source')
    return sorted(changed)


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError('evaluation provenance changed; use a new results root')
        return
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


def verify(old_root, new_root, repo, current_source):
    old_root, new_root = old_root.resolve(), new_root.resolve()
    if old_root == new_root or old_root in new_root.parents or new_root in old_root.parents:
        raise ValueError('old and new studies must be separate')
    old = read_json(old_root / 'frozen-run.json')
    if (old['Architectures'] != list(KINDS) or old['Seeds'] != [0,1,2] or old['Steps'] != 1000 or
        old['SequenceLength'] != 60 or old['Trajectories'] != 128 or old['TrainIterations'] != 4 or
        old['LearningRate'] != .00005 or old['AuxiliaryTeacherWeight'] != 0 or old['MaskLossWeight'] != 0 or
        old['InferenceProfile'] != 'hierarchical_full_grid0_iterations2_memory_topk_diverse'):
        raise ValueError('unsupported completed sidecar training protocol')
    changed = verify_sources(old, read_json(current_source), repo)
    original = read_json(old_root / 'train' / 'training-audit.json')
    # Recompute the full model/input/parameter audit, writing ONLY to new_root.
    fresh = audit(old_root / 'train', old['Seeds'], old['Steps'],
                  output=new_root / 'rechecked-training-audit.json')
    if fresh != original:
        raise ValueError('old completed training audit no longer matches the weights/inputs')
    reference = torch.load(old_root / 'pretrained-frozen.pth', map_location='cpu', weights_only=True)
    weight = next(row for row in old['Weights'] if row['Name'] == 'scaled_offline.pth')
    init = read_json(old_root / 'train' / 'seed_0' / 'mlp' / 'initialization-report.json')
    if (reference.get('source_checkpoint_sha256', '').lower() != weight['SHA256'].lower() or
        not reference.get('freeze_base_for_inference') or reference.get('backbone_architecture') != 'base' or
        tensor_hash(reference['model']) != init['sidecar']['base_tensor_sha256']):
        raise ValueError('frozen pretrained reference is not the original training backbone')
    artifacts = {}
    frozen_sha = file_sha(old_root / 'frozen-run.json')
    for seed in old['Seeds']:
        for kind in KINDS:
            folder = old_root / 'train' / f'seed_{seed}' / kind
            config = read_json(folder / 'run-config.json')
            if config['source_protocol_sha256'].lower() != frozen_sha:
                raise ValueError('training arm is not bound to the original study')
            for name in ('cotracker_three_final.pth', 'paired-steps.jsonl',
                         'initialization-report.json', 'run-config.json'):
                relative = str((folder / name).relative_to(old_root)).replace('\\','/')
                artifacts[relative] = file_sha(folder / name)
    proof = {'complete': True, 'training_read_only': True, 'retrained_arms': 0,
             'source_frozen_sha256': frozen_sha, 'source_training_audit_sha256':
             file_sha(old_root / 'train' / 'training-audit.json'),
             'pretrained_reference_sha256': file_sha(old_root / 'pretrained-frozen.pth'),
             'changed_sources': changed, 'evaluation_sources': read_json(current_source),
             'training_artifacts_sha256': artifacts,
             'note': 'No training or copied cached predictions. All evaluation arms rerun with explicit repository imports.'}
    immutable_json(new_root / 'evaluation-recovery-audit.json', proof)
    print('VERIFIED 12 completed arms; no training will be launched.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    a = p.parse_args()
    verify(a.old, a.new, a.repo, a.source)
