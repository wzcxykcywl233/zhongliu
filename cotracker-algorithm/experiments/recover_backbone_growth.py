"""Copy fully verified arms to a new study; never edit or repair the old study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ext' / 'co-tracker')]
from audit_backbone_growth import records, file_sha, validate_checkpoint
from cotracker.models.core.cotracker.backbone_growth import ARCHITECTURES, PARAMETERS, save_json
from cotracker.utils.teacher_sampling import TeacherPairSampler

OLD_TRAINER_HASHES = {'1531458d7c1da577e59cabc15fae4a4c6ec746311371c7cd722abbe0cd1dd5e4',
                      'd5534473c6b4c27d395d824ea6bafb3813025d8c1d3634b90442df737e03979d'}
TRAINER = 'cotracker-algorithm/ext/co-tracker/train_on_real_data.py'
ALLOWED_CHANGED = {TRAINER, 'scripts/run_backbone_growth_resumable.ps1',
                   'cotracker-algorithm/experiments/audit_backbone_growth.py',
                   'cotracker-algorithm/experiments/BACKBONE_GROWTH.zh-CN.md'}
ALLOWED_ADDED = {'cotracker-algorithm/ext/co-tracker/cotracker/utils/resumable_loader.py',
                 'cotracker-algorithm/experiments/recover_backbone_growth.py',
                 'cotracker-algorithm/tests/test_backbone_resume.py',
                 'scripts/repair_backbone_growth_resume.ps1'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify_transition(old, new, source_root):
    clean = lambda p: {k:v for k,v in p.items() if k not in ('Schema','Source','RecoverySource')}
    if clean(old) != clean(new):
        raise ValueError('weights, data, seeds or training/inference protocol changed')
    before = {p['File']:p['SHA256'].lower() for p in old['Source']}
    after = {p['File']:p['SHA256'].lower() for p in new['Source']}
    if before.get(TRAINER) not in OLD_TRAINER_HASHES:
        raise ValueError('unknown original trainer; automatic reuse is not permitted')
    if set(before) - set(after) or (set(after) - set(before)) - ALLOWED_ADDED:
        raise ValueError('unexpected added/deleted source files')
    changed = {p for p in before if before[p] != after[p]}
    if changed - ALLOWED_CHANGED:
        raise ValueError('numerical/data/model code changed: ' + str(sorted(changed - ALLOWED_CHANGED)))
    # Prove the actual trainer changes are ONLY epoch alignment. An arbitrary
    # trainer change is not accepted merely because its filename is allowlisted.
    trainer = (source_root / TRAINER).read_text(encoding='utf-8').replace('\r\n','\n')
    trainer = trainer.replace('from cotracker.utils.resumable_loader import set_resumable_loader_epoch\n', '')
    trainer = trainer.replace('            if args.backbone_growth_study:\n                set_resumable_loader_epoch(train_loader, epoch)\n', '')
    if hashlib.sha256(trainer.encode()).hexdigest() not in OLD_TRAINER_HASHES:
        raise ValueError('trainer change is not the verified resume-only fix')
    return sorted(changed)


def verify_schedule(rows, config, case_ids, steps):
    if len(case_ids) != 40 or config['sequence_len'] != 10:
        raise ValueError('unsupported original loader protocol')
    # Original Fabric single-rank DistributedSampler uses seed=0 and advances
    # once per epoch. The repaired iterator gives exactly the same uninterrupted
    # sequence, including at the mid-epoch resume boundary.
    sampler = torch.utils.data.DistributedSampler(case_ids, num_replicas=1, rank=0, shuffle=True, seed=0)
    teacher = TeacherPairSampler(3, seed=config['teacher_seed'])
    order = []
    for epoch in range((steps + 39) // 40):
        sampler.set_epoch(epoch)
        order.extend(case_ids[i] for i in sampler)
    for index in range(steps):
        signature = rows[index]
        if json.loads(signature[0]) != [order[index]] or json.loads(signature[3]) != teacher.sample().primary:
            raise ValueError('case/teacher schedule changed at step ' + str(index))


def choose_reusable(candidates):
    # Require at least two architectures with identical FULL input sequences,
    # rather than accepting the matching tail or trusting a single arm.
    groups = []
    for name, rows in candidates.items():
        for group in groups:
            if candidates[group[0]] == rows:
                group.append(name)
                break
        else:
            groups.append([name])
    accepted = [group for group in groups if len(group) >= 2]
    if len(accepted) != 1:
        raise ValueError('no unambiguous complete paired reference remains')
    return accepted[0]


def copy_atomic(source, target):
    temporary = target.with_name(target.name + '.import.tmp')
    with source.open('rb') as src, temporary.open('wb') as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if file_sha(temporary) != file_sha(source):
        raise ValueError('copy checksum differs')
    os.replace(temporary, target)


def recover(old_root, new_root, source_root):
    old_root, new_root = old_root.resolve(), new_root.resolve()
    if old_root == new_root or old_root in new_root.parents or new_root in old_root.parents:
        raise ValueError('recovery requires a separate sibling result directory')
    torch.set_num_threads(1)
    old, new = read_json(old_root / 'frozen-run.json'), read_json(new_root / 'frozen-run.json')
    changed = verify_transition(old, new, source_root)
    old_runtime, new_runtime = read_json(old_root / 'runtime-check.json'), read_json(new_root / 'runtime-check.json')
    for key in ('python','torch','torch_cuda','pytorch_lightning'):
        if old_runtime[key] != new_runtime[key]:
            raise ValueError('training environment changed: ' + key)
    case_ids = sorted({row['Case'] for row in old['Dataset'] if row['Split'] == 'train-40'})
    accepted, rejected, evidence = [], [], {}
    for seed in old['Seeds']:
        candidates, reasons = {}, {}
        for arch in ARCHITECTURES:
            relative = 'seed_' + str(seed) + '/' + arch
            folder = old_root / 'train' / relative
            try:
                config = read_json(folder / 'run-config.json')
                expected = {'profile':f'seed_{seed}_{arch}', 'architecture':arch, 'steps':old['Steps'],
                            'save_every_steps':old['SaveEverySteps'], 'seed':seed, 'training_seed':20261005+seed,
                            'paired_step_seed':20261005+seed, 'teacher_seed':20261005+seed,
                            'source_protocol_sha256':file_sha(old_root / 'frozen-run.json').upper(),
                            'sequence_len':10,'traj_per_sample':384,'train_iters':4,'lr':0.00005,
                            'auxiliary_teacher_weight':0,'confidence_target_mode':'hard','mask_loss_weight':0}
                # Dataset path must be exactly the path frozen with its hashes.
                paths = {row['File'].split('\\' + row['Case'] + '\\')[0] for row in old['Dataset'] if row['Split'] == 'train-40'}
                if len(paths) != 1:
                    raise ValueError('invalid frozen training paths')
                expected['dataset'] = next(iter(paths))
                if config != expected:
                    raise ValueError('original arm config differs from frozen protocol')
                init = read_json(folder / 'initialization-report.json')
                if init['architecture'] != arch or init['parameters'] != PARAMETERS[arch] or not init['initialization_outputs_exact'] or any(init['initialization_max_abs_difference']):
                    raise ValueError('initialization proof failed')
                rows = records(folder / 'paired-steps.jsonl', old['Steps'], arch)
                verify_schedule(rows, config, case_ids, old['Steps'])
                validate_checkpoint(folder, arch, old['Steps'])
                candidates[arch] = rows
            except (ValueError, KeyError, OSError, RuntimeError, EOFError) as error:
                reasons[arch] = str(error)
        chosen = choose_reusable(candidates)
        for arch in ARCHITECTURES:
            relative = 'seed_' + str(seed) + '/' + arch
            if arch not in chosen:
                rejected.append({'arm':relative,'reason':reasons.get(arch,'full sequence differs from the paired reference')})
                continue
            accepted.append(relative)
            source = old_root / 'train' / relative
            evidence[relative] = {name:file_sha(source / name) for name in
                                  ('cotracker_three_final.pth','paired-steps.jsonl','initialization-report.json','run-config.json')}
    proof = {'complete':True,'source_frozen_sha256':file_sha(old_root / 'frozen-run.json'),
             'original_image':read_json(old_root / 'training-image.json')['Id'],
             'changed_non_numerical_sources':changed,'reused_arms':accepted,'retrain_arms':rejected,
             'source_files_sha256':evidence,
             'note':'Only verified unchanged training arms are copied. Rejected arms are NOT repaired or imported. Original study is read-only. New sampler alignment leaves uninterrupted training unchanged.'}
    plan = new_root / 'recovery-plan.json'
    if plan.exists() and read_json(plan) != proof:
        raise ValueError('original recovery evidence changed; preserve both studies')
    save_json(plan, proof)
    for relative in accepted:
        source, target = old_root / 'train' / relative, new_root / 'train' / relative
        target.mkdir(parents=True, exist_ok=True)
        hashes = {('reused-run-config.json' if name == 'run-config.json' else name):value for name,value in evidence[relative].items()}
        provenance = {'original_arm':relative,'original_results':new['RecoverySource'],
                      'original_frozen_sha256':proof['source_frozen_sha256'], 'copied_files_sha256':hashes}
        provenance_path = target / 'reused-training-source.json'
        if provenance_path.exists():
            if read_json(provenance_path) != provenance or any(file_sha(target / name) != value for name,value in hashes.items()):
                raise ValueError('previously imported arm changed')
            continue
        # No material checkpoint may be replaced. A copy interrupted before the
        # manifest was committed can only be resumed if each existing file agrees.
        for name, expected_hash in hashes.items():
            original_name = 'run-config.json' if name == 'reused-run-config.json' else name
            if (target / name).exists():
                if file_sha(target / name) != expected_hash:
                    raise ValueError('refusing to overwrite a different existing artifact')
            else:
                copy_atomic(source / original_name, target / name)
        save_json(provenance_path, provenance)
    print(json.dumps({key:proof[key] for key in ('complete','reused_arms','retrain_arms')}), flush=True)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    recover(args.old, args.new, args.source)
