"""Evaluation-only identity repair. Keeps the original frozen package/manifest/jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import sysconfig
import uuid

PACKAGE = Path(__file__).resolve().parent
EVALUATOR_FILES = ('evaluate.py', 'helpers.py', 'monai_metrics.py', 'minimal_mha_simpleitk.py')
JOB_OUTPUTS = ('output/images/mri-linac-series-targets/output.mha', 'tracks.npz', 'diagnostics.json', 'prediction.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def identity():
    return {'uid': os.getuid() if hasattr(os, 'getuid') else None,
            'gid': os.getgid() if hasattr(os, 'getgid') else None}


def package_hashes(package):
    return {p.name: sha(p) for p in package.iterdir()
            if p.is_file() and p.suffix in ('.py', '.json', '.ps1', '.txt')}


def contained(root, path):
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def prepare(root, original_package, study, split, profile):
    if split not in ('validation-10', 'test-38') or not re.fullmatch(r'[A-Za-z0-9_]+', profile):
        raise ValueError('Unsafe repair scope')
    manifest_path = root / 'manifest.json'
    manifest = read(manifest_path)
    if manifest['package'] != package_hashes(original_package):
        raise ValueError('Original frozen package changed; this permission repair cannot authorize numerical/code changes')
    if sha(study / 'frozen-study.json') != manifest['study_sha256']:
        raise ValueError('Original frozen source study changed')
    if profile not in manifest['resolved']:
        raise ValueError('Unknown teacher profile')
    folder = contained(root, root / split / profile)
    output = contained(root, folder / 'evaluation')
    if not output.is_dir():
        raise ValueError('Original evaluation directory is missing')
    scope_cases = sorted({k.split('/')[1] for k in manifest['inputs'] if k.startswith(split + '/')})
    expected_count = 10 if split == 'validation-10' else 38
    if len(scope_cases) != expected_count or any(not re.fullmatch(r'[A-Za-z0-9_]+', c) for c in scope_cases):
        raise ValueError('Frozen case set is incomplete or unsafe')
    token = hashlib.sha256((sha(manifest_path) + '/' + split + '/' + profile).encode()).hexdigest()
    payloads = {'predictions.json': sha(folder / 'predictions.json')}
    for case in scope_cases:
        job = contained(root, folder / 'jobs' / case)
        verified = read(job / 'verified.json')
        if not verified.get('complete') or verified['token'] != token or set(verified['outputs']) != set(JOB_OUTPUTS):
            raise ValueError('Unverified original prediction: ' + case)
        for relative, digest in verified['outputs'].items():
            path = contained(job, job / relative)
            if sha(path) != digest:
                raise ValueError('Original prediction payload changed: ' + case + '/' + relative)
            payloads['jobs/' + case + '/' + relative] = digest
        payloads['jobs/' + case + '/verified.json'] = sha(job / 'verified.json')
    # No official result is recomputed if a verified metrics cache already exists.
    if (output / 'verified.json').exists():
        raise ValueError('A verified evaluator cache exists; resume through the original cache check instead')
    source = read(study / 'frozen-study.json')
    if source['Images']['Evaluation'] != manifest['evaluation_image']:
        raise ValueError('Official evaluator image changed')
    inventory = {r['File'].replace('\\', '/'): r['SHA256'].lower() for r in source['Source']}
    evaluator_source = {leaf: inventory['evaluation/' + leaf] for leaf in EVALUATOR_FILES}
    info = output.stat()
    uid, gid = int(info.st_uid), int(info.st_gid)
    if not 0 <= uid <= 2147483647 or not 0 <= gid <= 2147483647:
        raise ValueError('Unsupported output owner identity')
    attempt = root / 'permission-repair' / split / profile / uuid.uuid4().hex[:8]
    attempt.mkdir(parents=True, exist_ok=False)
    # Preserve any unverified/truncated metrics instead of silently losing it.
    previous = output / 'metrics.json'
    previous_digest = None
    if previous.exists():
        previous_digest = sha(previous)
        (attempt / 'previous-unverified-metrics.json').write_bytes(previous.read_bytes())
        if sha(attempt / 'previous-unverified-metrics.json') != previous_digest:
            raise ValueError('Unverified metrics snapshot failed')
    receipt = {'schema': 1, 'split': split, 'profile': profile, 'cases': scope_cases,
               'manifest_sha256': sha(manifest_path), 'original_package': manifest['package'],
               'repair_package': package_hashes(PACKAGE), 'inference_image': manifest['inference_image'],
               'evaluation_image': manifest['evaluation_image'], 'evaluator_source': evaluator_source,
               'prediction_payloads': payloads, 'owner_uid': uid, 'owner_gid': gid,
               'producer_identity': identity(), 'directory_mode': stat.S_IMODE(info.st_mode),
               'previous_unverified_metrics_sha256': previous_digest,
               'repair': 'Run ONLY the unchanged evaluator as the output-directory owner, with read-only inputs, dropped capabilities and no-new-privileges. No chmod/chown or host admin.'}
    receipt_path = attempt / 'preflight.json'
    save(receipt_path, receipt)
    save(attempt.parent / 'latest.json', {'receipt': receipt_path.relative_to(root).as_posix(),
                                       'sha256': sha(receipt_path), 'owner_uid': uid, 'owner_gid': gid})
    print(f'EVALUATOR PERMISSION PREFLIGHT {split}/{profile}: owner={uid}:{gid}, predictions={len(scope_cases)}, original manifest preserved', flush=True)
    return receipt_path


def write_probe(output):
    path = output / ('.permission-probe-' + uuid.uuid4().hex)
    # Prove actual writing BEFORE the expensive evaluator, not just os.access().
    try:
        with path.open('xb') as stream:
            stream.write(b'permission-check\n')
            stream.flush()
            os.fsync(stream.fileno())
    except PermissionError as error:
        info = output.stat()
        raise PermissionError(f'Evaluator write preflight failed: identity={identity()}, output owner={info.st_uid}:{info.st_gid}. No metric calculation started.') from error
    else:
        path.unlink()  # Only this newly-created, uniquely named temporary probe.


def evaluate(input_root, output, evaluator_root, receipt_path, manifest_path):
    receipt = read(receipt_path)
    if package_hashes(PACKAGE) != receipt['repair_package'] or sha(manifest_path) != receipt['manifest_sha256']:
        raise ValueError('Repair code or original manifest changed')
    current = identity()
    if current != {'uid': receipt['owner_uid'], 'gid': receipt['owner_gid']}:
        raise ValueError('Evaluator identity does not match recorded output owner')
    for leaf, digest in receipt['evaluator_source'].items():
        if sha(evaluator_root / leaf) != digest:
            raise ValueError('Pinned official evaluator source changed: ' + leaf)
    for relative, digest in receipt['prediction_payloads'].items():
        if sha(contained(input_root, input_root / relative)) != digest:
            raise ValueError('Original teacher output changed before evaluation: ' + relative)
    if (output / 'verified.json').exists():
        raise ValueError('Verified metrics appeared after preflight; refuse to overwrite')
    write_probe(output)
    # The directory may be writable while a stale file belongs to another UID.
    # A retained rename needs directory ownership, not chmod/chown of that file.
    existing = output / 'metrics.json'
    previous = receipt['previous_unverified_metrics_sha256']
    retained_name = None
    if existing.exists():
        if previous is None or sha(existing) != previous:
            raise ValueError('Unverified metrics changed after preflight')
        retained_name = 'metrics.unverified-' + uuid.uuid4().hex[:8] + '.json'
        existing.rename(output / retained_name)
    elif previous is not None:
        raise ValueError('Unverified metrics disappeared after preflight')
    print(f'EVALUATOR WRITE PROBE PASSED uid={current["uid"]} gid={current["gid"]}; official code unchanged', flush=True)
    # Changing UID must not hide packages installed with pip --user in the image.
    # Derive the version-specific site path rather than hardcoding Python 3.11.
    user_site = sysconfig.get_path('purelib', scheme='posix_user', vars={'userbase': '/home/user/.local'})
    env = dict(os.environ, HOME='/home/user', PYTHONUSERBASE='/home/user/.local', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(evaluator_root) + os.pathsep + user_site
    completed = subprocess.run([sys.executable, str(evaluator_root / 'evaluate.py')], cwd=str(evaluator_root), env=env, check=False)
    if completed.returncode:
        return completed.returncode
    if not (output / 'metrics.json').is_file():
        raise ValueError('Official evaluator returned success without metrics.json')
    save(output / 'permission-repair-receipt.json', {'preflight_sha256': sha(receipt_path),
         'manifest_sha256': receipt['manifest_sha256'], 'identity': current,
         'official_metrics_sha256': sha(output / 'metrics.json'), 'evaluator_source': receipt['evaluator_source'],
         'retained_unverified_metrics': retained_name,
         'numerical_changes': False, 'predictions_recomputed': False})
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', choices=['prepare', 'evaluate'], required=True)
    p.add_argument('--split', choices=['validation-10', 'test-38'], default='validation-10')
    p.add_argument('--profile', default='old')
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(Path('/results'), Path('/teacher-source'), Path('/study'), args.split, args.profile)
        return 0
    return evaluate(Path('/input'), Path('/output'), Path('/opt/app'), Path('/repair-receipt.json'), Path('/original-manifest.json'))


if __name__ == '__main__':
    raise SystemExit(main())
