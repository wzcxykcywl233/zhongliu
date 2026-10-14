"""Evaluation-only identity repair. Keeps the original frozen package/manifest/jobs."""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import uuid

PACKAGE = Path(__file__).resolve().parent
EVALUATOR_FILES = ('evaluate.py', 'helpers.py', 'monai_metrics.py', 'minimal_mha_simpleitk.py')
JOB_OUTPUTS = ('output/images/mri-linac-series-targets/output.mha', 'tracks.npz', 'diagnostics.json', 'prediction.json')
RUNTIME_TAG = 'TEACHER_EVALUATOR_RUNTIME '
METRICS_TAG = 'TEACHER_METRICS_JSON '
DEPENDENCIES = ('numpy', 'scipy', 'pandas', 'psutil')


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


def runtime(evaluator_root):
    """Run as the image's ORIGINAL user; discover, never guess, its dependencies."""
    modules = {}
    for name in DEPENDENCIES:
        module = importlib.import_module(name)
        file = Path(module.__file__).resolve()
        modules[name] = {'file': str(file), 'sha256': sha(file), 'version': str(getattr(module, '__version__', 'unknown'))}
    result = {'schema': 2, 'identity': identity(), 'python': sys.executable,
              'python_version': sys.version, 'home': os.environ.get('HOME'),
              'sys_path': list(sys.path), 'modules': modules,
              'evaluator_source': {leaf: sha(evaluator_root / leaf) for leaf in EVALUATOR_FILES}}
    print(RUNTIME_TAG + json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
    return result


def prepare(root, original_package, study, split, profile, runtime_path):
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
    original_runtime = read(runtime_path)
    if original_runtime['evaluator_source'] != evaluator_source or set(original_runtime['modules']) != set(DEPENDENCIES):
        raise ValueError('Original evaluator runtime audit is incomplete or source changed')
    for value in original_runtime['identity'].values():
        if not isinstance(value, int) or not 0 <= value <= 2147483647:
            raise ValueError('Original evaluator user identity unavailable')
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
    receipt = {'schema': 2, 'split': split, 'profile': profile, 'cases': scope_cases,
               'manifest_sha256': sha(manifest_path), 'original_package': manifest['package'],
               'repair_package': package_hashes(PACKAGE), 'inference_image': manifest['inference_image'],
               'evaluation_image': manifest['evaluation_image'], 'evaluator_source': evaluator_source,
               'original_evaluator_runtime': original_runtime,
               'prediction_payloads': payloads, 'owner_uid': uid, 'owner_gid': gid,
               'producer_identity': identity(), 'directory_mode': stat.S_IMODE(info.st_mode),
               'previous_unverified_metrics_sha256': previous_digest,
               'repair': 'Calculate with the unchanged evaluator and original image user/site packages in ephemeral output; publish byte-identical metrics with the output owner. No chmod/chown/install or host admin.'}
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
    if current != receipt['original_evaluator_runtime']['identity']:
        raise ValueError('Evaluator must use its original image user, not the prediction-directory owner')
    for leaf, digest in receipt['evaluator_source'].items():
        if sha(evaluator_root / leaf) != digest:
            raise ValueError('Pinned official evaluator source changed: ' + leaf)
    for relative, digest in receipt['prediction_payloads'].items():
        if sha(contained(input_root, input_root / relative)) != digest:
            raise ValueError('Original teacher output changed before evaluation: ' + relative)
    for name, expected in receipt['original_evaluator_runtime']['modules'].items():
        module = importlib.import_module(name)
        if str(Path(module.__file__).resolve()) != expected['file'] or sha(module.__file__) != expected['sha256']:
            raise ValueError('Original evaluator dependency changed: ' + name)
    if (output / 'verified.json').exists():
        raise ValueError('Verified metrics appeared after preflight; refuse to overwrite')
    write_probe(output)
    if (output / 'metrics.json').exists():
        raise ValueError('Ephemeral evaluator output must start empty')
    print(f'EVALUATOR ORIGINAL RUNTIME PASSED uid={current["uid"]} gid={current["gid"]}; psutil and numerical dependencies verified', flush=True)
    # Preserve original HOME, user-site, PYTHONPATH and numerical libraries.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    completed = subprocess.run([sys.executable, str(evaluator_root / 'evaluate.py')], cwd=str(evaluator_root), env=env, check=False)
    if completed.returncode:
        return completed.returncode
    if not (output / 'metrics.json').is_file():
        raise ValueError('Official evaluator returned success without metrics.json')
    data = (output / 'metrics.json').read_bytes()
    # ASCII JSON envelope carries DATA, not an encoded command. The receiver
    # checks SHA256 and writes the original UTF-8 bytes without reformatting.
    message = {'schema': 2, 'type': 'teacher-official-metrics', 'preflight_sha256': sha(receipt_path),
               'manifest_sha256': receipt['manifest_sha256'], 'evaluation_image': receipt['evaluation_image'],
               'identity': current, 'metrics_sha256': hashlib.sha256(data).hexdigest(),
               'metrics_json': data.decode('utf-8')}
    print(METRICS_TAG + json.dumps(message, ensure_ascii=True, allow_nan=False), flush=True)
    return 0


def publish(root, receipt_path, response_path):
    """Stdlib-only writer; never imports/installs/replaces evaluator libraries."""
    receipt, response = read(receipt_path), read(response_path)
    if package_hashes(PACKAGE) != receipt['repair_package'] or sha(root / 'manifest.json') != receipt['manifest_sha256']:
        raise ValueError('Repair code or original manifest changed before publication')
    if identity() != {'uid': receipt['owner_uid'], 'gid': receipt['owner_gid']}:
        raise ValueError('Metrics publisher must match the output-directory owner')
    if response['schema'] != 2 or response['type'] != 'teacher-official-metrics' or response['preflight_sha256'] != sha(receipt_path):
        raise ValueError('Invalid or stale evaluator response')
    for key in ('manifest_sha256', 'evaluation_image'):
        if response[key] != receipt[key]:
            raise ValueError('Evaluator response provenance changed: ' + key)
    if response['identity'] != receipt['original_evaluator_runtime']['identity']:
        raise ValueError('Evaluator did not use original image identity')
    data = response['metrics_json'].encode('utf-8')
    if hashlib.sha256(data).hexdigest() != response['metrics_sha256']:
        raise ValueError('Metrics transport checksum mismatch')
    metrics = json.loads(data)
    keys = ('dice_similarity_coefficient', 'hausdorff_distance_95', 'surface_distance_average', 'center_distance', 'relative_d98_dose')
    cases = [item['case_id'] for item in metrics['results']]
    if len(cases) != len(set(cases)) or set(cases) != set(receipt['cases']):
        raise ValueError('Official metrics case set mismatch')
    if not all(math.isfinite(float(metrics['aggregates'][k])) for k in (*keys, 'total_time')):
        raise ValueError('Invalid official aggregate metrics')
    for item in metrics['results']:
        if not all(math.isfinite(float(item[k])) for k in keys):
            raise ValueError('Invalid official case metrics')
    folder = contained(root, root / receipt['split'] / receipt['profile'])
    for relative, digest in receipt['prediction_payloads'].items():
        if sha(contained(folder, folder / relative)) != digest:
            raise ValueError('Original teacher prediction changed during evaluation')
    output = contained(root, folder / 'evaluation')
    if (output / 'verified.json').exists():
        raise ValueError('Verified metrics already exist; refuse to overwrite')
    write_probe(output)
    existing = output / 'metrics.json'
    retained_name = None
    if existing.exists():
        if sha(existing) != receipt['previous_unverified_metrics_sha256']:
            raise ValueError('Unverified metrics changed after preflight')
        retained_name = 'metrics.unverified-' + uuid.uuid4().hex[:8] + '.json'
        existing.rename(output / retained_name)
    elif receipt['previous_unverified_metrics_sha256'] is not None:
        raise ValueError('Unverified metrics disappeared after preflight')
    temp = output / ('metrics.publish-' + uuid.uuid4().hex[:8] + '.tmp')
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if sha(temp) != response['metrics_sha256']:
        raise ValueError('Metrics write checksum mismatch')
    temp.replace(existing)
    save(output / 'permission-repair-receipt.json', {'schema': 2, 'preflight_sha256': sha(receipt_path),
         'manifest_sha256': receipt['manifest_sha256'], 'evaluator_identity': response['identity'],
         'publisher_identity': identity(), 'official_metrics_sha256': sha(existing),
         'evaluator_source': receipt['evaluator_source'], 'original_evaluator_runtime': receipt['original_evaluator_runtime'],
         'retained_unverified_metrics': retained_name, 'metrics_byte_identical': True,
         'numerical_changes': False, 'predictions_recomputed': False})
    print('OFFICIAL METRICS PUBLISHED BYTE-IDENTICAL ' + receipt['split'] + '/' + receipt['profile'], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', choices=['runtime', 'prepare', 'evaluate', 'publish'], required=True)
    p.add_argument('--split', choices=['validation-10', 'test-38'], default='validation-10')
    p.add_argument('--profile', default='old')
    args = p.parse_args()
    if args.action == 'runtime':
        runtime(Path('/opt/app'))
        return 0
    if args.action == 'prepare':
        prepare(Path('/results'), Path('/teacher-source'), Path('/study'), args.split, args.profile, Path('/evaluator-runtime.json'))
        return 0
    if args.action == 'publish':
        publish(Path('/results'), Path('/repair-receipt.json'), Path('/metrics-response.json'))
        return 0
    return evaluate(Path('/input'), Path('/output'), Path('/opt/app'), Path('/repair-receipt.json'), Path('/original-manifest.json'))


if __name__ == '__main__':
    raise SystemExit(main())
