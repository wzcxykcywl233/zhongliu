"""Float32-only repair for the frozen teacher pilot's norm arm; raw arms unchanged."""
import argparse
import hashlib
import importlib
from pathlib import Path
import sys

ORIGINAL_PACKAGE = Path('/teacher-source')
DIAGNOSTIC_KEY = 'norm_dtype_repair'


def original_modules(package):
    package = package.resolve()
    for name in ('common', 'worker', 'predictor'):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve().parent != package:
            raise ValueError('Unexpected teacher module already imported: ' + name)
    sys.path.insert(0, str(package))
    return importlib.import_module('common'), importlib.import_module('worker')


def corrected_preprocess(array, intensity, target_shape, original, evidence=None):
    """Keep the original percentile formula; cast BEFORE its original raw resize."""
    if intensity != 'robust_1_99':
        return original(array, intensity, target_shape)
    import numpy as np
    import torch
    a = array.astype(np.float32, copy=True)
    low, high = np.percentile(a, (1, 99))
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError('Non-identifiable intensity bounds')
    normalized = np.clip((a - low) / (high - low), 0, 1) * 255
    normalized = np.ascontiguousarray(normalized, dtype=np.float32)
    result = original(normalized, 'raw', target_shape)
    if result.dtype != torch.float32 or not torch.isfinite(result).all():
        raise ValueError('Normalized teacher input must be finite float32')
    if evidence is not None:
        evidence.clear()
        evidence.update(input_array_dtype=str(array.dtype), percentile_bounds_dtype=str(np.asarray(low).dtype),
                        percentile_low=float(low), percentile_high=float(high),
                        resize_input_dtype=str(normalized.dtype), output_tensor_dtype=str(result.dtype),
                        video_sha256=hashlib.sha256(result.contiguous().numpy().tobytes()).hexdigest())
    return result


def norm_scope(common, output, split):
    manifest_path = output / 'manifest.json'
    manifest = common.read(manifest_path)
    if common.package_hashes() != manifest['package']:
        raise ValueError('Original frozen teacher package changed')
    configs = common.graph()[2]
    if manifest['resolved'].get('norm') != configs['norm'] or configs['norm']['intensity'] != 'robust_1_99':
        raise ValueError('Unexpected norm configuration')
    cases = sorted({key.split('/')[1] for key in manifest['inputs'] if key.startswith(split + '/')})
    if len(cases) != common.SPLITS[split]:
        raise ValueError('Norm repair requires the original complete case set')
    import numpy as np
    import torch
    proof = {'schema': 1, 'split': split, 'profile': 'norm', 'cases': cases,
             'manifest_sha256': common.sha(manifest_path), 'original_package': manifest['package'],
             'wrapper_sha256': common.sha(Path(__file__)), 'numpy_version': np.__version__,
             'torch_version': torch.__version__, 'percentiles': [1, 99], 'scale': 255,
             'repair': 'Unchanged percentile arithmetic, explicit float32 before unchanged raw bilinear resize',
             'output_dtype': 'torch.float32', 'weights_queries_tracking_and_render_unchanged': True}
    profile = output / split / 'norm'
    proof_path = output / 'dtype-repair' / split / 'norm.json'
    return manifest, profile, proof_path, proof


def validate_norm_jobs(common, output, split, require_complete=False):
    manifest, profile, proof_path, proof = norm_scope(common, output, split)
    if not proof_path.is_file() or common.read(proof_path) != proof:
        raise ValueError('Norm dtype repair proof missing or changed; preserve results, do not mix caches')
    token = hashlib.sha256((common.sha(output / 'manifest.json') + '/' + split + '/norm').encode()).hexdigest()
    required = [common.MASK, 'tracks.npz', 'diagnostics.json', 'prediction.json']
    for case in proof['cases']:
        folder = profile / 'jobs' / case
        if not common.verify_job(folder, token, required):
            if require_complete:
                raise ValueError('Incomplete repaired norm case: ' + case)
            continue
        diagnostic = common.read(folder / 'diagnostics.json')
        details = diagnostic.get(DIAGNOSTIC_KEY, {})
        if (details.get('proof_sha256') != common.sha(proof_path) or
                details.get('resize_input_dtype') != 'float32' or
                details.get('output_tensor_dtype') != 'torch.float32' or
                diagnostic.get('config') != manifest['resolved']['norm']):
            raise ValueError('Norm cache lacks matching float32 provenance: ' + case)
    return profile, proof_path, proof


def prepare_norm(common, output, split):
    _, profile, proof_path, proof = norm_scope(common, output, split)
    if not proof_path.exists():
        if any((profile / 'jobs').glob('*/verified.json')) or (profile / 'evaluation/verified.json').exists():
            raise ValueError('Existing norm cache has no dtype provenance; use a separate study, never overwrite')
        common.save(proof_path, proof)
    validate_norm_jobs(common, output, split)
    return profile, proof_path, proof


def run(argv=None, package=ORIGINAL_PACKAGE):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--action', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['validation-10', 'test-38'], default='validation-10')
    parser.add_argument('--profile', default='old')
    parser.add_argument('--group', default='weights')
    args, _ = parser.parse_known_args(argv)
    common, worker = original_modules(package)
    previous_argv = sys.argv
    sys.argv = [str(package / 'worker.py'), *argv]
    try:
        if args.action in ('select', 'report'):
            names = (common.read(args.output / 'selection.json')['test_profiles'] if args.split == 'test-38'
                     else common.group_profiles(args.group))
            if 'norm' in names:
                validate_norm_jobs(common, args.output, 'validation-10' if args.action == 'select' else args.split, True)
        if args.action == 'record' and args.profile == 'norm':
            validate_norm_jobs(common, args.output, args.split, True)
        if args.action != 'infer' or args.profile != 'norm':
            return worker.main()

        if args.split == 'test-38':
            validate_norm_jobs(common, args.output, 'validation-10', True)
        profile, proof_path, _ = prepare_norm(common, args.output, args.split)
        predictor = importlib.import_module('predictor')
        original_preprocess, original_commit = predictor.preprocess, worker.commit_job
        evidence = {}

        def preprocess(array, intensity, target_shape=(384, 512)):
            if intensity != 'robust_1_99':
                raise ValueError('Norm repair must not silently change another intensity arm')
            return corrected_preprocess(array, intensity, target_shape, original_preprocess, evidence)

        def commit(folder, token, required):
            # Only newly created norm attempts are annotated; completed payloads are never edited.
            folder.resolve().relative_to((profile / '.attempts').resolve())
            if not evidence or evidence['output_tensor_dtype'] != 'torch.float32':
                raise ValueError('Missing actual float32 preprocessing evidence')
            diagnostic = common.read(folder / 'diagnostics.json')
            if diagnostic['profile'] != 'norm' or diagnostic['config']['intensity'] != 'robust_1_99':
                raise ValueError('Unexpected case committed by norm repair')
            diagnostic[DIAGNOSTIC_KEY] = {'proof_sha256': common.sha(proof_path), **evidence}
            common.save(folder / 'diagnostics.json', diagnostic)
            original_commit(folder, token, required)
            evidence.clear()

        predictor.preprocess, worker.commit_job = preprocess, commit
        print('NORM FLOAT32 REPAIR ACTIVE: original package/manifest preserved; raw arms unchanged', flush=True)
        try:
            result = worker.main()
            validate_norm_jobs(common, args.output, args.split, True)
            return result
        finally:
            predictor.preprocess, worker.commit_job = original_preprocess, original_commit
    finally:
        sys.argv = previous_argv


if __name__ == '__main__':
    run()
