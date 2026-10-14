"""Frozen, teacher-only OFAT graph and verified cache utilities (no ML imports)."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath

PACKAGE = Path(__file__).resolve().parent
SPLITS = {'train-40': 40, 'validation-10': 10, 'test-38': 38}
METRICS = {'DSC': 'dice_similarity_coefficient', 'HD95': 'hausdorff_distance_95',
           'MASD': 'surface_distance_average', 'CD': 'center_distance', 'D98': 'relative_d98_dose'}
MASK = 'output/images/mri-linac-series-targets/output.mha'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def csv_save(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row)) or ['no_records']
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def package_hashes():
    return {p.name: sha(p) for p in PACKAGE.iterdir()
            if p.is_file() and p.suffix in ('.py', '.json', '.ps1', '.txt')}


def graph(spec=None):
    spec = spec or read(PACKAGE / 'experiments.json')
    resolved, profiles = {}, {}
    for row in spec['profiles']:
        name, parent, change = row['name'], row['parent'], row['change']
        if name in profiles or not name.replace('_', '').isalnum():
            raise ValueError('Duplicate/unsafe profile: ' + name)
        if parent is None:
            if resolved or change:
                raise ValueError('Exactly one unchanged root is required')
            config = dict(spec['base'])
        else:
            if parent not in resolved or len(change) != 1 or not set(change) <= set(spec['base']):
                raise ValueError('Each child must change exactly one known factor')
            config = {**resolved[parent], **change}
            if sum(config[k] != resolved[parent][k] for k in config) != 1:
                raise ValueError('No-op or multiple-factor child: ' + name)
        if config['checkpoint'] not in ('baseline_offline.pth', 'scaled_offline.pth'):
            raise ValueError('Only unmodified offline teachers are allowed')
        resolved[name], profiles[name] = config, row
    return spec, profiles, resolved


def group_profiles(group):
    _, profiles, _ = graph()
    if group != 'all' and group not in {p['group'] for p in profiles.values()}:
        raise ValueError('Unknown experiment group')
    needed = {n for n, p in profiles.items() if group == 'all' or p['group'] == group}
    for name in list(needed):
        parent = profiles[name]['parent']
        while parent:
            needed.add(parent)
            parent = profiles[parent]['parent']
    return [n for n in profiles if n in needed]


def protocol_files(protocol):
    groups = {s: {} for s in SPLITS}
    for row in protocol['Dataset']:
        split, case = row['Split'], row['Case']
        if split not in groups:
            raise ValueError('Unknown protocol split: ' + split)
        parts = PureWindowsPath(row['File']).parts
        indices = [i for i, part in enumerate(parts) if part == case]
        if len(indices) != 1:
            raise ValueError('Ambiguous dataset path')
        relative = Path(*parts[indices[0]:]).as_posix()
        if '..' in Path(relative).parts or relative in groups[split]:
            raise ValueError('Unsafe/duplicate dataset entry')
        groups[split][relative] = row['SHA256'].lower()
    ids = {s: {k.split('/')[0] for k in rows} for s, rows in groups.items()}
    if any(len(ids[s]) != count for s, count in SPLITS.items()) or len(set.union(*ids.values())) != 88:
        raise ValueError('Expected disjoint 40/10/38 case IDs')
    return groups


def inventory(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def verify_job(folder, token, required=None):
    marker = folder / 'verified.json'
    if not marker.exists():
        return False
    recorded = read(marker)
    if not recorded.get('complete') or recorded['token'] != token:
        raise ValueError('Cache belongs to another run: ' + str(folder))
    if required is not None and set(recorded['outputs']) != set(required):
        raise ValueError('Completed output inventory is incomplete: ' + str(folder))
    for relative, digest in recorded['outputs'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Unsafe completed output path')
        if sha(folder / relative) != digest:
            raise ValueError('Completed output changed: ' + relative)
    return True


def commit_job(folder, token, required):
    # Only named outputs count; interrupted .tmp files are never trusted.
    save(folder / 'verified.json', {'complete': True, 'token': token,
                                 'outputs': {p: sha(folder / p) for p in required}})


def scalar(path, expected):
    value = read(path)
    if isinstance(value, dict) and set(value) == {'value'}:
        value = value['value']
    if expected == 'number':
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid numeric metadata: ' + str(path))
    elif not isinstance(value, str) or not value:
        raise ValueError('Invalid region metadata: ' + str(path))
    return value


def official(path, cases):
    data = read(path)
    rows = data['results']
    actual = [r['case_id'] for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(cases):
        raise ValueError('Official evaluator cases are incomplete or mismatched')
    result = {k: float(data['aggregates'][v]) for k, v in METRICS.items()}
    result['TimeSec'] = float(data['aggregates']['total_time'])
    if not all(math.isfinite(x) for x in result.values()):
        raise ValueError('Official aggregate contains nonfinite metrics')
    for row in rows:
        if not all(math.isfinite(float(row[v])) for v in METRICS.values()):
            raise ValueError('Invalid official case metrics')
    return result, {r['case_id']: r for r in rows}
