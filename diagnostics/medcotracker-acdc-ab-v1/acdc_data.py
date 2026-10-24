"""Official ACDC acquisition and patient-level cine audit. No model training."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API = 'https://humanheart-project.creatis.insa-lyon.fr/database/api/v1'
TRAIN_FOLDER = '63721d7073e9f0047faa0525'
DOCUMENTS = {'LICENSE_TERMS.md': '66e290e0961576b1bad4ee3c',
             'MANDATORY_CITATION.md': '63721b4873e9f0047faa051f'}
GROUPS = ('DCM', 'HCM', 'MINF', 'NOR', 'RV')
SEED = 20261023
SCHEMA = 'medcotracker-acdc-preparation-v1'
# Acquisition schema stays unchanged: verified downloads remain reusable.
AUDIT_VERSION = 2
CLIP_FRAMES = 16
MIN_NATIVE_FRAMES = 2


def clip_length(native_frames, requested_frames=CLIP_FRAMES):
    if isinstance(requested_frames, bool) or not isinstance(requested_frames, int) or requested_frames < 2:
        raise ValueError('Requested clip length must be an integer >=2')
    if native_frames < MIN_NATIVE_FRAMES:
        raise ValueError(f'At least {MIN_NATIVE_FRAMES} native time frames required; got {native_frames}; no padding allowed')
    return min(native_frames, requested_frames)


def clip_policy():
    return {'target_frames': CLIP_FRAMES, 'minimum_native_frames': MIN_NATIVE_FRAMES,
            'length_rule': 'min(target_frames, native_frames)', 'stride': 1,
            'padding': 'forbidden', 'axes': 'T,Z,Y,X', 'batch_size': 1,
            'sample_unit': 'patient then slice then contiguous time window',
            'resize_hw': [384, 512], 'intensity': 'per-slice whole-cine p01/p99, clip 0..255 float32'}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith(API.rsplit('/api/v1', 1)[0] + '/'):
            raise ValueError('Unexpected redirect; inspect official site: ' + newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def request(endpoint):
    if not endpoint.startswith('/') or '..' in endpoint:
        raise ValueError('Invalid API endpoint')
    req = urllib.request.Request(API + endpoint, headers={'User-Agent': 'MedCoTracker-research-preparation/1'})
    return urllib.request.build_opener(OfficialRedirect()).open(req, timeout=60)


def api(endpoint):
    for attempt in range(4):
        try:
            with request(endpoint) as stream:
                return json.load(stream)
        except (urllib.error.URLError, TimeoutError) as exc:
            if getattr(exc, 'code', 0) in (401, 403, 404) or attempt == 3:
                raise
            time.sleep(2 ** attempt)


def listing(resource, **query):
    rows, offset = [], 0
    while True:
        page = api('/' + resource + '?' + urllib.parse.urlencode(
            dict(query, limit=100, offset=offset, sort='name', sortdir=1)))
        if not isinstance(page, list):
            raise ValueError('Expected public listing, not login/error response')
        rows.extend(page)
        if len(page) < 100:
            return rows
        offset += 100


def file_record(item, relative):
    if not re.fullmatch(r'[0-9a-f]{24}', item['_id']):
        raise ValueError('Invalid official item identifier')
    files = listing('item/' + item['_id'] + '/files')
    if len(files) != 1 or files[0]['name'] != item['name']:
        raise ValueError('Expected one physical file: ' + item['name'])
    f = files[0]
    if not re.fullmatch(r'[0-9a-f]{24}', f['_id']):
        raise ValueError('Invalid official file identifier')
    if int(f['size']) != int(item['size']) or int(f['size']) <= 0:
        raise ValueError('Inconsistent/empty official file')
    return {'path': relative, 'item_id': item['_id'], 'file_id': f['_id'], 'bytes': int(f['size']),
            'upstream_sha512': f.get('sha512'), 'upstream_sha256': f.get('sha256')}


def make_index(root, limit):
    index_path = root / 'download-index.json'
    if index_path.exists():
        index = read(index_path)
        if index['schema'] != SCHEMA or index['api'] != API or index['patient_limit'] != limit:
            raise ValueError('Existing index belongs to another scope; use a new DataRoot')
        return index
    folders = listing('folder', parentType='folder', parentId=TRAIN_FOLDER)
    expected = {'patient%03d' % i for i in range(1, 101)}
    if len(folders) != 100 or {f['name'] for f in folders} != expected:
        raise ValueError('Official training collection changed; expected patient001..100')
    records = []
    for name, item_id in DOCUMENTS.items():
        records.append(file_record(api('/item/' + item_id), name))
    for folder in sorted(folders, key=lambda f: f['name'])[:limit]:
        patient = folder['name']
        items = listing('item', folderId=folder['_id'])
        required = [i for i in items if i['name'] == 'Info.cfg' or
                    re.fullmatch(patient + r'_(?:4d|frame\d{2}(?:_gt)?)\.nii\.gz', i['name'])]
        if len(required) != 6 or {patient + '_4d.nii.gz', 'Info.cfg'} - {i['name'] for i in required}:
            raise ValueError('Unexpected cine/ED/ES structure: ' + patient)
        for item in required:
            records.append(file_record(item, 'training/' + patient + '/' + item['name']))
        print('INDEX ' + patient, flush=True)
    index = {'schema': SCHEMA, 'api': API, 'training_folder': TRAIN_FOLDER,
             'patient_limit': limit, 'patients': limit, 'files': records,
             'official_test_downloaded': False, 'total_bytes': sum(f['bytes'] for f in records),
             'note': 'Local hashes verify repeatability, not an upstream checksum when upstream hash is absent.'}
    save(index_path, index)
    return index


def safe_path(root, name):
    # No archive extraction and no server-controlled path outside DataRoot.
    if '\\' in name or ':' in name or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('Unsafe filename')
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    return path


def download_file(root, row):
    path = safe_path(root, row['path'])
    receipt = root / 'receipts' / (row['file_id'] + '.json')
    if path.exists():
        if not receipt.exists():
            raise ValueError('Existing file has no receipt; preserve it and use a fresh DataRoot: ' + str(path))
        r = read(receipt)
        if r['source'] != row or path.stat().st_size != row['bytes'] or digest(path) != r['sha256']:
            raise ValueError('Previously downloaded file changed: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    for attempt in range(4):
        try:
            with request('/file/' + row['file_id'] + '/download') as source, tmp.open('wb') as target:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            if tmp.stat().st_size != row['bytes']:
                raise OSError('Incomplete download: ' + row['path'])
            for algorithm in ('sha256', 'sha512'):
                upstream = row.get('upstream_' + algorithm)
                if upstream and digest(tmp, algorithm) != upstream:
                    raise ValueError('Official checksum mismatch: ' + row['path'])
            # Receipt first: a crash before rename can safely retry the partial file.
            save(receipt, {'source': row, 'sha256': digest(tmp)})
            os.replace(tmp, path)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if getattr(exc, 'code', 0) in (401, 403, 404) or attempt == 3:
                raise
            time.sleep(2 ** attempt)


def download(root, limit, accepted):
    if not accepted:
        raise ValueError('Confirm non-commercial scientific use and CC BY-NC-SA 4.0 with --accept-research-terms')
    root.mkdir(parents=True, exist_ok=True)
    index = make_index(root, limit)
    save(root / 'research-use.json', {'confirmed_by_operator': True, 'license': 'CC BY-NC-SA 4.0',
                                     'purpose': 'non-commercial scientific research', 'source': API})
    for n, row in enumerate(index['files'], 1):
        download_file(root, row)
        if row['path'].endswith('_4d.nii.gz') or n == len(index['files']):
            print(f"VERIFIED FILE {n}/{len(index['files'])} {row['path']}", flush=True)
    save(root / 'download-complete.json', {'complete': True, 'patients': limit,
         'files': len(index['files']), 'download_index_sha256': digest(root / 'download-index.json'),
         'official_test_downloaded': False})


def parse_info(text):
    fields = {}
    for line in text.splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            if k.strip() in fields:
                raise ValueError('Duplicate Info.cfg field')
            fields[k.strip()] = v.strip()
    if fields.get('Group') not in GROUPS:
        raise ValueError('Unknown/missing diagnostic group')
    return fields


def split_patients(patients):
    if len(patients) != 100 or len({r['patient'] for r in patients}) != 100:
        raise ValueError('Formal preparation requires all 100 unique official training patients')
    if Counter(p['group'] for p in patients) != Counter({g: 20 for g in GROUPS}):
        raise ValueError('Expected 20 distinct patients per disease group')
    # Hash ordering independent of Python/NumPy PRNG version and input file order.
    splits = {}
    for group in GROUPS:
        rows = sorted((r for r in patients if r['group'] == group),
                      key=lambda r: hashlib.sha256(f"{SEED}:{r['patient']}".encode()).hexdigest())
        for i, row in enumerate(rows):
            splits[row['patient']] = 'external-dev-20' if i < 4 else 'external-train-80'
    return splits


def audit_patient(folder):
    import numpy as np
    import SimpleITK as sitk
    patient = folder.name
    info = parse_info((folder / 'Info.cfg').read_text(encoding='utf-8-sig'))
    path = folder / (patient + '_4d.nii.gz')
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 4 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Require scalar X,Y,Z,T NIfTI, not ED/ES-only or vector image: ' + patient)
    array = sitk.GetArrayFromImage(image)
    if array.shape != tuple(reversed(image.GetSize())) or not np.isfinite(array).all():
        raise ValueError('Invalid T,Z,Y,X cine: ' + patient)
    frames, slices, height, width = array.shape
    actual_clip_frames = clip_length(frames)
    if slices < 1 or min(height, width) < 16:
        raise ValueError(f'Invalid spatial shape {array.shape}: {patient}')
    if np.ptp(array) <= 0:
        raise ValueError('Constant-intensity cine: ' + patient)
    if 'NbFrame' in info and int(info['NbFrame']) != frames:
        raise ValueError('Info.cfg frame count and NIfTI T disagree: ' + patient)
    spacing = image.GetSpacing()
    if not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError('Invalid image spacing')
    annotated = {}
    for phase in ('ED', 'ES'):
        number = int(info[phase])
        if not 1 <= number <= frames:
            raise ValueError('ED/ES index out of range')
        stem = f'{patient}_frame{number:02d}'
        ref = sitk.ReadImage(str(folder / (stem + '.nii.gz')))
        mask = sitk.ReadImage(str(folder / (stem + '_gt.nii.gz')))
        for other in (ref, mask):
            if other.GetDimension() != 3 or other.GetSize() != image.GetSize()[:3]:
                raise ValueError('Annotation geometry mismatch: ' + stem)
            if not np.allclose(other.GetSpacing(), spacing[:3], atol=1e-5):
                raise ValueError('Annotation spacing mismatch: ' + stem)
            if not np.allclose(other.GetOrigin(), image.GetOrigin()[:3], atol=1e-4):
                raise ValueError('Annotation origin mismatch: ' + stem)
            if not np.allclose(np.array(other.GetDirection()).reshape(3, 3),
                               np.array(image.GetDirection()).reshape(4, 4)[:3, :3], atol=1e-5):
                raise ValueError('Annotation orientation mismatch: ' + stem)
        if not np.allclose(sitk.GetArrayFromImage(ref), array[number-1], rtol=1e-5, atol=1e-4):
            raise ValueError('ED/ES does not match cine time index: ' + stem)
        labels = np.unique(sitk.GetArrayFromImage(mask))
        if not set(labels.tolist()).issubset({0, 1, 2, 3}) or len(labels) < 2:
            raise ValueError('Invalid/missing ED/ES labels: ' + stem)
        annotated[phase] = number - 1
    return {'patient': patient, 'group': info['Group'], 'shape_tzyx': list(array.shape),
            'clip_frames': actual_clip_frames, 'short_native_cine': frames < CLIP_FRAMES,
            'spacing_xyzt': list(spacing), 'annotation_frames_zero_based': annotated,
            'cine': 'training/' + patient + '/' + path.name,
            'files_sha256': {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file() and not p.name.endswith('.partial')},
            'time_axis_verified_against_ed_es': True}


def prepare(root, output, sample_only=False):
    completion = read(root / 'download-complete.json')
    index = read(root / 'download-index.json')
    if not completion['complete'] or completion['download_index_sha256'] != digest(root / 'download-index.json'):
        raise ValueError('Incomplete/changed acquisition')
    for row in index['files']:
        path = safe_path(root, row['path'])
        receipt = read(root / 'receipts' / (row['file_id'] + '.json'))
        if receipt['source'] != row or path.stat().st_size != row['bytes'] or digest(path) != receipt['sha256']:
            raise ValueError('Source file/receipt changed: ' + str(path))
    patients = []
    for folder in sorted((root / 'training').glob('patient*')):
        if not re.fullmatch(r'patient(?:00[1-9]|0[1-9]\d|100)', folder.name):
            raise ValueError('Unexpected patient; official test-50 must not enter pretraining')
        patients.append(audit_patient(folder))
        row = patients[-1]
        print(f"AUDIT {folder.name} native_frames={row['shape_tzyx'][0]} clip_frames={row['clip_frames']}", flush=True)
    splits = {} if sample_only else split_patients(patients)
    for row in patients:
        row['split'] = splits.get(row['patient'], 'audit-only-not-for-training')
    manifest = {'schema': SCHEMA, 'audit_version': AUDIT_VERSION, 'complete': True, 'training_ready': not sample_only,
                'seed': SEED, 'download_index_sha256': digest(root / 'download-index.json'),
                'patients': patients, 'split_counts': dict(Counter(r['split'] for r in patients)),
                'clip': clip_policy(),
                'native_frame_counts': dict(sorted(Counter(str(r['shape_tzyx'][0]) for r in patients).items())),
                'short_native_cines': [{'patient': r['patient'], 'native_frames': r['shape_tzyx'][0],
                                       'clip_frames': r['clip_frames'], 'split': r['split']}
                                      for r in patients if r['short_native_cine']],
                'labels': 'ED/ES masks only; no dense point truth; no mask loss in A/B pilot',
                'official_test_downloaded': False}
    dest = output / ('sample-audit.json' if sample_only else 'data-manifest.json')
    if dest.exists() and read(dest) != manifest:
        raise ValueError('Audit changed; keep old results and use new OutputRoot')
    save(dest, manifest)
    print('ACDC DATA AUDIT COMPLETE ' + str(dest), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('download', 'prepare', 'sample-audit'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('/output'))
    p.add_argument('--patients', type=int, choices=(1, 100), default=100)
    p.add_argument('--accept-research-terms', action='store_true')
    a = p.parse_args()
    if a.stage == 'download':
        download(a.root, a.patients, a.accept_research_terms)
    else:
        prepare(a.root, a.output, a.stage == 'sample-audit')
