"""Narrow, reviewed compatibility for historical full-study *packaging* files.

The old full study may predate ddd8004 (output-path length repair). That commit
changed no Python training/data/model code: only README paths, the launcher's
default OutputRoot/error text, and the test of that path guard. Do NOT ignore
documentation or launcher changes in general. Both byte hashes must be members
of the exact reviewed transition below. LF/CRLF variants are explicitly pinned;
there is no whitespace/BOM normalization, Git invocation or network at runtime.
All other files must still match the historical SHA256 byte-for-byte.
"""
import hashlib

PREFIX = 'diagnostics/medcotracker-acdc-full-v1/'
BEFORE = '70f795c'
AFTER = 'ddd8004'
REVIEWED = {
    PREFIX + 'README.txt': {
        BEFORE: ('5c5e30ae39468f61a6caa515134cca320dfc7cedde98ebdc19ae5ab0ea886da7',
                 '0247805ad7c38490ef48a4c95dacbca207e1340c1947230f8420c7d9c0500c1f'),
        AFTER: ('76a3c1557c45916cf77dc04f6d5531833d350ad4e10fe42866d0bdc294575647',
                '1e1311445b95e8d432c8126aac30041689ad2cca0f17ab47dad08e0b9e2222ac')},
    PREFIX + 'run_experiment.ps1': {
        BEFORE: ('987f9a172dafe979aea086af68f775efd57b06d9d0478266b060057040d113d0',
                 '9f9262a1b12922084601b7e53746135eb85cb78d946039d2d5a0fce9577f12a1'),
        AFTER: ('e97d893fa7cfefa5098734f2fd6678c83a4c6b2b7c1ca9a586107d4268209d6e',
                'b149b7b642c5988e0ce29122fe1f2b99fae60a54ace514b54eeb3aa03f92fe5d')},
    PREFIX + 'test_runner.ps1': {
        BEFORE: ('04c31d00a388b7b5e4cac2955e277495d2e91eec0b9978a34edfdde7963a35bb',
                 '8973719976c85f932c351c4f4ef837681631301367da6b89e080dfc96f39cbbe'),
        AFTER: ('3b86b6cc2823421ab5256ff43783117ce13543bfee87fafa94debb84299c7d70',
                'e6248deeeb37e2a1b2360939d242004a69fea34acdcfb29f38aaa46cdc1f3bb3')},
}


def verify_historical_sources(root, expected_files, safe_file):
    compatible, errors = [], []
    for relative, expected in sorted(expected_files.items()):
        # Same path-containment guard as strict artifact verification.
        path = safe_file(root, relative)
        if not path.is_file():
            errors.append(f'{relative}: missing (expected={expected})')
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual == expected:
            continue
        versions = REVIEWED.get(relative, {})
        old_version = next((v for v, hashes in versions.items() if expected in hashes), None)
        new_version = next((v for v, hashes in versions.items() if actual in hashes), None)
        if old_version and new_version and (old_version == new_version or (old_version, new_version) == (BEFORE, AFTER)):
            compatible.append({'file': relative, 'expected_sha256': expected, 'actual_sha256': actual,
                               'historical_version': old_version, 'checkout_version': new_version,
                               'reason': 'reviewed_output_path_fix' if old_version != new_version else 'reviewed_LF_CRLF_only'})
        else:
            errors.append(f'{relative}: expected={expected} actual={actual}')
    if errors:
        raise ValueError('Historical source mismatch; unreviewed changes blocked:\n' + '\n'.join(errors))
    for row in compatible:
        print(f'PLAN BASELINE SOURCE COMPAT {row["file"]}: {row["reason"]} '
              f'{row["historical_version"]}->{row["checkout_version"]}', flush=True)
    return {'schema': 1, 'checked_files': len(expected_files), 'accepted_compatibilities': compatible,
            'policy': 'Exact bytes except pinned full-study path-fix packaging files; no model/data/weight/label exception.'}
