"""One-patient ablation; preserve historical non-082 clips and teacher draws."""
from collections import Counter
import random
import numpy as np

EXCLUDED = 'patient082'
POLICY = {
    'excluded_patients': [EXCLUDED], 'eligible_external_train_patients': 79,
    'external_dev_patients': 20, 'schedule': 'original 80-slot epochs',
    'replacement': 'same-group remaining train patients, seeded balanced rotation',
    'unchanged_steps': 'exact original video and cached teacher labels',
    'duplicate_074_076': 'unchanged; not an independent external development split',
    'scope': 'entire patient, all slices and frames; no source files removed',
}


def inventory(manifest):
    rows = manifest['patients']
    if len(rows) != 100 or len({r['patient'] for r in rows}) != 100:
        raise ValueError('Expected original 100 unique patient records')
    train = sorted((r for r in rows if r['split'] == 'external-train-80'), key=lambda r: r['patient'])
    dev = sorted(r['patient'] for r in rows if r['split'] == 'external-dev-20')
    if len(train) != 80 or len(dev) != 20 or EXCLUDED not in {r['patient'] for r in train}:
        raise ValueError('Do not re-split: patient082 must belong to original train-80')
    target = next(r for r in train if r['patient'] == EXCLUDED)
    if target['group'] != 'RV' or Counter(r['group'] for r in train) != Counter({g: 16 for g in ('DCM', 'HCM', 'MINF', 'NOR', 'RV')}):
        raise ValueError('Original stratified training inventory changed')
    candidates = [i for i, r in enumerate(train) if r['group'] == target['group'] and r['patient'] != EXCLUDED]
    return train, dev, candidates


def replacement_index(train, candidates, epoch, index, seed):
    if train[index]['patient'] != EXCLUDED:
        return index
    cycle, offset = divmod(epoch, len(candidates))
    # Independent generator: does not consume teacher/query/model RNG streams.
    return int(np.random.default_rng(seed + 820000 + cycle).permutation(candidates)[offset])


def make_schedule(manifest, seed, steps=3000):
    train, dev, candidates = inventory(manifest)
    rows = []
    for step in range(steps):
        epoch, offset = divmod(step, 80)
        index = int(np.random.default_rng(seed + epoch).permutation(80)[offset])
        chosen = replacement_index(train, candidates, epoch, index, seed)
        rows.append({'step': step, 'epoch': epoch, 'original_index': index, 'index': chosen,
                     'original_patient': train[index]['patient'], 'patient': train[chosen]['patient'],
                     'slot': random.Random(seed + 7919 * step).randrange(3),
                     'replaced': index != chosen})
    eligible = [r['patient'] for r in train if r['patient'] != EXCLUDED]
    assert EXCLUDED not in {r['patient'] for r in rows}
    return {'policy': POLICY, 'seed': seed, 'steps': steps, 'eligible_train': eligible,
            'unchanged_dev': dev, 'replacement_steps': sum(r['replaced'] for r in rows),
            'exposure_counts': dict(sorted(Counter(r['patient'] for r in rows).items())), 'rows': rows}


def dataset_class(base):
    class ExcludingDataset(base):
        def __init__(self, root, manifest, seed):
            # Original loader checks source hashes; it never decodes a cine in __init__.
            super().__init__(root, manifest, seed=seed)
            train, _, self.candidates = inventory(self.manifest)
            if self.rows != train:
                raise ValueError('Original loader order changed')

        def __getitem__(self, index):
            chosen = replacement_index(self.rows, self.candidates, self.epoch, index, self.seed)
            result = super().__getitem__(chosen)
            if result['patient'] == EXCLUDED:
                raise ValueError('Excluded patient reached training')
            return result
    return ExcludingDataset
