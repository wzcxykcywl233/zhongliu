"""Run only G1, with immutable cached labels from the completed D1 comparison."""
from common import *
import train as T
import torch


def run(root, previous, weights, audit, acdc, data, seed):
    target = root/'g1'
    authorize_g1(root, previous, target, 'full', 'G1')
    _, reference, refproof = training_receipt(previous, 'D1', seed)
    repair = module('followup_short_repair', PACKAGE.parent/'region-short-teacher-fix-v1/worker.py')
    gate, labels, old_receipt = T.gate, T.labels, T.receipt

    def override(r, phase, arm):
        authorize_g1(root, previous, r, phase, arm)

    def readonly_labels(r, plan, w, sample, s, step, models):
        marker = previous/f'labels/seed_{s}/{step:05d}.json'
        proof = C.cached(previous, marker)
        if not proof: raise ValueError('Required old label cache missing; no regeneration allowed')
        if repair.short_row(proof['binding']):
            binding = previous/repair.BINDING
            if proof['files'].get(repair.BINDING) != C.sha(binding):
                raise ValueError('Short teacher label provenance missing')
        # The original labels() function verifies the current sample binding on a cache hit.
        value, binding = labels(previous, plan, w, sample, s, step, models)
        expected = reference[step]
        for key, tensor in [('label_tracks','tracks'),('label_visibility','visibility'),('label_confidence','confidence')]:
            if T.S.tensor_sha(value[tensor]) != expected[key]:
                raise ValueError('Cached label differs from original D1 training')
        return value, binding

    def bound_receipt(r, path, files, **extra):
        return old_receipt(r, path, list(files)+[target/'authorization.json'], **extra)

    T.gate, T.labels, T.receipt = override, readonly_labels, bound_receipt
    try:
        repair.train_fixed(target, weights, audit, acdc, data, 'full', 'G1', seed, 180)
    finally:
        T.gate, T.labels, T.receipt = gate, labels, old_receipt
    folder, rows, proof = training_receipt(target, 'G1', seed)
    compare_g1(rows, reference, proof, refproof)
    receipt(root, folder/'followup-verified.json',
        [folder/n for n in ('final.pth','proof.json','steps.json','verified.json')],
        seed=seed, arm='G1', paired_with='previous D1', post_gate_exploration=True)
    print(f'AUDIT PASSED G1 seed={seed}; exact paired inputs/labels; frozen backbone; coordinate loss zero', flush=True)
