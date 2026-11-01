"""Explicit new full-run authorization; never promotes the old numerical gate."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'lld-stability-verification-v1'))
import support as H
sys.path.insert(0, str(HERE))
V, B, P, C, T, S = H.V, H.B, H.P, H.C, H.T, H.S
ARMS = {a: P.ARMS[a] for a in ('M0', 'F0', 'Fprobe', 'F001', 'F010')}
RULE = dict(P.RULE, arms={a: list(v) for a, v in ARMS.items()}, full_steps=2000, seeds=[0,1,2],
    direct_full=True, fresh_initialization=True, checkpoint_every=25,
    low_memory_F_only=B.CONFIG, step_seconds_limit=180., maximum_device_fraction=.85,
    authorization='User explicitly requested direct full training after 100-update stability results; skip pilot.',
    previous_numerical_gate_changed=False, numerical_equivalence_claimed=False,
    zero_control='Actual detached graph verified by prior stability experiment; report Fprobe/F0 differences, no bitwise identity claim.',
    selection='F001/F010 qualify vs M0,F0,Fprobe: mean DSC>0, mean D98>=-.002, mean HD95<=.02; '
              '>=2/3 positive DSC seeds, worst DSC>=-.002; rank mean DSC,HD95,name. '
              'All five predeclared arms retained for explicit test, regardless of eligibility.',
    note='Fresh 3-seed full experiment under explicit user authorization, not a numerical-equivalence waiver. '
         'LLD is synthetic motion, not real video. 1000 TrackRAD + 1000 LLD updates/arm/seed, not every LLD scan. '
         'Validation selects; public test-38 repeatedly inspected, not blind clinical validation.')


def sources():
    return {p.relative_to(C.CHECKOUT).as_posix(): C.sha(p) for p in sorted(HERE.iterdir())
            if p.suffix in ('.py', '.ps1', '.txt')}


def stability_binding(stability, evidence, baseline, previous, audit, deep=False):
    expected = dict(rule=H.RULE, sources=H.sources(),
        evidence=H.evidence_binding(evidence, baseline, previous, audit, deep),
        upstream=B.identity(baseline, previous, audit))
    if C.read(stability/'plan.json') != expected: raise ValueError('Stability source/inputs changed')
    if C.sha(stability/'environment.json') != C.sha(evidence/'environment.json'):
        raise ValueError('Stability environment differs')
    result = C.read(stability/'summary.json')
    for k in ('complete', 'graph_isolation_passed', 'paired_inputs_and_labels'):
        if result.get(k) is not True: raise ValueError('Incomplete stability verification: '+k)
    for k in ('previous_strict_gate_changed', 'numerical_equivalence_claimed', 'full_training_authorized', 'validation_or_test_accessed'):
        if result.get(k) is not False: raise ValueError('Unexpected prior diagnostic semantics: '+k)
    files = ['plan.json', 'environment.json', 'summary.json']; paired = None
    for name in ['graph']+['train/'+a for a in H.ARMS]:
        f = stability/name; kind = 'graph' if name == 'graph' else name.split('/')[1]
        receipt = C.read(f/'complete.json')
        expected_files = {'summary.json'} if kind == 'graph' else {'resume.pth','steps.json','probes.json','proof.json'}
        if (receipt['complete'], receipt['kind'], receipt['plan_sha256'], receipt['environment_sha256']) != (
                True, kind, C.sha(stability/'plan.json'), C.sha(stability/'environment.json')) or set(receipt['files']) != expected_files:
            raise ValueError('Invalid stability receipt')
        selected = {n: h for n, h in receipt['files'].items() if deep or n != 'resume.pth'}
        C.verify(f, selected)
        files += [name+'/complete.json']+[name+'/'+n for n in selected if n != 'resume.pth']
        if kind == 'graph':
            g=C.read(f/'summary.json')
            if not g['complete'] or not g['weights_unchanged'] or not g['optimizer_unchanged']:
                raise ValueError('Graph isolation proof incomplete')
            if [r['domain'] for r in g['rows']]!=['trackrad','lld_synthetic'] or any(
                    r['zero_student_gradient_paths']!=0 or not r['zero_embedding_gradient_is_none'] for r in g['rows']):
                raise ValueError('Graph isolation proof failed')
        else:
            arm = result['arms'][kind]; proof = C.read(f/'proof.json')
            if not arm['technical_checks_passed'] or not arm['frozen_exact'] or arm['updates'] != 100:
                raise ValueError('Failed prior stability arm')
            if arm['needs_loss_review'] or arm['fixed_probe_loss_review_alerts']:
                raise ValueError('This full protocol requires the reviewed no-alert stability results')
            if proof['arm']!=kind or proof['updates']!=100 or not proof['frozen_exact'] or not proof['sidecar_updated'] or not proof['partial_encoder_updated']:
                raise ValueError('Invalid stability weight proof')
            rows = C.read(f/'steps.json'); H.validate_records(rows)
            if len(rows) != 100: raise ValueError('Incomplete stability updates')
            bindings=[{k:r[k] for k in P.PAIR_KEYS} for r in rows]
            if paired is None: paired=bindings
            elif bindings!=paired: raise ValueError('Prior stability inputs not paired')
            probes=C.read(f/'probes.json'); H.validate_probes(probes,100)
            for p in probes:
                first=next(r for r in probes if r['sample_index']==p['sample_index'] and r['after_updates']==0)
                if H.loss_alert(first['point_loss'],p['point_loss']): raise ValueError('Unreviewed stability loss alert')
    return {n: C.sha(stability/n) for n in files}


def prepare(root, previous, audit, baseline, evidence, stability):
    bound = stability_binding(stability, evidence, baseline, previous, audit, deep=True)
    old = P.checked(baseline, previous)
    plan = dict(rule=RULE, sources=sources(), stability_files=bound,
        upstream=B.identity(baseline, previous, audit), stability_sources=H.sources(),
        same_state_sources=V.sources(), previous_files=old['previous_files'],
        ids=old['ids'], images=old['images'], external=old['external'],
        external_manifest_sha256=old['external_manifest_sha256'])
    C.freeze(root/'plan.json', plan)
    C.freeze(root/'full-roster.json', dict(plan=C.sha(root/'plan.json'), arms=list(ARMS), seeds=RULE['seeds'], steps=2000,
        authorization=RULE['authorization'], pilot_skipped=True))
    print('PLAN VERIFIED direct full: M0/F0/Fprobe/F001/F010 x seeds0/1/2 x2000; old numerical gate unchanged', flush=True)


def checked(root, previous):
    plan = C.read(root/'plan.json')
    if plan['rule'] != RULE or plan['sources'] != sources() or plan['stability_sources'] != H.sources() or plan['same_state_sources'] != V.sources():
        raise ValueError('Full study source/rule changed; preserve results')
    C.verify(previous, plan['previous_files']); C.checked(previous)
    return plan


def provenance(root, previous, audit, baseline, evidence, stability):
    plan = checked(root, previous)
    if plan['upstream'] != B.identity(baseline, previous, audit): raise ValueError('Upstream changed')
    C.verify(stability, plan['stability_files'])
    # Existing evidence remains immutable, including every recorded strict failure.
    if H.evidence_binding(evidence, baseline, previous, audit) != C.read(stability/'plan.json')['evidence']:
        raise ValueError('Numerical evidence changed')
    return plan


def arms_for(root, phase):
    if phase != 'full': raise ValueError('No smoke/pilot phase in direct full protocol')
    return list(ARMS)


def seeds_for(phase):
    if phase != 'full': raise ValueError('Only full phase')
    return RULE['seeds']


def gate(root, phase, arm, seed):
    if arm not in arms_for(root, phase) or seed not in seeds_for(phase): raise ValueError('Unplanned arm/seed')
    r = C.read(root/'full-roster.json')
    if r != dict(plan=C.sha(root/'plan.json'), arms=list(ARMS), seeds=RULE['seeds'], steps=2000,
                 authorization=RULE['authorization'], pilot_skipped=True): raise ValueError('Full roster changed')


def test_gate(root):
    f = root/'reports/full/validation-10'
    r = P.cached(root, f/'verified.json')
    if not r or r['phase'] != 'full' or r['split'] != 'validation-10': raise ValueError('Full validation report required before test')
    d = C.read(f/'selection.json')
    if d['selection_split'] != 'validation-10' or d['test_arms'] != list(ARMS) or not d['no_test_selection']:
        raise ValueError('Invalid frozen test roster')
    return list(ARMS)
