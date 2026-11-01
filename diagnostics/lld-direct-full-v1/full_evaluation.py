"""Original point inference and official metric transport; direct-full paired report."""
import statistics
import full_context as X
from training import audit_training
P,C,RULE=X.P,X.C,X.RULE
E=P.module('lld_direct_original_evaluation',P.PACKAGE/'evaluation.py')
E.checked=X.checked; E.arms_for=X.arms_for; E.seeds_for=X.seeds_for; E.test_gate=X.test_gate
E.audit_training=audit_training; E.RULE=RULE
infer,record=E.infer,E.record
R=E.R
checked,arms_for,seeds_for,test_gate=X.checked,X.arms_for,X.seeds_for,X.test_gate
cached,receipt=P.cached,P.receipt
diagnostics=E.diagnostics


def selection(metrics,seeds,arms):
    def qualifies(a,b):
        return C.qualifies([{m:metrics[a,s][m]-metrics[b,s][m] for m in C.FIELDS} for s in seeds])
    valid=[a for a in ('F001','F010') if all(qualifies(a,b) for b in ('M0','F0','Fprobe'))]
    ranked=sorted(valid,key=lambda a:(-statistics.mean(metrics[a,s]['DSC'] for s in seeds),
        statistics.mean(metrics[a,s]['HD95'] for s in seeds),a))
    return dict(selection_split='validation-10',eligible=valid,promoted=ranked[:1],
        winner_qualifies_vs_pretrained={a:qualifies(a,'pretrained') for a in ranked[:1]},
        partial_no_adversarial_qualifies_vs_M0=qualifies('F0','M0'),rule=RULE['selection'])


def report(root,previous,phase,split):
    plan=checked(root,previous)
    if phase=='smoke': raise ValueError('Smoke is not an accuracy report')
    arms=arms_for(root,phase)
    if split=='test-38': arms=test_gate(root)
    if not arms: raise ValueError('No eligible full experiment')
    audit_training(root,previous,phase)
    seeds=seeds_for(phase); metrics={}; cases={}; rows=[]; summary=[]; files=[]
    for arm in ['pretrained']+arms:
        group=[]
        for seed in ([0] if arm=='pretrained' else seeds):
            folder=root/phase/split/f'seed_{seed}'/arm
            inf=cached(root,folder/'inference-verified.json')
            if not inf or inf['ids']!=plan['ids'][split] or not cached(root,folder/'metrics-verified.json'):
                raise ValueError('Missing complete official endpoint')
            agg,case=R.official(folder/'metrics.json',plan['ids'][split])
            metrics[arm,seed]=agg; cases[arm,seed]=case
            row=dict(Split=split,Phase=phase,Arm=arm,Seed=seed,**agg); rows.append(row); group.append(row)
            files += [folder/'metrics.json',folder/'inference-verified.json',folder/'metrics-verified.json']
        result=dict(Split=split,Phase=phase,Arm=arm,N=len(group))
        for m in list(C.FIELDS)+['TimeSec']:
            result[m+'_Mean']=statistics.mean(r[m] for r in group)
            result[m+'_SD']=statistics.stdev(r[m] for r in group) if len(group)>1 else 0.
        summary.append(result)
        if arm=='pretrained':
            for seed in seeds: metrics[arm,seed]=metrics[arm,0]; cases[arm,seed]=cases[arm,0]
    pairs={(a,'M0') for a in arms if a!='M0'} | {(a,'pretrained') for a in arms}
    pairs |= {(a,'F0') for a in arms if a.startswith('F') and a!='F0'}
    pairs |= {(a,'Fprobe') for a in ('F001','F010') if a in arms}
    deltas=[]; case_rows=[]
    for candidate,reference in sorted(pairs):
        for seed in seeds:
            prefix=dict(Split=split,Seed=seed,Candidate=candidate,Reference=reference)
            deltas.append(dict(**prefix,**{'Delta_'+m:metrics[candidate,seed][m]-metrics[reference,seed][m] for m in C.FIELDS}))
            for case in plan['ids'][split]:
                case_rows.append(dict(**prefix,Case=case,**{'Delta_'+m:cases[candidate,seed][case][m]-cases[reference,seed][case][m] for m in C.FIELDS}))
    folder=root/'reports'/phase/split
    for name,content in [('summary',summary),('per-seed',rows),('paired-deltas',deltas),('case-deltas',case_rows)]:
        path=folder/(name+'.csv'); R.csv_save(path,content); files.append(path)
    files.append(diagnostics(root,phase,arms,seeds,folder))
    if split=='validation-10':
        d=selection(metrics,seeds,arms)
        d.update(test_arms=arms if phase=='full' else [],no_test_selection=True,
            any_candidate_qualifies=bool(d['promoted'] or d['partial_no_adversarial_qualifies_vs_M0']),note=RULE['note'])
        C.freeze(folder/'selection.json',d); files.append(folder/'selection.json')
    receipt(root,folder/'verified.json',files,phase=phase,split=split)
    for row in summary:
        print(f'RESULT {row["Arm"]}: DSC={row["DSC_Mean"]:.6f}; D98={row["D98_Mean"]:.6f}',flush=True)
    print(f'REPORT COMPLETE {phase}/{split}',flush=True)
