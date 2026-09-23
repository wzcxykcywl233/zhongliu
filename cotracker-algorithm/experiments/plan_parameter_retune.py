"""Validation-only staged selection. Python standard library; never reads test results."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import RETUNE_CATALOG as CATALOG
from parameter_retune import AXES, PAIRS

FIELDS = {'DSC':'dice_similarity_coefficient','HD95':'hausdorff_distance_95',
          'MASD':'surface_distance_average','CD':'center_distance','D98':'relative_d98_dose',
          'TimeSec':'total_time'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): value.update(chunk)
    return value.hexdigest()


def freeze(path, data):
    if path.exists():
        if read(path) != data:
            raise ValueError(f'Frozen selection changed: {path}. Use a new ResultsRoot.')
        return
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def manifest(profiles):
    names=list(dict.fromkeys(['rt_control','rt_decay'] + profiles))
    return {'schema_version':1, 'reference':'rt_control', 'training':False,
            'profiles':names, 'audit_output_hashes':True,
            'selection_split':'validation-10', 'splits':{'train':40,'validation':10,'test':38}}


def load_stage(root, stage, split='validation-10'):
    plan = read(root/(stage+'.json'))
    rows = {}; evidence = {}; ids = None
    for profile in plan['profiles']:
        path = root/stage/split/profile/'metrics.json'
        data = read(path)
        current = sorted(r['case_id'] for r in data['results'])
        count=10 if split=='validation-10' else 38
        if len(current) != count or len(set(current)) != count or (ids is not None and ids != current):
            raise ValueError('Incomplete or mismatched validation case IDs: '+profile)
        ids = current
        row = {key:float(data['aggregates'][field]) for key,field in FIELDS.items()}
        if not all(math.isfinite(x) for x in row.values()):
            raise ValueError('Non-finite aggregate: '+profile)
        hashes = []
        for case in ids:
            diagnostic = path.parent/'checkpoint'/'jobs'/case/'diagnostics.json'
            item = read(diagnostic)
            if item['profile'] != profile or item['config'] != CATALOG[profile]['config']:
                raise ValueError('Wrong executed config: '+profile+'/'+case)
            value = item['prediction']['array_sha256']
            if not value: raise ValueError('Missing prediction hash')
            output=diagnostic.parent/'output'/'images'/'mri-linac-series-targets'/'output.mha'
            if not (diagnostic.parent/'.complete').is_file() or digest(output)!=item['prediction'].get('output_file_sha256'):
                raise ValueError('Invalid committed output: '+profile+'/'+case)
            hashes.append(value)
            evidence[str(diagnostic.relative_to(root))] = digest(diagnostic)
        row['hashes'] = hashes
        rows[profile] = row
        evidence[str(path.relative_to(root))] = digest(path)
    return rows, evidence


def report(root,stage,rows,split='validation-10'):
    records=[]
    plan=read(root/(stage+'.json'))
    for name,row in rows.items():
        refs=[CATALOG[name]['reference']]
        refs+=plan.get('leave_one_change_out',{}).get(name,[])
        for ref in dict.fromkeys(refs):
            if ref not in rows or name==ref: continue
            other=rows[ref]
            record={'Split':split,'Profile':name,'Reference':ref,
                    'Changes':json.dumps(CATALOG[name]['changes'],sort_keys=True),
                    'Cases':len(row['hashes']),
                    'ChangedCases':sum(a!=b for a,b in zip(row['hashes'],other['hashes']))}
            for metric in FIELDS:
                record[metric]=row[metric]
                record['Delta_'+metric]=row[metric]-other[metric]
            records.append(record)
    if records:
        target=root/stage/split/'retune-matched-comparisons.csv'
        temp=target.with_suffix('.tmp')
        with temp.open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0])); writer.writeheader();writer.writerows(records)
        os.replace(temp,target)


def rank(row):
    return (-row['DSC'], row['HD95'], row['CD'], -row['D98'])


def choose_combinations(rows):
    chosen = {}
    for axis in AXES:
        candidates = [p for p in rows if CATALOG[p].get('axis') == axis]
        # Collapse exact-output aliases within an axis; do not call them independent effects.
        distinct = []; seen = set()
        reference = 'rt_decay' if axis == 'tau' else 'rt_control'
        for p in sorted(candidates, key=lambda p:(rank(rows[p]),p)):
            key = tuple(rows[p]['hashes'])
            if key == tuple(rows[reference]['hashes']) or key in seen:
                continue
            seen.add(key); distinct.append(p)
        chosen[axis] = [CATALOG[p]['value'] for p in distinct[:2]]
    selected = []
    for p, entry in CATALOG.items():
        if entry['stage'] != 'combination': continue
        a,b = entry['axes']; av,bv = entry['values']
        if av in chosen[a] and bv in chosen[b]: selected.append(p)
    return selected, chosen


def shortlist(rows):
    candidates = [p for p in rows if p not in ('rt_control','rt_repeat','rt_decay')]
    if not candidates: return []
    # Predeclared roles, no test labels. Speed role allows <=0.001 absolute DSC loss.
    accuracy = min(candidates, key=lambda p:(rank(rows[p]),p))
    eligible = [p for p in candidates if rows[p]['DSC'] >= rows['rt_control']['DSC']-.001]
    selected = [accuracy]
    if eligible:
        selected += [min(eligible,key=lambda p:(rows[p]['HD95'],rows[p]['MASD'],p)),
                     min(eligible,key=lambda p:(rows[p]['TimeSec'],rank(rows[p]),p))]
    return list(dict.fromkeys(selected))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['single','combination','final','verify-final','audit-test'],required=True)
    args=parser.parse_args(); root=args.root; root.mkdir(parents=True,exist_ok=True)
    freeze(root/'catalog.json',CATALOG)
    if args.stage == 'audit-test':
        rows,_=load_stage(root,'final','test-38')
        report(root,'final',rows,'test-38')
        print('Final 38-case configs, output hashes and metrics verified.',flush=True)
        return
    if args.stage == 'verify-final':
        rows,_=load_stage(root,'final')
        plan=read(root/'final.json')
        for p, expected in plan['validation_output_hashes'].items():
            if rows[p]['hashes'] != expected:
                raise ValueError('Final validation repeat differs: '+p)
        report(root,'final',rows)
        print('Final validation repeats match; test stage may proceed.',flush=True)
        return
    if args.stage == 'single':
        profiles=[p for p,e in CATALOG.items() if e['stage']!='combination']
        freeze(root/'single.json',manifest(profiles))
    else:
        rows,evidence=load_stage(root,'single')
        report(root,'single',rows)
        if rows['rt_control']['hashes'] != rows['rt_repeat']['hashes']:
            raise ValueError('Control repeat is not exact. Stop selection and investigate nondeterminism.')
        if any(abs(rows['rt_control'][k]-rows['rt_repeat'][k])>1e-8 for k in FIELDS if k!='TimeSec'):
            raise ValueError('Repeated control metrics differ')
        if args.stage == 'combination':
            profiles,axes=choose_combinations(rows)
            plan=manifest(profiles); plan['selected_axis_values']=axes
            plan['selection_evidence']=evidence
            freeze(root/'combination.json',plan)
        else:
            combo,extra=load_stage(root,'combination'); evidence.update(extra)
            report(root,'combination',combo)
            for control in ('rt_control','rt_decay'):
                if combo[control]['hashes'] != rows[control]['hashes']:
                    raise ValueError('Control output changed between stages: '+control)
            rows.update(combo)
            profiles=shortlist(rows)
            plan=manifest(profiles)
            plan['roles']='Validation DSC, boundary HD95, speed with DSC loss <=0.001; deduplicated'
            plan['selection_evidence']=evidence
            plan['validation_scores']={p:{k:v for k,v in rows[p].items() if k!='hashes'} for p in plan['profiles']}
            plan['leave_one_change_out']={}
            for p in profiles:
                entry=CATALOG[p]
                if entry['stage']=='combination':
                    controls=[]
                    for field in entry['changes']:
                        target=dict(entry['config'])
                        target[field]=CATALOG[entry['reference']]['config'][field]
                        controls += [n for n,e in CATALOG.items() if e['config']==target and n in rows]
                    plan['leave_one_change_out'][p]=controls
            removals=[p for group in plan['leave_one_change_out'].values() for p in group]
            plan['profiles']=list(dict.fromkeys(plan['profiles']+removals))
            plan['validation_output_hashes']={p:rows[p]['hashes'] for p in plan['profiles']}
            freeze(root/'final.json',plan)
    path=root/(args.stage+'.json')
    print('FROZEN '+str(path),flush=True)
    print('Profiles: '+str(len(read(path)['profiles'])),flush=True)


if __name__=='__main__': main()
