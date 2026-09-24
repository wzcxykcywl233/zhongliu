"""Explicit, audited reuse from the pre-gridfix validation run. Never edits source."""
import argparse
import shutil
import uuid
from pathlib import Path
from plan_parameter_retune import read, digest, freeze, CATALOG, FIELDS

OLD_HELPERS = {
 'cotracker-algorithm/resources/model.py': {
  'e1351aa85c1f87c3dcba0b93fd355e32532cb703a5c67e6a9c4c525966ed4360',
  'cb77d6b0f7dbe4df848e45b5b001cec5b0169ff57a533a61eb782eca9bece251'},
 'cotracker-algorithm/ext/co-tracker/cotracker/models/core/cotracker/cotracker3_offline.py': {
  '6b234c7a757554ea8b319e76829a1743e0b92fbf199c8a60c5978d573c7b667b',
  '0a3c7a183ebfb81b84830bfb5e101f3a52b860dbc35aee6978145da2b5d9c754'},
}
CONTROLS = ('rt_control','rt_repeat','rt_decay')


def verify_protocol(old, new):
    for key in ('Manifest','CheckpointSHA256','Dataset'):
        if old[key] != new[key]: raise ValueError('Reuse protocol mismatch: '+key)
    def runtime(data):
        return {r['File']:r['SHA256'].lower() for r in data['Source']
                if (r['File'].startswith('cotracker-algorithm/') or r['File'].startswith('evaluation/'))
                and not r['File'].startswith(('cotracker-algorithm/tests/','cotracker-algorithm/experiments/'))}
    a,b=runtime(old),runtime(new)
    if set(a)!=set(b): raise ValueError('Runtime file set changed')
    for name, value in a.items():
        if name in OLD_HELPERS:
            if value not in OLD_HELPERS[name]: raise ValueError('Unrecognized pre-fix implementation: '+name)
        elif value != b[name]: raise ValueError('Unrelated runtime change: '+name)


def verify_profile(directory, name, case_ids):
    metrics=read(directory/'metrics.json')
    rows=metrics['results']
    if sorted(r['case_id'] for r in rows)!=case_ids: raise ValueError('Incomplete case coverage: '+name)
    hashes={}
    for case in case_ids:
        job=directory/'checkpoint'/'jobs'/case
        diagnostic=read(job/'diagnostics.json')
        if diagnostic['profile']!=name or diagnostic['config']!=CATALOG[name]['config']:
            raise ValueError('Wrong stored config: '+name+'/'+case)
        output=job/'output'/'images'/'mri-linac-series-targets'/'output.mha'
        if not (job/'.complete').is_file() or not (job/'prediction.json').is_file():
            raise ValueError('Incomplete checkpoint: '+name+'/'+case)
        if digest(output)!=diagnostic['prediction'].get('output_file_sha256'):
            raise ValueError('Corrupt prediction: '+name+'/'+case)
        array=diagnostic['prediction'].get('array_sha256')
        if not array: raise ValueError('Missing array hash')
        hashes[case]=array
    return metrics,hashes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--old',type=Path,required=True)
    parser.add_argument('--new',type=Path,required=True,help='New single-stage directory')
    args=parser.parse_args()
    old=args.old/'single'; new=args.new
    before=read(old/'frozen-run.json'); after=read(new/'frozen-run.json')
    verify_protocol(before,after)
    cases=sorted({r['Case'] for r in after['Dataset'] if r['Split']=='validation-10'})
    if len(cases)!=10: raise ValueError('Expected ten validation cases')
    for profile in CONTROLS:
        om,oh=verify_profile(old/'validation-10'/profile,profile,cases)
        nm,nh=verify_profile(new/'validation-10'/profile,profile,cases)
        if oh!=nh: raise ValueError('Grid0 control output changed; reuse forbidden: '+profile)
        om={r['case_id']:r for r in om['results']}; nm={r['case_id']:r for r in nm['results']}
        for case in cases:
            for metric in FIELDS.values():
                if metric!='total_time' and abs(float(om[case][metric])-float(nm[case][metric]))>1e-8:
                    raise ValueError('Control metric changed: '+profile+'/'+case)
    eligible=[p for p in after['Manifest']['profiles'] if p not in CONTROLS and CATALOG[p]['config']['support_grid_size']==0]
    receipt={'schema':1,'source_frozen_sha256':digest(old/'frozen-run.json'),
             'source_images':read(old/'validation-10'/'frozen-images.json'),
             'new_images':read(new/'validation-10'/'frozen-images.json'),
             'controls_recomputed':list(CONTROLS),'profiles':eligible,
             'note':'Mixed-provenance reuse, not original-runtime identity. Grid0 controls match exactly; source files/weights/data checked. Timings across images are exploratory.'}
    freeze(new/'reuse-provenance.json',receipt)
    for profile in eligible:
        source=old/'validation-10'/profile; target=new/'validation-10'/profile
        _,expected=verify_profile(source,profile,cases)
        proof={'source_metrics_sha256':digest(source/'metrics.json'),
               'source_images':receipt['source_images'],'array_hashes':expected}
        if target.exists():
            if read(target/'reuse-origin.json')!=proof: raise ValueError('Reuse target lacks matching provenance: '+profile)
            _,actual=verify_profile(target,profile,cases)
            if actual!=expected or digest(target/'metrics.json')!=proof['source_metrics_sha256']:
                raise ValueError('Reused result changed: '+profile)
            print('REUSE VERIFIED '+profile,flush=True); continue
        pending=new/'.imports'/(profile+'-'+uuid.uuid4().hex)
        pending.parent.mkdir(exist_ok=True)
        shutil.copytree(source,pending,ignore=shutil.ignore_patterns('.attempts'))
        _,actual=verify_profile(pending,profile,cases)
        if actual!=expected or digest(pending/'metrics.json')!=proof['source_metrics_sha256']:
            raise ValueError('Source changed during copy')
        freeze(pending/'reuse-origin.json',proof)
        pending.rename(target)
        print('REUSED '+profile,flush=True)


if __name__=='__main__': main()
