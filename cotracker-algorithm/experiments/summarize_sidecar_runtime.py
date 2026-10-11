"""Report fresh-process timeout coverage or real paired training phase costs."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).parent))
from audit_backbone_growth import file_sha
from summarize_backbone_growth import write_csv
from cotracker.models.core.cotracker.backbone_growth import save_json


def isolated(root):
    manifest=json.loads((root/'probe-manifest.json').read_text(encoding='utf-8-sig'))
    rows,hashes=[],{}
    for task in manifest:
        path=root/'isolated'/task['name']/'result.json'
        data=json.loads(path.read_text(encoding='utf-8-sig'))
        hashes[str(path.relative_to(root))]=file_sha(path)
        runtime=task['runtime']
        if data['kind']!=task['kind'] or data['runtime']!=runtime: raise ValueError('Unexpected isolated runtime')
        row={'Name':task['name'],'Kind':task['kind'],'Backend':runtime['backend'],
            'Chunk':runtime['point_chunk_size'],'Checkpoint':runtime['checkpoint_blocks'],
            'Status':'complete' if data['complete'] else data['status'],'Seconds':'','PeakAllocatedMiB':'','PeakReservedMiB':'',
            'SpeedupVsFreshReference':''}
        if data['complete']:
            if not data.get('fresh_process_per_configuration') or not data['correctness']['output_and_all_parameter_gradients_close']:
                raise ValueError('Missing per-configuration correctness/process proof')
            if not math.isfinite(data['median_step_seconds']) or data['median_step_seconds']<=0: raise ValueError('Invalid timing')
            row.update({'Seconds':data['median_step_seconds'],'PeakAllocatedMiB':data['peak_allocated_mib'],
                'PeakReservedMiB':data['peak_reserved_mib']})
        elif (data['status'],data['exit_code']) not in (('timeout',124),('killed',137)):
            raise ValueError('Unexpected failed probe; not a valid budget exit')
        rows.append(row)
    refs={r['Kind']:r['Seconds'] for r in rows if r['Backend']=='reference' and r['Status']=='complete'}
    for row in rows:
        if row['Status']=='complete' and row['Kind'] in refs:
            row['SpeedupVsFreshReference']=refs[row['Kind']]/row['Seconds']
    write_csv(root/'isolated-summary.csv',rows)
    save_json(root/'isolated-summary.json',{'complete':True,'configurations':len(rows),
        'completed':sum(r['Status']=='complete' for r in rows),'timeouts':sum(r['Status']=='timeout' for r in rows),
        'killed':sum(r['Status']=='killed' for r in rows),
        'result_sha256':hashes,'rows':rows,
        'note':'Budget exits are censored failures, not successful measurements. 137 can be timeout escalation or other SIGKILL/OOM, not proof of cause. Each config is a fresh process; external GPU/Windows load is not controlled. No automatic accelerator selection.'})


def training(root,steps):
    from audit_feature_sidecar import audit
    # Audit named two-arm subset, while the original default remains four arms.
    audit(root/'training',[0],steps,kinds=['mlp','mamba'])
    rows=[]; hashes={}
    for kind in ('mlp','mamba'):
        path=root/'training'/'seed_0'/kind/'phase-profile.json'
        data=json.loads(path.read_text())
        if not data['complete'] or data['diagnostic_steps']!=steps or data['scheduler_num_steps']!=1000:
            raise ValueError('Partial or changed training probe')
        hashes[kind]=file_sha(path)
        for step in data['rows']:
            for phase,item in step['phases'].items():
                rows.append({'Kind':kind,'Step':step['step'],'Phase':phase,**item,
                    'PeakAllocatedMiB':step['memory_after']['peak_allocated_mib'],
                    'PeakReservedMiB':step['memory_after']['peak_reserved_mib'],
                    'DriverFreeAfterMiB':step['memory_after']['driver_free_mib']})
            rows.append({'Kind':kind,'Step':step['step'],'Phase':'data_load','calls':1,
                'inclusive_seconds':step['data_load_seconds'],'exclusive_seconds':step['data_load_seconds'],
                'PeakAllocatedMiB':step['memory_after']['peak_allocated_mib'],
                'PeakReservedMiB':step['memory_after']['peak_reserved_mib'],
                'DriverFreeAfterMiB':step['memory_after']['driver_free_mib']})
    summary=[]
    for kind,phase in sorted({(r['Kind'],r['Phase']) for r in rows}):
        group=[r for r in rows if r['Kind']==kind and r['Phase']==phase]
        summary.append({'Kind':kind,'Phase':phase,'ObservedSteps':len(group),
            'InclusiveMeanSec':statistics.mean(r['inclusive_seconds'] for r in group),
            'ExclusiveMeanSec':statistics.mean(r['exclusive_seconds'] for r in group),
            'CallsMean':statistics.mean(r['calls'] for r in group)})
    write_csv(root/'training-phase-steps.csv',rows)
    write_csv(root/'training-phase-summary.csv',summary)
    save_json(root/'training-phase-summary.json',{'complete':True,'steps':steps,'kinds':['mlp','mamba'],
        'phase_profile_sha256':hashes,'rows':summary,
        'note':'Three synchronized updates diagnose phases, not steady-state throughput or training quality. Inclusive scopes overlap; use exclusive times to avoid double counting. Backward_total is not pure sidecar backward.'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['profile','training'],required=True)
    p.add_argument('--steps',type=int,default=3)
    a=p.parse_args()
    isolated(a.root) if a.stage=='profile' else training(a.root,a.steps)
