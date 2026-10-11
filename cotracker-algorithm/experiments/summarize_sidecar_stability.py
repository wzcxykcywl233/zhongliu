"""Bounded soak diagnostics, with fresh-process checkpoint resume comparison."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).parent))
from audit_backbone_growth import records,file_sha
from audit_feature_sidecar import audit
from summarize_backbone_growth import write_csv
from cotracker.models.core.cotracker.backbone_growth import save_json


def stability(rows):
    if len(rows)!=100 or [r['step'] for r in rows]!=list(range(100)):
        raise ValueError('Expected all 100 ordered updates')
    for row in rows:
        if not row['complete'] or row['optimizer_updates']!=1 or not row.get('gradients_finite'):
            raise ValueError('Incomplete/nonfinite update')
        for key in ('wall_seconds_including_data','loss','gradient_norm','residual_rms_max','learning_rate'):
            if not math.isfinite(row[key]) or row[key]<0: raise ValueError('Invalid stability measurement: '+key)
        mem=row['memory_after']
        if not all(math.isfinite(mem[k]) for k in ('peak_allocated_mib','peak_reserved_mib','driver_free_mib','allocated_mib')):
            raise ValueError('Invalid memory measurement')
        if max(mem['peak_allocated_mib'],mem['peak_reserved_mib'])>20480 or mem['driver_free_mib']<2048:
            raise ValueError('Memory headroom failed')
    early,late=rows[10:30],rows[80:100]
    median=lambda group,key:statistics.median(r[key] for r in group)
    first,last=median(early,'wall_seconds_including_data'),median(late,'wall_seconds_including_data')
    if first<=0: raise ValueError('Zero step time')
    memory_median=lambda group,key:statistics.median(r['memory_after'][key] for r in group)
    allocated_growth=memory_median(late,'allocated_mib')-memory_median(early,'allocated_mib')
    peak_growth=memory_median(late,'peak_allocated_mib')-memory_median(early,'peak_allocated_mib')
    result={'early_median_seconds':first,'late_median_seconds':last,'late_early_ratio':last/first,
        'allocated_growth_mib':allocated_growth,'peak_allocated_growth_mib':peak_growth,
        'peak_allocated_mib':max(r['memory_after']['peak_allocated_mib'] for r in rows),
        'peak_reserved_mib':max(r['memory_after']['peak_reserved_mib'] for r in rows),
        'minimum_free_after_mib':min(r['memory_after']['driver_free_mib'] for r in rows),
        'early_loss_median':median(early,'loss'),'late_loss_median':median(late,'loss'),
        'maximum_gradient_norm':max(r['gradient_norm'] for r in rows),
        'maximum_residual_rms':max(r['residual_rms_max'] for r in rows)}
    # Different cases/teacher types and external GPU load can trigger review;
    # this flag identifies a failed diagnostic gate, not a proven memory leak.
    result['passed']=last/first<=2 and allocated_growth<=256 and peak_growth<=1024
    return result


def compare_tree(a,b,path='',tolerant=False):
    """Compare numerical state, not serialization hashes/paths or wall times."""
    differences=[]
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype:
            raise ValueError('State shape/dtype mismatch: '+path)
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()): raise ValueError('Nonfinite state: '+path)
        exact=torch.equal(a,b)
        close=exact or (tolerant and not path.endswith('/step') and a.is_floating_point() and torch.allclose(a,b,rtol=1e-5,atol=1e-7))
        if not close: raise ValueError('Resume state differs beyond tolerance: '+path)
        if not exact: differences.append({'state':path,'max_abs':float((a-b).abs().max())})
    elif isinstance(a,np.ndarray):
        if not isinstance(b,np.ndarray) or a.dtype!=b.dtype or not np.array_equal(a,b): raise ValueError('RNG array differs: '+path)
    elif isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b): raise ValueError('State keys differ: '+path)
        for key in a: differences+=compare_tree(a[key],b[key],path+'/'+str(key),tolerant)
    elif isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b): raise ValueError('State structure differs: '+path)
        for i,(x,y) in enumerate(zip(a,b)): differences+=compare_tree(x,y,path+'/'+str(i),tolerant)
    elif a!=b: raise ValueError('Scalar state differs: '+path)
    return differences


def summarize(root):
    proof={'complete':False,'steps':100,'interruption_step':50,'kinds':['mlp','mamba'],
        'note':'Synchronized, single-seed soak diagnostics; no steady throughput/accuracy claim. Late/early thresholds require review, not proof of a leak. No large training is authorized automatically.'}
    rows=[];states=[];hashes={};step_rows=[]
    try:
        for group in ('training','resumed'):
            audit(root/group,[0],100,kinds=['mlp','mamba'])
        for kind in ('mlp','mamba'):
            continuous=root/'training'/'seed_0'/kind
            resumed=root/'resumed'/'seed_0'/kind
            if records(continuous/'paired-steps.jsonl',100,'base')!=records(resumed/'paired-steps.jsonl',100,'base'):
                raise ValueError('Continuous/resumed teacher/case/query/video differs: '+kind)
            for group,folder in (('continuous',continuous),('resumed',resumed)):
                path=folder/'phase-profile.json';data=json.loads(path.read_text())
                if not data['complete'] or not data.get('soak') or data['scheduler_num_steps']!=1000 or data['timing_errors']:
                    raise ValueError('Incomplete soak phase report')
                if data['resumed_from_step']!=(50 if group=='resumed' else 0): raise ValueError('Wrong process boundary')
                if group=='resumed':
                    for name,key in (('resume-boundary-final.pth','resume_checkpoint_sha256'),('phase-profile-first50.json','first_segment_profile_sha256')):
                        p=folder/name
                        if file_sha(p)!=data[key]: raise ValueError('Resume boundary proof changed')
                        hashes[str(p.relative_to(root)).replace('\\','/')]=file_sha(p)
                rows.append({'Kind':kind,'Path':group,**stability(data['rows'])})
                for r in data['rows']:
                    step_rows.append({'Kind':kind,'Path':group,'Step':r['step'],
                        'WallSeconds':r['wall_seconds_including_data'],'Loss':r['loss'],
                        'GradientNorm':r['gradient_norm'],'ResidualRMSMax':r['residual_rms_max'],
                        'LearningRate':r['learning_rate'],'PeakAllocatedMiB':r['memory_after']['peak_allocated_mib'],
                        'PeakReservedMiB':r['memory_after']['peak_reserved_mib'],'DriverFreeAfterMiB':r['memory_after']['driver_free_mib']})
                for file in ('phase-profile.json','paired-steps.jsonl','cotracker_three_final.pth','model_cotracker_three_00000100.pth'):
                    p=folder/file;hashes[str(p.relative_to(root)).replace('\\','/')]=file_sha(p)
            a=torch.load(continuous/'model_cotracker_three_00000100.pth',map_location='cpu',weights_only=False)
            b=torch.load(resumed/'model_cotracker_three_00000100.pth',map_location='cpu',weights_only=False)
            differences=[]
            for key in ('model','optimizer','scheduler','total_steps','epoch','next_batch','teacher_sampler','rng_state'):
                differences+=compare_tree(a[key],b[key],key,tolerant=key in ('model','optimizer'))
            for folder,state in ((continuous,a),(resumed,b)):
                final=torch.load(folder/'cotracker_three_final.pth',map_location='cpu',weights_only=True)
                compare_tree(state['model'],final['model'],'final-model')
            states.append({'kind':kind,'inputs_exact':True,'model_optimizer_close':True,
                'all_state_exact':not differences,'nonexact_tensors':differences,'rtol':1e-5,'atol':1e-7})
        proof.update(complete=True,passed=all(r['passed'] for r in rows),rows=rows,resume_checks=states,artifact_sha256=hashes)
        write_csv(root/'stability-summary.csv',rows)
        write_csv(root/'stability-steps.csv',step_rows)
        if not proof['passed']: raise ValueError('Stability thresholds failed; review measurements before further training')
    except BaseException as error:
        proof.update(passed=False,error=type(error).__name__+': '+str(error))
        raise
    finally: save_json(root/'stability-summary.json',proof)
    print(json.dumps(proof),flush=True)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    summarize(p.parse_args().root)
