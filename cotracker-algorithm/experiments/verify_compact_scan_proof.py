"""Do not start even three real steps until fresh GPU memory/correctness proofs pass."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from audit_backbone_growth import file_sha
from cotracker.models.core.cotracker.backbone_growth import save_json


RUNTIME={'backend':'compact','point_chunk_size':0,'gain':1,'max_correction_pixels':0,'checkpoint_blocks':False}


def verify(root):
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    spec=read(root/'frozen-study.json')
    if not spec.get('CompactScan') or spec['ProbeSteps']!=3: raise ValueError('Wrong compact proof study')
    manifest=read(root/'probe-manifest.json')
    if len(manifest)!=3 or {r['kind'] for r in manifest}!={'mlp','mamba_short','mamba'}: raise ValueError('Missing compact arms')
    source=read(root/'source-training-audit.json')['final_checkpoints_sha256']
    hashes={}; rows=[]
    for arm in manifest:
        if arm['runtime']!=RUNTIME: raise ValueError('Compact runtime changed')
        path=root/'isolated'/arm['name']/'result.json'
        result=read(path)
        if not result['complete'] or result['kind']!=arm['kind'] or result['runtime']!=RUNTIME:
            raise ValueError('Compact probe failed/timed out; no real training will run')
        if not result.get('fresh_process_per_configuration') or not result['correctness']['output_and_all_parameter_gradients_close']:
            raise ValueError('Missing GPU output/gradient proof')
        parity=result.get('model_parity',{})
        if not parity.get('complete') or len(parity.get('max_abs_pvc',[]))!=3 or not all(math.isfinite(x) and x>=0 for x in parity['max_abs_pvc']) or parity['max_abs_pvc'][0]>.001 or max(parity['max_abs_pvc'][1:])>1e-5:
            raise ValueError('Missing GPU model P/V/C proof')
        expected=source[f"seed_0/{arm['kind']}/cotracker_three_final.pth"]
        if result['source_checkpoint_sha256'].lower()!=expected.lower(): raise ValueError('Probe checkpoint changed')
        values=[result['peak_allocated_mib'],result['peak_reserved_mib'],result['median_step_seconds']]
        if not result['trials'] or not all(math.isfinite(x) and x>0 for x in values) or result['peak_allocated_mib']>16384 or result['peak_reserved_mib']>16384:
            raise ValueError('Compact microbenchmark exceeds 16 GiB budget; do not train')
        if any(not all(math.isfinite(x) for x in (t['driver_total_mib'],t['driver_free_after_mib'])) or t['driver_total_mib']<20480 or t['driver_free_after_mib']<2048 for t in result['trials']):
            raise ValueError('Insufficient physical memory/headroom in compact proof')
        relative=str(path.relative_to(root)).replace('\\','/')
        hashes[relative]=file_sha(path)
        rows.append({'kind':arm['kind'],'step_seconds':result['median_step_seconds'],
            'peak_allocated_mib':result['peak_allocated_mib'],'peak_reserved_mib':result['peak_reserved_mib']})
    proof={'complete':True,'runtime':RUNTIME,'result_sha256':hashes,'rows':rows,
        'note':'Small GPU correctness and 60x128x4 synthetic memory gate. Not proof of end-to-end speed, full training headroom or official performance. Only three real diagnostic updates are permitted next.'}
    save_json(root/'compact-readiness.json',proof)
    print('COMPACT GPU PROOF PASSED; only three real diagnostic updates permitted',flush=True)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    verify(p.parse_args().root)
