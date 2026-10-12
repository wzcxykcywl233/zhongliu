"""Validate legacy long-path output in a pinned container, copying not deleting.

No model forward pass. Only the narrow sidecar validation recovery is supported.
Legacy logs have no execution receipt: provenance relies additionally on the
unchanged study/source/data/weights/image guard in the calling repair queue.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import numpy as np
import SimpleITK as sitk

if Path('/opt/app/experiments.py').is_file(): sys.path.insert(0,'/opt/app')
from experiments import get_experiment_config


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def log_text(raw):
    # Windows PowerShell 5 Tee-Object creates UTF-16 logs. A runner log can
    # also mix UTF-8 host messages and BOM-less native UTF-16 append chunks.
    # Required evidence is ASCII (JSON uses ensure_ascii). Keep raw bytes in
    # snapshots/copies; normalize only the parsed view of the log.
    if raw.startswith((b'\xff\xfe',b'\xfe\xff')): return raw.decode('utf-16')
    return raw.decode('utf-8-sig',errors='replace').replace('\x00','')


def validate(folder,case,case_id,profile,kind,runtime):
    output=folder/'output/images/mri-linac-series-targets/output.mha'
    log=folder/'case.log'
    cache=folder/'output/sidecar-tracks.npz'
    if not all(p.is_file() and not p.is_symlink() for p in (output,log,cache)):
        raise ValueError('Legacy output/log/track cache is missing')
    if any(folder.resolve() not in p.resolve().parents for p in (output,log,cache)):
        raise ValueError('Legacy output escaped its attempt directory')
    text=log_text(log.read_bytes())
    if re.search(r'Traceback|RuntimeError|CUDA out of memory',text): raise ValueError('Legacy inference log contains an error')
    lines=[line.strip()[len('TRACKRAD_DIAGNOSTICS_JSON='):] for line in text.splitlines() if line.strip().startswith('TRACKRAD_DIAGNOSTICS_JSON=')]
    if len(lines)!=1: raise ValueError('Expected exactly one legacy diagnostic')
    d=json.loads(lines[0]);expected=asdict(get_experiment_config(profile)[1])
    canonical=json.dumps(expected,sort_keys=True,separators=(',',':'))
    if d['profile']!=profile or d['config']!=expected or d['config_sha256']!=hashlib.sha256(canonical.encode()).hexdigest():
        raise ValueError('Legacy experiment configuration differs')
    model=d['model']
    if model.get('feature_sidecar')!=kind or model.get('sidecar_runtime')!=runtime or not model.get('frozen_pretrained') or model.get('backbone_architecture')!='base':
        raise ValueError('Legacy sidecar runtime/backbone differs')
    times=[]
    for phase in ('loading','algorithm','writing'):
        matches=re.findall(r'^\s*Runtime '+phase+r':\s*([0-9.]+) s\s*$',text,re.M)
        if len(matches)!=1: raise ValueError('Incomplete legacy inference/write timing')
        value=float(matches[0])
        if not math.isfinite(value) or value<0: raise ValueError('Invalid runtime')
        times.append(value)
    mask=sitk.GetArrayFromImage(sitk.ReadImage(str(output)))
    frames=sitk.GetArrayFromImage(sitk.ReadImage(str(case/'images'/f'{case_id}_frames.mha')))
    if mask.dtype!=np.uint8 or mask.shape!=frames.shape or list(mask.shape)!=d['prediction']['shape'] or not np.isin(mask,[0,1]).all():
        raise ValueError('Legacy mask shape/type/values differ')
    if int(np.count_nonzero(mask))!=d['prediction']['foreground_voxels'] or hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()!=d['prediction']['array_sha256']:
        raise ValueError('Legacy mask array differs from diagnostic')
    with np.load(cache,allow_pickle=False) as data:
        tracks=data['trajectories'];queries=data['queries']
        if tracks.ndim!=4 or tracks.shape[0]!=1 or tracks.shape[-1]!=2 or tracks.shape[1]!=mask.shape[-1] or tracks.shape[2]!=d['mechanism']['query_points'] or queries.shape!=(1,tracks.shape[2],3):
            raise ValueError('Legacy track cache shape differs')
        if not np.isfinite(tracks).all() or not np.isfinite(queries).all(): raise ValueError('Nonfinite track cache')
        for field in ('visibility','confidence'):
            values=data[field]
            if values.shape!=tracks.shape[:3] or not np.isfinite(values).all() or not ((values>=0)&(values<=1)).all():
                raise ValueError('Invalid visibility/confidence cache')
    end=datetime.fromtimestamp(log.stat().st_mtime,timezone.utc)
    return {'source_attempt':folder.name,'output_file_sha256':sha(output),'cache_sha256':sha(cache),
        'log_sha256':sha(log),'array_sha256':d['prediction']['array_sha256'],
        'started_at':(end-timedelta(seconds=sum(times))).isoformat(),'completed_at':end.isoformat(),
        'timestamps_estimated_from_log_mtime':True}


def recover(previous,output,case,case_id,profile,kind,runtime,checkpoint_sha,runner_log=None):
    if kind not in ('mlp','mamba') or not re.fullmatch(r'[A-Z]_\d{3}',case_id) or not re.fullmatch(r'[0-9a-fA-F]{64}',checkpoint_sha):
        raise ValueError('Unsupported recovery identity')
    if previous.resolve()==output.resolve() or previous.resolve() in output.resolve().parents:
        raise ValueError('Source/output must be separate')
    candidates=[p for p in previous.iterdir() if p.is_dir() and not p.is_symlink() and re.fullmatch(re.escape(case_id)+r'-[0-9a-f]{8}',p.name)
        and (p/'output/images/mri-linac-series-targets/output.mha').is_file()]
    # Ambiguity must be inspected, never silently choose a different old run.
    if len(candidates)>1: raise ValueError('Multiple legacy outputs; inspect before reuse')
    receipt={'recovered':False,'case_id':case_id,'profile':profile}
    if candidates:
        folder=candidates[0];details=validate(folder,case,case_id,profile,kind,runtime)
        if any(p.is_file() and p.name!='recovery.log' for p in output.rglob('*')): raise ValueError('Recovery destination contains files')
        if runner_log is not None:
            snapshot_bytes=runner_log.read_bytes()
            snapshot=log_text(snapshot_bytes)
            start=None;bounds=[]
            for line in snapshot.splitlines():
                match=re.match(r'^\[([^]]+)\] (.*)$',line)
                if not match: continue
                stamp=datetime.fromisoformat(match[1]).astimezone(timezone.utc)
                message=match[2]
                if message==f'Running: {profile}/{case_id}': start=stamp
                if start and message==f'FAILED {profile}: {profile}/{case_id} did not produce a valid output.mha':
                    if abs((stamp-datetime.fromisoformat(details['completed_at'])).total_seconds())<=60:
                        bounds.append((start,stamp))
            if len(bounds)!=1: raise ValueError('Cannot reconstruct an unambiguous original runner time interval')
            details.update(started_at=bounds[0][0].isoformat(),completed_at=bounds[0][1].isoformat(),
                timestamps_estimated_from_log_mtime=False,runtime_estimated=True,
                timestamp_basis='original runner Running/FAILED event bounds; not exact Docker start/finish timestamps')
            (output/'runner-snapshot.log').write_bytes(snapshot_bytes)
            details['runner_snapshot_sha256']=sha(output/'runner-snapshot.log')
        destination=output/'output/images/mri-linac-series-targets/output.mha'
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(folder/'output/images/mri-linac-series-targets/output.mha',destination)
        shutil.copy2(folder/'output/sidecar-tracks.npz',output/'output/sidecar-tracks.npz')
        shutil.copy2(folder/'case.log',output/'case.log')
        if sha(output/'output/images/mri-linac-series-targets/output.mha')!=details['output_file_sha256'] or sha(output/'output/sidecar-tracks.npz')!=details['cache_sha256']:
            raise ValueError('Recovery copy changed bytes')
        receipt.update(recovered=True,checkpoint_sha256=checkpoint_sha.lower(),runtime=runtime,**details,
            note='Original attempt retained. Recovered TimeSec is estimated, not an exact speed benchmark. Inference time stays in copied case.log. No model inference/retraining.')
    (output/'recovery.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('RECOVERY '+json.dumps(receipt),flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('previous','output','case','runtime'): p.add_argument('--'+name,type=Path,required=True)
    for name in ('case-id','profile','kind','checkpoint-sha256'): p.add_argument('--'+name,required=True)
    p.add_argument('--runner-log',type=Path)
    a=p.parse_args()
    recover(a.previous,a.output,a.case,a.case_id,a.profile,a.kind,json.loads(a.runtime.read_text(encoding='utf-8-sig')),a.checkpoint_sha256,a.runner_log)
