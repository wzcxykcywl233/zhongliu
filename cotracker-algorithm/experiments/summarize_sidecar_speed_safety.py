"""Official split performance plus correction->trajectory->mask diagnostics."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'ext' / 'co-tracker'),str(Path(__file__).parent)]
from audit_backbone_growth import file_sha
from summarize_backbone_growth import FIELDS, PROFILE, write_csv
from cotracker.models.core.cotracker.backbone_growth import save_json


def read_json(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def load(folder, count, arm=None):
    metrics = folder / PROFILE / 'metrics.json'
    data = read_json(metrics)
    cases = {r['case_id']:r for r in data['results']}
    if len(cases) != count or len(data['results']) != count: raise ValueError('Incomplete case coverage')
    for case,row in cases.items():
        if any(not math.isfinite(float(row[f])) for f in FIELDS.values()): raise ValueError('Nonfinite metric')
        job = folder / PROFILE / 'checkpoint' / 'jobs' / case
        d = read_json(job / 'diagnostics.json')
        output = job / 'output' / 'images' / 'mri-linac-series-targets' / 'output.mha'
        if not (job / '.complete').is_file() or file_sha(output) != d['prediction']['output_file_sha256']:
            raise ValueError('Uncommitted or changed output')
        if not d['model'].get('frozen_pretrained') or d['profile'] != PROFILE: raise ValueError('Wrong executed control')
        if arm:
            if d['model'].get('feature_sidecar') != arm['kind'] or d['model'].get('sidecar_runtime') != arm['runtime']:
                raise ValueError('Executed runtime differs from frozen intervention')
            if not d['mechanism'].get('sidecar_correction_count'): raise ValueError('Sidecar did not execute')
        elif d['model'].get('feature_sidecar','none') != 'none': raise ValueError('Reference is not frozen pretrained')
        if not (job / 'output' / 'sidecar-tracks.npz').is_file(): raise ValueError('Track diagnostic cache missing')
    aggregate = {**{k:float(data['aggregates'][f]) for k,f in FIELDS.items()},
                 'TimeSec':float(data['aggregates']['total_time'])}
    if any(not math.isfinite(v) for v in aggregate.values()): raise ValueError('Nonfinite aggregate')
    return aggregate,cases,file_sha(metrics)


def compare_caches(candidate, reference):
    with np.load(candidate / 'output' / 'sidecar-tracks.npz',allow_pickle=False) as c, np.load(reference / 'output' / 'sidecar-tracks.npz',allow_pickle=False) as r:
        if not np.array_equal(c['queries'],r['queries']) or c['trajectories'].shape != r['trajectories'].shape:
            raise ValueError('Queries/track frames are not paired')
        if c['trajectories'].ndim!=4 or c['trajectories'].shape[1]<=1 or c['trajectories'].shape[-1]!=2:
            raise ValueError('Invalid trajectory cache shape')
        distances = np.linalg.norm(c['trajectories'][:,1:]-r['trajectories'][:,1:],axis=-1)
        if not np.isfinite(distances).all(): raise ValueError('Nonfinite trajectory comparison')
        frame_count = c['trajectories'].shape[1]
        track_mean,track_max = float(distances.mean()),float(distances.max())
    path = 'output/images/mri-linac-series-targets/output.mha'
    cm = sitk.GetArrayFromImage(sitk.ReadImage(str(candidate / path))) > 0
    rm = sitk.GetArrayFromImage(sitk.ReadImage(str(reference / path))) > 0
    if cm.shape != rm.shape: raise ValueError('Unpaired mask shapes')
    return {'Frames':frame_count,'TrackMeanDisplacementInputPixels':track_mean,
            'TrackMaxDisplacementInputPixels':track_max,'MaskChangedVoxels':int(np.count_nonzero(cm!=rm)),
            'MaskChangedFraction':float(np.mean(cm!=rm))}


def choose_candidates(rows):
    selected = []
    for kind in ('mlp','mamba'):
        eligible = [r for r in rows if r['Kind']==kind and r['Delta_DSC_Mean']>=0 and r['Delta_D98_Mean']>=-.001]
        if eligible:
            best = sorted(eligible,key=lambda r:(-r['DSC_Mean'],r['HD95_Mean'],r['CD_Mean'],-r['D98_Mean']))[0]
            selected.append(best['Arm'])
    return selected


def cohort_diagnostics(rows):
    # Collapse seeds within each case first. Seed repetitions are not extra
    # patients. Cohort IDs describe observed errors, never inference rules.
    paired = {}
    for row in rows:
        paired.setdefault((row['Arm'],row['Case']),[]).append(row)
    grouped = {}
    fields = ['Delta_'+k for k in FIELDS] + ['Frames','CorrectionMeanInputPixels',
        'TrackMeanDisplacementInputPixels','MaskChangedFraction','ClippedFraction']
    for (arm,case),replicates in paired.items():
        if len(replicates)!=3 or len({r['Seed'] for r in replicates})!=3:
            raise ValueError('Cohort diagnosis requires three paired seeds')
        average = {key:statistics.mean(r[key] for r in replicates) for key in fields}
        for cohort in ('all',case.split('_')[0]):
            grouped.setdefault((arm,cohort),[]).append(average)
    output = []
    for (arm,cohort),cases in sorted(grouped.items()):
        output.append({'Arm':arm,'Cohort':cohort,'Cases':len(cases),
            **{key+'_Mean':statistics.mean(r[key] for r in cases) for key in fields},
            'DSCImprovedCases':sum(r['Delta_DSC']>1e-8 for r in cases),
            'DSCWorsenedCases':sum(r['Delta_DSC']<-1e-8 for r in cases),
            'D98WorsenedCases':sum(r['Delta_D98']<-1e-8 for r in cases)})
    return output


def summarize(root,split):
    spec = read_json(root / 'study-spec.json')
    count = 10 if split=='validation-10' else 38
    arms = spec['arms']
    if split=='test-38':
        selection = read_json(root / 'selection.json')
        included = set(selection['selected_arms']) | {'mlp_full','mamba_full'}
        arms = [a for a in arms if a['name'] in included]
    ref,rcases,rhash = load(root / split / 'pretrained',count)
    hashes = {'pretrained/'+PROFILE+'/metrics.json':rhash}
    result = [{'Seed':-1,'Arm':'pretrained','Kind':'pretrained',**ref}]
    cases, collected, backend_checks = [], {}, []
    for arm in arms:
        values = []
        for seed in (0,1,2):
            folder = root / split / f'seed_{seed}' / arm['name']
            aggregate, ccases, checksum = load(folder,count,arm)
            if set(ccases)!=set(rcases): raise ValueError('Case IDs differ')
            hashes[str((folder / PROFILE / 'metrics.json').relative_to(root / split)).replace('\\','/')] = checksum
            values.append(aggregate)
            result.append({'Seed':seed,'Arm':arm['name'],'Kind':arm['kind'],**aggregate})
            for case in sorted(rcases):
                job = folder / PROFILE / 'checkpoint' / 'jobs' / case
                refjob = root / split / 'pretrained' / PROFILE / 'checkpoint' / 'jobs' / case
                d = read_json(job / 'diagnostics.json')['mechanism']
                n = d['sidecar_correction_count']
                cases.append({'Seed':seed,'Arm':arm['name'],'Case':case,
                    **{'Delta_'+k:float(ccases[case][f])-float(rcases[case][f]) for k,f in FIELDS.items()},
                    'RawCorrectionMeanInputPixels':d['sidecar_raw_correction_sum']/n,
                    'RawCorrectionMaxInputPixels':d['sidecar_raw_correction_max'],
                    'CorrectionMeanInputPixels':d['sidecar_applied_correction_sum']/n,
                    'CorrectionMaxInputPixels':d['sidecar_applied_correction_max'],
                    'ClippedFraction':d['sidecar_clipped_corrections']/n,
                    **compare_caches(job,refjob)})
            if split=='validation-10' and arm['name']=='mamba_full' and seed==0:
                shadow_arm = {'kind':'mamba','runtime':{'backend':'reference','point_chunk_size':0,'gain':1.,'max_correction_pixels':0.,'checkpoint_blocks':False}}
                sfolder = root / split / 'seed_0' / 'mamba_reference'
                _,scases,shash = load(sfolder,count,shadow_arm)
                if set(scases)!=set(rcases): raise ValueError('Scan shadow case IDs differ')
                hashes['seed_0/mamba_reference/'+PROFILE+'/metrics.json'] = shash
                for case in sorted(scases):
                    shadowjob = sfolder / PROFILE / 'checkpoint' / 'jobs' / case
                    fastjob = folder / PROFILE / 'checkpoint' / 'jobs' / case
                    backend_checks.append({'Case':case,'Delta_DSC':float(ccases[case][FIELDS['DSC']])-float(scases[case][FIELDS['DSC']]),
                                           **compare_caches(fastjob,shadowjob)})
        collected[arm['name']] = (arm,values)
    summary = []
    for name,(arm,values) in collected.items():
        row = {'Arm':name,'Kind':arm['kind'],'N':3}
        for k in FIELDS:
            row[k+'_Mean'] = statistics.mean(v[k] for v in values)
            row[k+'_SD'] = statistics.stdev(v[k] for v in values)
            row['Delta_'+k+'_Mean'] = row[k+'_Mean']-ref[k]
        summary.append(row)
    write_csv(root / split / 'sidecar-safety-results.csv',result)
    write_csv(root / split / 'sidecar-safety-summary.csv',summary)
    write_csv(root / split / 'sidecar-safety-case-diagnostics.csv',cases)
    write_csv(root / split / 'sidecar-safety-cohort-diagnostics.csv',cohort_diagnostics(cases))
    if split=='validation-10':
        write_csv(root / split / 'scan-backend-agreement.csv',backend_checks)
        mean_error = abs(statistics.mean(r['Delta_DSC'] for r in backend_checks))
        passed = mean_error<=1e-5 and max(abs(r['Delta_DSC']) for r in backend_checks)<=1e-4 and max(r['TrackMaxDisplacementInputPixels'] for r in backend_checks)<=.001
        save_json(root / 'selection.json',{'selection_split':'validation-10','backend_agreement_passed':passed,
                  'selected_arms':choose_candidates(summary) if passed else [],
                  'rule':'Same setting across all seeds. DSC mean >= frozen control, D98 delta >= -0.001; rank DSC, HD95, CD, D98. No test-case rule.'})
    save_json(root / split / 'safety-audit.json',{'complete':True,'split':split,'cases':count,'seeds':[0,1,2],'metrics_sha256':hashes,
              'selection_sha256':file_sha(root / 'selection.json'),
              'note':'Performance is official metrics.json only. Track/correction/mask changes are diagnostic, not accuracy by themselves. Test-38 repeatedly inspected, no blind-test or patient-independence claim.'})
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--split',choices=['validation-10','test-38'],required=True)
    a = p.parse_args()
    summarize(a.root,a.split)
