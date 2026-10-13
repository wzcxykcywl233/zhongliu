"""Bounded diagnosis only. Never trains, updates weights or fits a test gate."""
import argparse
import gc
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np
import SimpleITK as sitk
import torch

sys.path[:0]=['/opt/app','/opt/app/ext/co-tracker']
from probe_common import (PROFILE,CASES,ROUTES,MASK,read,sha,save,csv_save,split_displacement,
                         summarize_masks,native_xy,input_manifest,check_input_hashes)
from probe_common import frozen_output

def plan(training,study,dataset,weights,output,inference_image,training_image):
    protocol,hashes=input_manifest(training,study,dataset,weights)
    package={p.name:sha(p) for p in Path(__file__).parent.iterdir()
             if p.is_file() and p.suffix in ('.py','.ps1','.md')}
    manifest={'schema':2,'inputs':hashes,'package':package,'inference_image':inference_image,
              'training_image':training_image,'cases':list(CASES),'routes':list(ROUTES),
              'probe_seed':0,'probe_frames':60,'probe_points':128,'iterations':4,
              'training':False,'split':'validation-10'}
    frozen_output(output,manifest)
    save(output/'plan.json',{'complete':True,'manifest_sha256':sha(output/'manifest.json'),
                           'cpu_cached_pairs':24,'gpu_jobs':12,'smoke_jobs':1})

def verify_manifest(training,study,dataset,weights,output):
    manifest=read(output/'manifest.json')
    check_input_hashes(manifest['inputs'],training,study,dataset,weights)
    package={p.name:sha(p) for p in Path(__file__).parent.iterdir()
             if p.is_file() and p.suffix in ('.py','.ps1','.md')}
    if manifest['package']!=package: raise ValueError('Diagnostic package changed; use a new OutputRoot')
    return manifest

def full_tensor_hash(state):
    import hashlib
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def observe_corrections(model,points,frames,corrections):
    branch=model.updateformer.feature_sidecar
    if branch.runtime!={'backend':'reference','point_chunk_size':0,'gain':1.,'max_correction_pixels':0.,'checkpoint_blocks':False}:
        raise ValueError('Unexpected sidecar execution backend/runtime')
    captured={}
    def out_hook(module,args,result): captured['out']=result.detach()
    def gate_hook(module,args,result): captured['gate']=result.detach()
    def update_hook(module,args,result):
        residual=(captured.pop('out')*torch.sigmoid(captured.pop('gate'))*.1).reshape(1,points,frames,-1)
        raw=torch.nn.functional.linear(residual,model.updateformer.flow_head.weight)
        raw=raw.permute(0,2,1,3)*model.stride
        corrections.append(raw.detach().cpu().numpy())
    return [branch.output.register_forward_hook(out_hook),branch.gate.register_forward_hook(gate_hook),
            model.updateformer.register_forward_hook(update_hook)]

def verify_job(output,manifest):
    file=output/'verified.json'
    if not file.exists(): return False
    recorded=read(file)
    if recorded['manifest_sha256']!=sha(manifest): raise ValueError('Job belongs to another manifest')
    for name,checksum in recorded['outputs'].items():
        if sha(output/name)!=checksum: raise ValueError('Completed diagnostic output changed: '+name)
    return True

def commit_job(output,manifest):
    names=[p.name for p in output.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.npz') and p.name!='verified.json']
    save(output/'verified.json',{'complete':True,'manifest_sha256':sha(manifest),
                               'outputs':{name:sha(output/name) for name in names}})

def load_thw(path):
    a=sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if a.ndim!=3 or not np.isfinite(a).all(): raise ValueError('Invalid MRI/mask')
    return a.transpose(2,1,0)

def truth_for(dataset,case):
    p=dataset/case/'targets'/f'{case}_staple_labels.mha'
    if not p.exists(): p=dataset/case/'targets'/f'{case}_labels.mha'
    return load_thw(p)>0

def activate():
    # Historical training image may predate the newer activation helper.
    # Select copied source directly, never a stale installed wheel.
    import importlib
    import os
    source=Path(os.environ.get('COTRACKER_SOURCE_DIR','/opt/app/ext/co-tracker')).resolve()
    package=source/'cotracker'
    if not (package/'__init__.py').is_file(): raise ValueError('Copied CoTracker source is unavailable')
    for name,module in tuple(sys.modules.items()):
        if name=='cotracker' or name.startswith('cotracker.'):
            location=getattr(module,'__file__',None)
            if not location or package not in Path(location).resolve().parents:
                raise ValueError('CoTracker already imported from an unverified wheel')
    sys.path[:]=[str(source)]+[p for p in sys.path if p!=str(source)]
    importlib.invalidate_caches()
    imported=importlib.import_module('cotracker')
    if package not in Path(imported.__file__).resolve().parents: raise ValueError('Wrong import source')
    return source

def inspect_surface(training,study,output,surface):
    activate()
    from cotracker.datasets.trackrad_video_dataset import TrackRADVideoDataset
    p=Path(inspect.getfile(TrackRADVideoDataset))
    recorded=read(training/'frozen-run.json') if surface=='training' else read(study/'frozen-study.json')
    suffix='ext/co-tracker/cotracker/datasets/trackrad_video_dataset.py'
    match=[r for r in recorded['Source'] if r['File'].replace('\\','/').endswith(suffix)]
    if len(match)!=1 or sha(p)!=match[0]['SHA256'].lower(): raise ValueError('Actual dataset reader differs from frozen '+surface+' source')
    verified={suffix:sha(p)}
    if surface=='inference':
        for relative in ('model.py','resources/model.py','resources/reshape.py','resources/seg_to_tap.py',
                         'resources/tap_to_seg.py','ext/co-tracker/cotracker/models/core/cotracker/feature_sidecar.py',
                         'ext/co-tracker/cotracker/models/core/cotracker/cotracker3_offline.py',
                         'ext/co-tracker/cotracker/models/core/cotracker/cotracker.py'):
            match=[r for r in recorded['Source'] if r['File'].replace('\\','/').endswith('cotracker-algorithm/'+relative)]
            actual_file=Path('/opt/app')/relative
            if len(match)!=1 or sha(actual_file)!=match[0]['SHA256'].lower():
                raise ValueError('Actual inference numerical source differs from frozen study: '+relative)
            verified[relative]=sha(actual_file)
    # Non-square sentinel distinguishes T,H,W from T,W,H unambiguously.
    raw=np.arange(7*5*4,dtype=np.float32).reshape(7,5,4)
    sentinel=output/('sentinel-'+surface+'.mha')
    sitk.WriteImage(sitk.GetImageFromArray(raw),str(sentinel))
    actual=TrackRADVideoDataset._read_mha(sentinel)
    low,high=TrackRADVideoDataset._robust_intensity_bounds(raw)
    normalized=np.clip((raw-low)/(high-low),0,1)*255
    expected=normalized.transpose(2,1,0)
    swapped=normalized.transpose(2,0,1)
    actual=actual.numpy()
    canonical=actual.shape==expected.shape and np.allclose(actual,expected,atol=1e-4,rtol=0)
    transposed=actual.shape==swapped.shape and np.allclose(actual,swapped,atol=1e-4,rtol=0)
    if not canonical and not transposed: raise ValueError('Unknown dataset preprocessing; do not infer orientation')
    info={'complete':True,'surface':surface,'dataset_reader_sha256':sha(p),'source_verified':True,
          'actual_shape':list(actual.shape),'canonical_shape':list(expected.shape),
          'training_transposed':transposed,'percentile_normalization':True,
          'verified_source_files':verified,
          'note':'Measured actual frozen reader using a non-square W,H,T sentinel. This is not a model accuracy result.'}
    save(output/(surface+'-preprocessing.json'),info)
    return info

def render_tracks(points,native_shape,first_mask,config,transposed=False,visibility=None,confidence=None):
    import resources
    from tuning import build_validity_mask
    tensor=torch.from_numpy(np.ascontiguousarray(points)).float()
    validity=None
    if visibility is not None:
        validity=build_validity_mask(torch.from_numpy(visibility),torch.from_numpy(confidence),
                                     config['visibility_threshold'],config['confidence_threshold'])
    mask=resources.convert_point_trajectory_to_mask_sequence(tensor,video_shape=(384,512),validity=validity,
         morph_close_kernel=config.get('morph_close_kernel',1),keep_largest_component=config.get('keep_largest_component',False))
    shape=native_shape[::-1] if transposed else native_shape
    mask=resources.reshape_video(mask,target_shape=shape)[0].numpy()
    if transposed: mask=mask.transpose(0,2,1)
    if config.get('lock_first_mask',True): mask[0]=first_mask
    return mask.astype(bool)

def cached(study,training,dataset,output,root):
    activate(); torch.set_num_threads(1)
    hashes=read(root/'manifest.json')['inputs']
    spec=read(study/'study-spec.json')
    arms={a['name']:a for a in spec['arms']}
    all_rows,frame_rows=[] ,[]
    for case in CASES:
        gt=truth_for(dataset,case)
        first=load_thw(dataset/case/'targets'/f'{case}_first_label.mha')[0]>.5
        base=study/'validation-10/pretrained'/PROFILE/'checkpoint/jobs'/case
        rd=read(base/'diagnostics.json');cfg=rd['config']
        if cfg.get('mask_appearance_radius',0) or cfg.get('temporal_median_window',1)!=1:
            raise ValueError('Unexpected postprocessing; exact cached replay cannot be assumed')
        with np.load(base/'output/sidecar-tracks.npz',allow_pickle=False) as z: ref={k:z[k].copy() for k in z.files}
        original_ref=load_thw(base/MASK)>0
        replay_ref=render_tracks(ref['trajectories'],gt.shape[1:],first,cfg,visibility=ref['visibility'],confidence=ref['confidence'])
        if sha(base/MASK)!=rd['prediction']['output_file_sha256'] or not np.array_equal(replay_ref,original_ref):
            raise ValueError('Reference cache replay is not bitwise identical')
        for seed in (0,1,2):
            for kind in ('mlp','mamba'):
                arm=kind+'_full';job=study/'validation-10'/f'seed_{seed}'/arm/PROFILE/'checkpoint/jobs'/case
                d=read(job/'diagnostics.json')
                if d['config']!=cfg or d['model'].get('sidecar_runtime')!=arms[arm]['runtime']:
                    raise ValueError('Unpaired candidate config/runtime')
                with np.load(job/'output/sidecar-tracks.npz',allow_pickle=False) as z: cand={k:z[k].copy() for k in z.files}
                if not np.array_equal(ref['queries'],cand['queries']): raise ValueError('Unmatched query identities')
                replay_full=render_tracks(cand['trajectories'],gt.shape[1:],first,cfg,visibility=cand['visibility'],confidence=cand['confidence'])
                original=load_thw(job/MASK)>0
                if sha(job/MASK)!=d['prediction']['output_file_sha256'] or not np.array_equal(replay_full,original):
                    raise ValueError('Candidate cache replay is not bitwise identical')
                global_delta,local_delta=split_displacement(ref['trajectories'],cand['trajectories'])
                native_global=native_xy(global_delta,gt.shape[1:])-native_xy(np.zeros_like(global_delta),gt.shape[1:])
                native_local=native_xy(local_delta,gt.shape[1:])-native_xy(np.zeros_like(local_delta),gt.shape[1:])
                variants={'reference':replay_ref,'full':replay_full}
                # The counterfactuals deliberately reuse the reference V/C. No
                # thresholds are tuned; current profile has no V/C mask filter.
                if cfg['visibility_threshold'] is not None or cfg['confidence_threshold'] is not None:
                    raise ValueError('Density split expects unfiltered contour')
                for name,delta in (('global_only',global_delta),('local_only',local_delta)):
                    variants[name]=render_tracks(ref['trajectories']+delta,gt.shape[1:],first,cfg)
                for name,masks in variants.items():
                    rows,means=summarize_masks(masks,gt)
                    all_rows.append({'Case':case,'Seed':seed,'Kind':kind,'Variant':name,**means,
                        'GlobalMeanDXModelPixels':float(global_delta[:,1:,:,0].mean()),
                        'GlobalMeanDYModelPixels':float(global_delta[:,1:,:,1].mean()),
                        'GlobalMeanDXNativePixels':float(native_global[:,1:,:,0].mean()),
                        'GlobalMeanDYNativePixels':float(native_global[:,1:,:,1].mean()),
                        'LocalResidualRMSNativePixels':float(np.sqrt(np.mean(native_local[:,1:]**2))),
                        'LocalResidualRMSModelPixels':float(np.sqrt(np.mean(local_delta[:,1:]**2)))})
                    frame_rows.extend({'Case':case,'Seed':seed,'Kind':kind,'Variant':name,**r} for r in rows)
        print('CACHED '+case,flush=True)
    csv_save(output/'cached-split-summary.csv',all_rows);csv_save(output/'cached-split-frames.csv',frame_rows)
    collapsed=[]
    for case in CASES:
        for kind in ('mlp','mamba'):
            for variant in ('reference','full','global_only','local_only'):
                subset=[r for r in all_rows if r['Case']==case and r['Kind']==kind and r['Variant']==variant]
                means={key:float(np.mean([r[key] for r in subset if r[key] is not None]))
                       if any(r[key] is not None for r in subset) else None
                       for key in subset[0] if key not in ('Case','Seed','Kind','Variant')}
                collapsed.append({'Case':case,'Kind':kind,'Variant':variant,'Seeds':3,**means})
    csv_save(output/'cached-split-case-summary.csv',collapsed)
    save(output/'cached-complete.json',{'complete':True,'cases':list(CASES),'seeds':[0,1,2],
        'all_reference_and_full_replays_exact':True,'inputs':hashes,
        'note':'Diagnostic two-model counterfactual, not deployable single-model performance. No official DSC/D98 or new test-38 result.'})

def prepare_route(dataset,case,route,preprocessing,frames,points):
    import resources
    from cotracker.datasets.trackrad_video_dataset import TrackRADVideoDataset
    path=dataset/case/'images'/f'{case}_frames.mha'
    raw=sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    if raw.ndim!=3 or not np.isfinite(raw).all(): raise ValueError('Invalid input video')
    canonical=raw.transpose(2,1,0)
    gt=truth_for(dataset,case)[:frames]
    first=load_thw(dataset/case/'targets'/f'{case}_first_label.mha')[0]>.5
    transposed=route=='training_exact' and preprocessing['training_transposed']
    if route=='deployment_raw': full=canonical
    else:
        if ('dataset_reader_sha256' in preprocessing and
            sha(Path(inspect.getfile(TrackRADVideoDataset)))!=preprocessing['dataset_reader_sha256']):
            raise ValueError('Inference reader differs from original training reader; cannot recreate training_exact')
        full=TrackRADVideoDataset._read_mha(path).numpy()
        if preprocessing['training_transposed'] and not transposed: full=full.transpose(0,2,1)
    if len(full)<frames: raise ValueError('Fixed clip exceeds actual case frames; no padding allowed')
    model_first=first.T if transposed else first
    video=torch.from_numpy(np.ascontiguousarray(full[:frames]))[None,:,None].float()
    video=resources.reshape_video(video,(384,512)).expand(-1,-1,3,-1,-1).contiguous()
    mask=torch.from_numpy(np.ascontiguousarray(model_first))[None,None].float()
    mask=resources.reshape_video(mask,(384,512))
    queries=resources.convert_mask_to_points(mask,points)
    return video,queries,gt,first,transposed,{'RawMin':float(raw.min()),'RawMax':float(raw.max()),
        'VideoMean':float(video.mean()),'VideoStd':float(video.std()),
        'VideoSHA256':__import__('hashlib').sha256(video.numpy().tobytes()).hexdigest(),
        'QueriesSHA256':__import__('hashlib').sha256(queries.numpy().tobytes()).hexdigest()}

def headroom():
    free,total=torch.cuda.mem_get_info()
    if free<2*1024**3: raise RuntimeError('Less than 2 GiB free GPU memory; stop this probe')
    if max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>20*1024**3:
        raise RuntimeError('Probe exceeded 20 GiB GPU headroom gate')
    return {'FreeMiB':free/1024**2,'PeakAllocatedMiB':torch.cuda.max_memory_allocated()/1024**2,
            'PeakReservedMiB':torch.cuda.max_memory_reserved()/1024**2}

def probe(training,study,dataset,weights,output,root,case,route,smoke=False):
    activate(); torch.set_num_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable; CPU fallback is not allowed')
    preprocessing=read(root/'training-preprocessing.json')
    frames,points=(10,32) if smoke else (60,128)
    video,queries,gt,first,transposed,identities=prepare_route(dataset,case,route,preprocessing,frames,points)
    video,queries=video.cuda(),queries.cuda()
    from resources.model import setup_model
    from cotracker.models.core.cotracker.feature_sidecar import tensor_hash
    from cotracker.models.build_cotracker import build_cotracker
    from cotracker.models.core.model_utils import get_points_on_a_grid,get_uniformly_sampled_pts
    predictions={};iterations={};increments=[];metrics=[];memory=[];applied_corrections={}
    started=time.monotonic();torch.cuda.reset_peak_memory_stats();headroom()
    from cotracker.models.core.cotracker.backbone_growth import state_from_checkpoint
    ref_checkpoint=torch.load(training/'pretrained-frozen.pth',map_location='cpu',weights_only=True)
    source_checkpoint=torch.load(weights/'scaled_offline.pth',map_location='cpu',weights_only=True)
    if (not ref_checkpoint.get('freeze_base_for_inference') or
        ref_checkpoint.get('source_checkpoint_sha256')!=sha(weights/'scaled_offline.pth') or
        tensor_hash(ref_checkpoint['model'])!=tensor_hash(state_from_checkpoint(source_checkpoint))):
        raise ValueError('Frozen reference is not the original pretrained tensors')
    del source_checkpoint,ref_checkpoint
    cfg={'visibility_threshold':None,'confidence_threshold':None,'lock_first_mask':True}
    for kind in ('reference','mlp','mamba'):
        print('FORWARD '+case+' '+route+' '+kind,flush=True)
        path=training/'pretrained-frozen.pth' if kind=='reference' else training/'train/seed_0'/kind/'cotracker_three_final.pth'
        model=setup_model(str(path),device='cuda')
        baseline_hash=full_tensor_hash(model.state_dict())
        if kind=='reference': frozen_hash=tensor_hash(model.state_dict())
        elif tensor_hash(model.state_dict())!=frozen_hash: raise ValueError('Candidate backbone differs from frozen reference')
        iteration_positions=[];corrections=[]
        if kind!='reference':
            handles=observe_corrections(model,points,frames,corrections)
        else: handles=[]
        def observer(positions,features): iteration_positions.extend([p.cpu().numpy() for p in positions])
        with torch.no_grad():
            result=model(video=video,queries=queries,iters=4,is_train=False,iteration_observer=observer)
        torch.cuda.synchronize()
        if full_tensor_hash(model.state_dict())!=baseline_hash: raise ValueError('Model parameters mutated during forward')
        for h in handles:h.remove()
        if smoke:
            with torch.no_grad(): parity=model(video=video,queries=queries,iters=4,is_train=False)
            if not np.array_equal(result[0].cpu().numpy(),parity[0].cpu().numpy()):
                raise ValueError('Diagnostic hooks changed coordinate output')
            if any(not torch.equal(a,b) for a,b in zip(result[:3],parity[:3])):
                raise ValueError('Diagnostic hooks changed P/V/C')
            del parity
        if len(iteration_positions)!=4 or (kind!='reference' and len(corrections)!=4): raise ValueError('Missing iteration diagnostics')
        iterations[kind]=iteration_positions
        predictions[kind]=iteration_positions[-1]
        if any(not np.isfinite(p).all() for p in iteration_positions): raise ValueError('Nonfinite iteration coordinates')
        if kind!='reference':
            applied_corrections[kind]=corrections
            for it,c in enumerate(corrections,1):
                native=native_xy(c,np.array(gt.shape[1:]),transposed)-native_xy(np.zeros_like(c),np.array(gt.shape[1:]),transposed)
                mean=c[:,1:].mean(axis=(0,1,2));global_delta=c.mean(axis=2,keepdims=True)
                increments.append({'Case':case,'Route':route,'Kind':kind,'Iteration':it,
                    'CorrectionMeanDXModelPixels':float(mean[0]),'CorrectionMeanDYModelPixels':float(mean[1]),
                    'CorrectionMeanDXNativePixels':float(native[:,1:,: ,0].mean()),
                    'CorrectionMeanDYNativePixels':float(native[:,1:,: ,1].mean()),
                    'CorrectionMeanLengthModelPixels':float(np.linalg.norm(c[:,1:],axis=-1).mean()),
                    'GlobalCorrectionRMSModelPixels':float(np.sqrt(np.mean(global_delta[:,1:]**2))),
                    'LocalCorrectionRMSModelPixels':float(np.sqrt(np.mean((c-global_delta)[:,1:]**2)))})
        for it,p in enumerate(iteration_positions,1):
            masks=render_tracks(p,gt.shape[1:],first,cfg,transposed)
            _,means=summarize_masks(masks,gt)
            metrics.append({'Case':case,'Route':route,'Kind':kind,'Iteration':it,**means})
        memory.append({'Kind':kind,**headroom()})
        csv_save(output/'iteration-corrections.csv',increments)
        csv_save(output/'iteration-geometry.csv',metrics)
        save(output/'phase-profile.json',{'complete':False,'last_completed':kind,'case':case,'route':route,
             'seconds':time.monotonic()-started,'gpu_memory':memory,'input_identity':identities})
        print('OBSERVED '+kind+' iterations=4',flush=True)
        del model,result,handles;gc.collect();torch.cuda.empty_cache()
    meta=read(training/'train/seed_0/mlp/meta.json')
    if meta['teacher_types']!=['cotracker2v1','online_cotracker_three','offline_cotracker_three'] or meta['train_sift_size']!=0:
        raise ValueError('Unsupported actual teacher recipe; do not approximate silently')
    teacher_rows=[];teacher_iteration_rows=[]
    for name,filename,v2,offline in (('cotracker2v1','cotracker2v1.pth',True,False),
             ('online_cotracker_three','baseline_online.pth',False,False),
             ('offline_cotracker_three','baseline_offline.pth',False,True)):
        print('TEACHER '+case+' '+route+' '+name,flush=True)
        teacher=build_cotracker(checkpoint=str(weights/filename),v2=v2,offline=offline,window_len=60 if offline else 16).cuda().eval()
        q=queries.clone()
        if offline:
            torch.manual_seed(20261013)
            extra=get_uniformly_sampled_pts(100,frames,[384,512],device='cuda')
        else:
            if meta['train_grid_size']>0:
                grid=get_points_on_a_grid(meta['train_grid_size'],[384,512],device='cuda')
                extra=torch.cat((torch.zeros_like(grid[:,:,:1]),grid),-1)
            else: extra=q[:,:0]
        q=torch.cat((q,extra),1)
        with torch.no_grad(): result=teacher(video,q) # actual teacher default refinement count
        torch.cuda.synchronize()
        p=result[0][:,:,:points].detach().cpu().numpy();v=result[1][:,:,:points].detach().cpu().numpy()
        if not np.isfinite(p).all() or not np.isfinite(v).all(): raise ValueError('Nonfinite teacher output')
        delta=p-iterations['reference'][-1]
        common=(v>.9);common[:,0]=False
        native_delta=native_xy(p,gt.shape[1:],transposed)-native_xy(iterations['reference'][-1],gt.shape[1:],transposed)
        direction={}
        for kind in ('mlp','mamba'):
            branch_delta=predictions[kind]-predictions['reference']
            dot=np.sum(delta*branch_delta,axis=-1);den=np.linalg.norm(delta,axis=-1)*np.linalg.norm(branch_delta,axis=-1)
            eligible=common&(den>1e-8)
            direction['TeacherVs'+kind.title()+'FinalTrackDeltaCosineMean']=float((dot[eligible]/den[eligible]).mean()) if eligible.any() else None
            direction['TeacherVs'+kind.title()+'DirectionPointFrames']=int(eligible.sum())
            for index,correction in enumerate(applied_corrections[kind]):
                # Subtracting the observed branch term reconstructs the
                # post-base/pre-branch location approximately (float rounding).
                remaining=p-(iterations[kind][index]-correction)
                dot=np.sum(remaining*correction,axis=-1)
                den=np.linalg.norm(remaining,axis=-1)*np.linalg.norm(correction,axis=-1)
                eligible=common&(den>1e-8)
                teacher_iteration_rows.append({'Case':case,'Route':route,'Teacher':name,'Kind':kind,
                    'Iteration':index+1,'DirectionPointFrames':int(eligible.sum()),
                    'AppliedResidualVsTeacherRemainingCosineMean':float((dot[eligible]/den[eligible]).mean()) if eligible.any() else None})
        masks=render_tracks(p,gt.shape[1:],first,cfg,transposed)
        _,means=summarize_masks(masks,gt)
        teacher_rows.append({'Case':case,'Route':route,'Teacher':name,'VisiblePointFrames':int(common.sum()),
            'TeacherMinusReferenceDXModelPixels':float(delta[...,0][common].mean()) if common.any() else None,
            'TeacherMinusReferenceDYModelPixels':float(delta[...,1][common].mean()) if common.any() else None,
            'TeacherMinusReferenceDXNativePixels':float(native_delta[...,0][common].mean()) if common.any() else None,
            'TeacherMinusReferenceDYNativePixels':float(native_delta[...,1][common].mean()) if common.any() else None,
            **direction,
            'ExtraTeacherQueries':int(extra.shape[1]),**means})
        predictions[name]=p
        memory.append({'Kind':name,**headroom()})
        csv_save(output/'teacher-directions.csv',teacher_rows)
        csv_save(output/'teacher-iteration-directions.csv',teacher_iteration_rows)
        save(output/'phase-profile.json',{'complete':False,'last_completed':name,'case':case,'route':route,
             'seconds':time.monotonic()-started,'gpu_memory':memory,'input_identity':identities})
        print('OBSERVED '+name,flush=True)
        del teacher,result;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(output/'tracks.npz',queries=queries.cpu().numpy(),**predictions,
                        **{kind+'_iter'+str(i+1):p for kind,ps in iterations.items() for i,p in enumerate(ps)})
    csv_save(output/'iteration-corrections.csv',increments);csv_save(output/'iteration-geometry.csv',metrics)
    csv_save(output/'teacher-directions.csv',teacher_rows)
    csv_save(output/'teacher-iteration-directions.csv',teacher_iteration_rows)
    save(output/'complete.json',{'complete':True,'case':case,'route':route,'frames':frames,'points':points,
         'seed':0,'smoke':smoke,'training_transposed':transposed,'input_identity':identities,'gpu_memory':memory,
         'seconds':time.monotonic()-started,'note':'First contiguous clip, boundary queries, plain 4-iteration tracker. Not original random training batches or the hierarchical full-video pipeline. No optimizer/gradients or official test metric.'})
    save(output/'phase-profile.json',{'complete':True,'last_completed':'all','case':case,'route':route,
         'seconds':time.monotonic()-started,'gpu_memory':memory,'input_identity':identities})

def summarize(root):
    import csv
    combined={name:[] for name in ('iteration-corrections','iteration-geometry','teacher-directions','teacher-iteration-directions')}
    jobs=[]
    for case in CASES:
        for route in ROUTES:
            folder=root/'probes'/(case+'-'+route)
            if not verify_job(folder,root/'manifest.json'): continue
            info=read(folder/'complete.json')
            if info['case']!=case or info['route']!=route or info['smoke'] or info['frames']!=60 or info['points']!=128:
                raise ValueError('Unexpected diagnostic job definition')
            jobs.append(info)
            for name in combined:
                with (folder/(name+'.csv')).open(encoding='utf-8-sig',newline='') as f:
                    combined[name].extend(dict(row) for row in csv.DictReader(f))
    for name,rows in combined.items():
        if rows: csv_save(root/(name+'-all.csv'),rows)
    cached_ready=verify_job(root/'cached',root/'manifest.json')
    preprocessing={s:read(root/(s+'-preprocessing.json')) for s in ('training','inference')}
    save(root/'summary.json',{'complete':cached_ready and len(jobs)==12,'split':'validation-10',
         'cached_split_complete':cached_ready,'gpu_jobs_completed':len(jobs),'gpu_jobs_expected':12,
         'cases':list(CASES),'probe_seed':0,'routes':list(ROUTES),'preprocessing':preprocessing,
         'manifest_sha256':sha(root/'manifest.json'),
         'limitations':['No training, optimizer, backward, parameter update or test-38 run.',
          'GPU probes use first 60 frames, 128 boundary queries, four plain-model iterations; not original sampled training batches.',
          'Cache counterfactuals require two existing models and are not a deployable single-model method.',
          'Teacher alignment is descriptive, not proof of the learning cause; teachers have no point-level ground truth.',
          'No fitted threshold, p-value, blind-test claim or official DSC/HD95/MASD/CD/D98 recomputation.']})

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('plan','inspect','cached','probe','smoke','summary'),required=True)
    for key in ('training','study','dataset','output'): parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--weights',type=Path);parser.add_argument('--surface',choices=('training','inference'))
    parser.add_argument('--case',choices=CASES);parser.add_argument('--route',choices=ROUTES)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inference-image');parser.add_argument('--training-image')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='plan':
        plan(args.training,args.study,args.dataset,args.weights,args.output,args.inference_image,args.training_image)
    else:
        verify_manifest(args.training,args.study,args.dataset,args.weights,args.root)
        if args.stage=='inspect': inspect_surface(args.training,args.study,args.output,args.surface)
        elif args.stage=='summary':
            summarize(args.root)
        elif not verify_job(args.output,args.root/'manifest.json'):
            if args.stage=='cached': cached(args.study,args.training,args.dataset,args.output,args.root)
            else: probe(args.training,args.study,args.dataset,args.weights,args.output,args.root,args.case,args.route,args.stage=='smoke')
            verify_manifest(args.training,args.study,args.dataset,args.weights,args.root)
            commit_job(args.output,args.root/'manifest.json')
        else: print('SKIP verified '+str(args.output),flush=True)
