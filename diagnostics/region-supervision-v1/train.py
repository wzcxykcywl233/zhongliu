"""Paired deterministic mixed training. Checkpoint commits records + optimizer together."""
import argparse
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from core import *
import smoke as S
from data import sample
from region import RegionHead, region_loss, boundary_loss

PREFIX='updateformer.feature_sidecar.'

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def steps_for(phase):
    if phase not in ('smoke','pilot','full'): raise ValueError('Invalid phase')
    return RULE[phase+'_steps']

def gate(root,phase,arm):
    if arm not in ARMS: raise ValueError('Invalid arm')
    if phase!='smoke':
        if not cached(root,root/'smoke/audit.json'): raise ValueError('GPU smoke audit required')
    if phase=='full' or (phase=='pilot' and arm=='G1'):
        if not cached(root,root/'reports/pilot/validation-10/verified.json'):
            raise ValueError('Pilot report required')
        if arm=='G1' and not read(root/'reports/pilot/validation-10/selection.json')['region_gate']:
            raise ValueError('G1 gate closed: D1 region output did not qualify')

def new_model(weights,seed,arm):
    from cotracker.models.build_cotracker import build_cotracker
    from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar,configure_sidecar_runtime,set_sidecar_training_mode
    seed_all(RULE['seed_base']+seed)
    m=build_cotracker(str(weights/'scaled_offline.pth'),offline=True,window_len=60)
    seed_all(RULE['seed_base']+seed)
    attach_sidecar(m,'mamba'); m.frozen_pretrained=True
    configure_sidecar_runtime(m,backend='compact')
    set_sidecar_training_mode(m)
    if any(p.requires_grad != n.startswith(PREFIX) for n,p in m.named_parameters()):
        raise ValueError('Only sidecar may train')
    seed_all(RULE['seed_base']+seed+777)
    head=RegionHead() if arm in ('D1','G1') else None
    return m,head

def features(m,video):
    with torch.no_grad():
        return F.interpolate(m.fnet(2*(video/255)-1),size=RULE['region_feature_size'],mode='bilinear',align_corners=False)

def decode_training(m,head,video,mask,tracks,annotated):
    # Only annotated non-query frames consume the region decoder in training.
    ids=annotated.nonzero().flatten(); ids=ids[ids!=0]
    if len(ids)==0: raise ValueError('No non-query region targets')
    qfeat=features(m,video[:1])
    values=[]
    for batch in ids.split(4):
        feat=features(m,video[batch])
        values.append(head(feat,mask[0],tracks[batch],qfeat,tracks[0]))
    logits=torch.cat(values)
    # region_loss excludes element0; insert the query placeholder explicitly.
    logits=torch.cat((logits[:1]*0,logits))
    selected=torch.cat((mask[:1],mask[ids]))
    return region_loss(logits,selected,torch.ones(len(logits),device=video.device,dtype=torch.bool))

def labels(root,plan,weights,s,seed,step,teacher_models):
    from cotracker.models.build_cotracker import build_cotracker
    from cotracker.models.core.model_utils import get_points_on_a_grid,get_uniformly_sampled_pts
    path=root/f'labels/seed_{seed}/{step:05d}.pth'; marker=path.with_suffix('.json')
    slot=int(np.random.default_rng(s['draw']+41).integers(3))
    # Force all six (domain,teacher) combinations in the bounded smoke prefix.
    if step<6: slot=step//2
    binding=dict(video=S.tensor_sha(s['video']),queries=S.tensor_sha(s['queries']),
                 sampling=s['sampling'],domain=s['domain'],slot=slot,draw=s['draw'])
    r=cached(root,marker)
    if r:
        if r['binding']!=binding: raise ValueError('Paired input changed on resume')
        return torch.load(path,map_location='cpu',weights_only=True),binding
    if slot not in teacher_models:
        teacher_models[slot]=build_cotracker(str(weights/plan['pool'][slot]['checkpoint']),
            offline=slot==2,v2=slot==0,window_len=60 if slot==2 else 16).eval().requires_grad_(False)
    model=teacher_models[slot].cuda()
    before=S.state_hashes(model)
    seed_all(s['draw']+43)
    q=s['queries'].cuda(); v=s['video'][None].cuda()
    if slot<2:
        xy=get_points_on_a_grid(5,v.shape[-2:],device='cpu')
        draw=torch.cat((torch.zeros_like(xy[...,:1]),xy),-1).cuda()
    else: draw=get_uniformly_sampled_pts(100,v.shape[1],v.shape[-2:],device='cpu').cuda()
    with torch.no_grad():
        p,vis,c,audit=S.load_protocol(CHECKOUT).teacher_forward(slot,S.TEACHER_TYPES[slot],model,v,q,draw,'control',plan['pool'])
    if S.state_hashes(model)!=before: raise ValueError('Teacher changed')
    if not torch.isfinite(p).all() or not torch.isfinite(vis).all() or not bool((vis>.9).any()):
        raise ValueError('Invalid teacher labels')
    value=dict(tracks=p.cpu(),visibility=vis.cpu(),confidence=c.cpu(),audit=audit)
    S.atomic_torch(path,value); receipt(root,marker,[path],binding=binding)
    model.cpu()
    return value,binding

def train(root,weights,audit,acdc,trackrad,phase,arm,seed,max_step_seconds=180):
    plan=checked(root); gate(root,phase,arm)
    if seed not in ((0,) if phase!='full' else (0,1,2)): raise ValueError('Invalid seed')
    verify(weights,plan['weights'])
    if sha(audit/'data-manifest.json')!=plan['audit_sha256']: raise ValueError('Audit changed')
    verify(trackrad,plan['files']['train-40']); verify(acdc,plan['external_files'])
    folder=root/phase/f'train/seed_{seed}'/arm; folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'verified.json'
    if cached(root,marker): print(f'CACHE VERIFIED {phase}/{seed}/{arm}',flush=True); return
    m,h=new_model(weights,seed,arm); initial=S.state_hashes(m)
    frozen={n:v for n,v in initial.items() if not n.startswith(PREFIX)}
    hinit=S.state_hashes(h) if h else {}
    m.cuda()
    if h is not None: h.cuda().train()
    params=[p for p in m.parameters() if p.requires_grad]+(list(h.parameters()) if h is not None else [])
    opt=torch.optim.AdamW(params,lr=RULE['lr'],weight_decay=RULE['weight_decay'])
    checkpoint=folder/'resume.pth'; records=[]; steps=steps_for(phase)
    if checkpoint.exists():
        c=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if c['manifest']!=sha(root/'manifest.json') or (c['phase'],c['arm'],c['seed'])!=(phase,arm,seed):
            raise ValueError('Stale resume checkpoint')
        m.load_state_dict(c['model'],strict=True)
        if h is not None: h.load_state_dict(c['region_head'],strict=True)
        opt.load_state_dict(c['optimizer']); records=c['records']
        if [r['step'] for r in records]!=list(range(len(records))) or len(records)>steps:
            raise ValueError('Invalid committed step history')
    teacher_models={}; audit_data=read(audit/'data-manifest.json')
    for step in range(len(records),steps):
        start=time.perf_counter(); s=sample(plan,audit_data,acdc,trackrad,seed,step)
        label,binding=labels(root,plan,weights,s,seed,step,teacher_models)
        seed_all(s['draw']+47)
        video=s['video'].cuda(); q=s['queries'].cuda(); masks=s['masks'].cuda(); ann=s['annotated'].cuda()
        target={k:label[k].cuda() for k in ('tracks','visibility')}
        opt.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
        out=m(video=video[None],queries=q,iters=RULE['iterations'],is_train=True)
        pred=torch.cat((q[0,:,1:][None],out[0][0,1:]),dim=0)  # match deployment's query-frame lock
        point=S.training_loss(out,target) if arm!='G1' else pred.sum()*0
        weak=boundary_loss(pred,masks,ann,target['visibility'][0]>.9,RULE['weak_distance_scale_pixels']) if arm!='M0' else pred.sum()*0
        region=decode_training(m,h,video,masks,pred,ann) if h is not None else pred.sum()*0
        loss=point+RULE['weak_weight']*weak+RULE['region_weight']*region
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        loss.backward()
        grads=[n for n,p in m.named_parameters() if p.grad is not None]
        if not grads or any(not n.startswith(PREFIX) for n in grads): raise ValueError('Wrong/frozen gradient path')
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params): raise ValueError('Nonfinite gradient')
        norm=torch.nn.utils.clip_grad_norm_(params,10); opt.step(); torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        record=dict(step=step,**binding,label_tracks=S.tensor_sha(label['tracks']),label_visibility=S.tensor_sha(label['visibility']),
            label_confidence=S.tensor_sha(label['confidence']),loss=float(loss.detach()),point=float(point.detach()),
            weak=float(weak.detach()),region=float(region.detach()),grad_norm=float(norm),seconds=elapsed,
            max_memory_bytes=torch.cuda.max_memory_allocated(),annotated_target_frames=int(ann[1:].sum()),
            empty_annotated_target_frames=int(((masks[1:].sum((1,2,3))==0)&ann[1:]).sum()),
            teacher_visible_fraction=float((target['visibility']>.9).float().mean()),
            query_spread_pixels=float(pred[0].detach().std(dim=0).norm()),
            mean_predicted_spread_pixels=float(pred[1:].detach().std(dim=1).norm(dim=-1).mean()))
        records.append(record)
        # Resume state and records share one atomic boundary. Replayed work cannot add duplicate rows.
        if (step+1)%25==0 or step+1==steps or phase=='smoke':
            S.atomic_torch(checkpoint,dict(model=m.state_dict(),region_head=h.state_dict() if h else None,
                optimizer=opt.state_dict(),records=records,manifest=sha(root/'manifest.json'),phase=phase,arm=arm,seed=seed))
        if phase=='smoke' or step%25==0: print(f'TRAIN {phase} seed={seed} arm={arm} step={step+1}/{steps} loss={record["loss"]:.5f} sec={elapsed:.2f}',flush=True)
        if elapsed>max_step_seconds: raise ValueError(f'Runtime guard: {elapsed:.1f}s > {max_step_seconds}s; retain results')
        del out,pred,loss,point,weak,region,video,q,masks,ann,target
    after=S.state_hashes(m)
    if any(after[n]!=v for n,v in frozen.items()): raise ValueError('Frozen backbone changed')
    if not any(after[n]!=initial[n] for n in after if n.startswith(PREFIX)): raise ValueError('Sidecar did not update')
    if h and S.state_hashes(h)==hinit: raise ValueError('Region head did not update')
    final=folder/'final.pth'
    S.atomic_torch(final,dict(model=m.cpu().state_dict(),feature_sidecar='mamba',backbone_architecture='base',
        freeze_base_for_inference=True,region_head=h.cpu().state_dict() if h else None,arm=arm,seed=seed,steps=steps,
        manifest=sha(root/'manifest.json')))
    save(folder/'steps.json',records)
    save(folder/'proof.json',dict(frozen_backbone_exact=True,initial_sidecar={n:v for n,v in initial.items() if n.startswith(PREFIX)},
        initial_head=hinit,parameters=sum(p.numel() for p in m.parameters())+(sum(p.numel() for p in h.parameters()) if h else 0),
        trainable_parameters=sum(p.numel() for p in params),steps=steps,domain_counts={d:sum(r['domain']==d for r in records) for d in ('acdc','trackrad')}))
    # Actual serialized checkpoint reload + forward, not only a state-key test.
    from resources.model import setup_model
    from cotracker.models.core.cotracker.feature_sidecar import configure_sidecar_runtime
    del m,opt,params,teacher_models; torch.cuda.empty_cache()
    loaded=setup_model(str(final),device='cuda'); configure_sidecar_runtime(loaded,backend='compact')
    with torch.no_grad():
        p=loaded(video=s['video'][None].cuda(),queries=s['queries'].cuda(),iters=2)[0][0]
        if not torch.isfinite(p).all(): raise ValueError('Reload inference failed')
        if h is not None:
            h2=RegionHead().cuda().eval(); h2.load_state_dict(torch.load(final,map_location='cpu',weights_only=True)['region_head'])
            feat=features(loaded,s['video'][:2].cuda())
            logits=h2(feat,s['masks'][0].cuda(),p[:2],feat[:1],p[0])
            if not torch.isfinite(logits).all(): raise ValueError('Region reload inference failed')
    receipt(root,marker,[final,folder/'steps.json',folder/'proof.json'],arm=arm,seed=seed,phase=phase,steps=steps,reload_gpu_forward=True)
    print(f'FINISHED TRAINING {phase}/{seed}/{arm}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['smoke','pilot','full'],required=True)
    p.add_argument('--arm',choices=ARMS,required=True); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-step-seconds',type=float,default=180); a=p.parse_args()
    train(Path('/study'),Path('/weights'),Path('/audit'),Path('/acdc'),Path('/data'),a.phase,a.arm,a.seed,a.max_step_seconds)
