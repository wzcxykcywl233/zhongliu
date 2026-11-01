"""Fresh full runs; reuse validated objective and low-memory implementation."""
import time
import math
import torch
import full_context as X
H,V,B,P,C,T,S = X.H,X.V,X.B,X.P,X.C,X.T,X.S


def make_model(weights, arm, seed):
    model, _ = T.new_model(weights, seed, 'M0'); partial = V.configure(model, arm)
    if partial: V.install(model)
    model.cuda()
    side = [p for n,p in model.named_parameters() if n.startswith(P.PREFIX)]
    enc = [p for n,p in model.named_parameters() if n.startswith(P.ENCODER_PREFIXES) and p.requires_grad]
    groups = [dict(params=side, lr=X.RULE['sidecar_lr'])]
    if enc: groups.append(dict(params=enc, lr=X.RULE['encoder_lr']))
    opt = torch.optim.AdamW(groups, weight_decay=X.RULE['weight_decay'])
    head = disc = tap = None
    if arm not in ('M0','F0'):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(C.RULE['seed_base']+seed+991); head = V.DomainHead(model.latent_dim)
        head.cuda().train(); disc = torch.optim.AdamW(head.parameters(), lr=X.RULE['classifier_lr'], weight_decay=X.RULE['weight_decay'])
        tap = V.Tap(model, 'fnet')
    return model,opt,head,disc,tap,side+enc


def validate(rows, count):
    if [r['step'] for r in rows] != list(range(len(rows))) or len(rows)>count: raise ValueError('Broken step history')
    for r in rows:
        for name in ('point','domain_loss','student_grad_norm','classifier_grad_norm','grl_strength'):
            if not math.isfinite(r[name]) or r[name]<0: raise ValueError('Invalid training value: '+name)
        if not 0<r['seconds']<=X.RULE['step_seconds_limit']: raise ValueError('Step exceeds180s/invalid; cannot skip on resume')
        m=r['runtime_memory']
        if not math.isfinite(m['device_total']) or not 0<m['peak_allocated']<=X.RULE['maximum_device_fraction']*m['device_total']:
            raise ValueError('Memory margin violated; cannot skip on resume')


def frozen_check(model, frozen):
    state=T.S.state_hashes(model)
    if any(state[n]!=v for n,v in frozen.items()): raise ValueError('Frozen tensors changed')
    return state


def reload_check(final, after, study, lld, data, seed, count):
    from resources.model import setup_model
    from cotracker.models.core.cotracker.feature_sidecar import configure_sidecar_runtime
    model=setup_model(str(final),device='cuda').requires_grad_(False)
    configure_sidecar_runtime(model,backend='compact')
    if T.S.state_hashes(model)!=after: raise ValueError('Serialized student changed')
    s=V.sample(study,lld,data,seed,count-1)
    with torch.no_grad(): output=model(video=s['video'][None].cuda(),queries=s['queries'].cuda(),iters=2)
    if any(not torch.isfinite(v).all() for v in output[:3]): raise ValueError('Reload GPU inference failed')


def train(root,previous,weights,audit,lld,data,arm,seed):
    study=X.checked(root,previous); X.gate(root,'full',arm,seed)
    old=C.read(previous/'manifest.json'); C.verify(weights,old['weights']); C.verify(data,old['files']['train-40'])
    if C.sha(audit/'data-manifest.json')!=study['external_manifest_sha256']: raise ValueError('External audit changed')
    _,reference,refproof=P.old_training(previous,seed)
    out=root/f'full/train/seed_{seed}'/arm; out.mkdir(parents=True,exist_ok=True)
    if P.cached(root,out/'verified.json'): print(f'CACHE VERIFIED full/{seed}/{arm}',flush=True); return
    if (out/'failure.json').exists(): raise ValueError('Previous explicit stability failure requires review')
    model,opt,head,disc,tap,params=make_model(weights,arm,seed)
    initial=T.S.state_hashes(model); names=[n for n,p in model.named_parameters() if p.requires_grad]
    frozen={n:h for n,h in initial.items() if n not in names}
    if {n:h for n,h in initial.items() if n.startswith(P.PREFIX)}!=refproof['initial_sidecar']:
        raise ValueError('Fresh sidecar differs from paired original reference')
    count=X.RULE['full_steps']; rows=[]; checkpoint=out/'resume.pth'
    def save():
        T.S.atomic_torch(checkpoint,dict(plan=C.sha(root/'plan.json'),environment=C.sha(root/'environment.json'),
            arm=arm,seed=seed,initial=initial,rows=rows,state=S.capture(model,opt,head,disc)))
    try:
        if checkpoint.exists():
            c=H.load(checkpoint)
            if (c['plan'],c['environment'],c['arm'],c['seed'],c['initial'])!=(C.sha(root/'plan.json'),C.sha(root/'environment.json'),arm,seed,initial):
                raise ValueError('Resume identity mismatch')
            validate(c['rows'],count)
            for r in c['rows']:
                binding=C.read(root/f'inputs/seed_{seed}/{r["step"]:05d}.json')
                if any(r[k]!=binding[k] for k in P.PAIR_KEYS): raise ValueError('Resume pairing changed')
            S.restore(c['state'],model,opt,head,disc); rows=c['rows']; del c
            frozen_check(model,frozen)
            print(f'RESUME full/{seed}/{arm} completed={len(rows)}',flush=True)
        else: save()
        for step in range(len(rows),count):
            torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
            def progress(phases):
                C.save(out/'phase-profile.json',dict(seed=seed,arm=arm,step=step,**phases.report()))
            phases=V.Phases(progress)
            with phases.part('data_read_and_sampling'): s=V.sample(study,lld,data,seed,step)
            with phases.part('label_binding'):
                label,row=B.BASE.input_labels(root,previous,old,weights,s,seed,step,reference[step])
            metrics,_=V.execute(model,s,label,step,arm,params,list(head.parameters()) if head is not None else [],
                opt,disc,head,tap,phases,step in (0,1) or step%100 in (0,1),evidence=False)
            torch.cuda.synchronize(); row.update(metrics,seconds=time.perf_counter()-start,
                max_memory_bytes=torch.cuda.max_memory_allocated(),runtime_memory=V.memory(),**phases.report())
            rows.append(row)
            validate([dict(row,step=0)],1)
            if any(not torch.isfinite(p).all() for p in params): raise ValueError('Nonfinite updated student')
            if head is not None and any(not torch.isfinite(p).all() for p in head.parameters()): raise ValueError('Nonfinite classifier')
            if (step+1)%X.RULE['checkpoint_every']==0 or step+1==count:
                frozen_check(model,frozen); save(); C.save(out/'steps.json',rows)
                print(f'TRAIN full/{seed}/{arm} {step+1}/{count} point={row["point"]:.6f} sec={row["seconds"]:.2f} peak={row["max_memory_bytes"]/2**30:.2f}GiB',flush=True)
        after=frozen_check(model,frozen)
        if not any(after[n]!=initial[n] for n in names if n.startswith(P.PREFIX)): raise ValueError('Sidecar did not update')
        if arm.startswith('F') and not any(after[n]!=initial[n] for n in names if n.startswith(P.ENCODER_PREFIXES)):
            raise ValueError('Encoder did not update')
        final=out/'final.pth'
        T.S.atomic_torch(final,dict(model=model.cpu().state_dict(),feature_sidecar='mamba',backbone_architecture='base',
            freeze_base_for_inference=True,region_head=None,arm=arm,seed=seed,steps=count,plan=C.sha(root/'plan.json')))
        C.save(out/'steps.json',rows)
        C.save(out/'proof.json',dict(frozen_tensors_exact=True,partial_encoder=arm.startswith('F'),trainable_names=names,
            initial_state_hashes=initial,initial_sidecar=refproof['initial_sidecar'],student_state_hashes=after,steps=count,
            parameters=sum(p.numel() for p in model.parameters()),trainable_student_parameters=sum(p.numel() for p in params),
            training_only_classifier_parameters=sum(p.numel() for p in head.parameters()) if head is not None else 0,
            domain_counts={d:sum(r['domain']==d for r in rows) for d in ('trackrad','lld_synthetic')}))
    except (ValueError,FloatingPointError,AssertionError) as exc:
        C.save(out/'failure.json',dict(completed_updates=len(rows),error=str(exc),review_required=True)); raise
    finally:
        if tap: tap.close()
    del model,opt,head,disc,tap,params; torch.cuda.empty_cache()
    reload_check(final,after,study,lld,data,seed,count)
    P.receipt(root,out/'verified.json',[final,out/'steps.json',out/'proof.json',out/'phase-profile.json'],
        phase='full',arm=arm,seed=seed,steps=count,reload_gpu_forward=True)
    print(f'FINISHED TRAINING full/{seed}/{arm}',flush=True)


def audit_training(root,previous,phase='full',require_inference=False):
    X.checked(root,previous); X.arms_for(root,phase)
    if P.cached(root,root/'full/audit.json'): print('CACHE VERIFIED full paired audit',flush=True); return
    files=[]; count=X.RULE['full_steps']
    for seed in X.RULE['seeds']:
        _,old_rows,refproof=P.old_training(previous,seed); paired=None; initialization=None
        for arm in X.ARMS:
            out=root/f'full/train/seed_{seed}'/arm; r=P.cached(root,out/'verified.json')
            if not r or (r['phase'],r['arm'],r['seed'],r['steps'],r['reload_gpu_forward'])!=('full',arm,seed,count,True):
                raise ValueError('Incomplete full checkpoint')
            rows=C.read(out/'steps.json'); info=C.read(out/'proof.json'); validate(rows,count)
            if len(rows)!=count: raise ValueError('Incomplete step history')
            if not info['frozen_tensors_exact'] or info['initial_sidecar']!=refproof['initial_sidecar']:
                raise ValueError('Frozen/initialization proof failed')
            if initialization is None: initialization=info['initial_state_hashes']
            elif initialization!=info['initial_state_hashes']: raise ValueError('Initial models differ')
            bindings=[{k:r[k] for k in P.PAIR_KEYS} for r in rows]
            if paired is None: paired=bindings
            elif paired!=bindings: raise ValueError('Unpaired inputs/labels')
            for i,row in enumerate(rows):
                bound=C.read(root/f'inputs/seed_{seed}/{i:05d}.json')
                if any(row[k]!=bound[k] for k in P.PAIR_KEYS): raise ValueError('Input binding changed')
                if i%2==0 and any(row[k]!=old_rows[i][k] for k in P.TRACKRAD_PAIR_KEYS): raise ValueError('TrackRAD label stream changed')
            if info['domain_counts']!={'trackrad':count//2,'lld_synthetic':count//2}: raise ValueError('Unbalanced updates')
            files += [out/n for n in ('verified.json','final.pth','steps.json','proof.json')]
    P.receipt(root,root/'full/audit.json',files,phase='full',paired=True,
        zero_control_graph='prior stability graph proof; no bitwise CUDA equality assumption')
    print('AUDIT PASSED full; paired three-seed inputs/labels and frozen tensors',flush=True)
