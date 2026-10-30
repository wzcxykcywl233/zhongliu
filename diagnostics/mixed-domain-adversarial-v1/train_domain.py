"""Fresh paired students; two optimizers; atomic records and optimizer resume."""
import time
import torch
from protocol import *
import train as T
from domain import configure, Tap, DomainHead, domain_loss, schedule, gradient_norm


def input_labels(previous, plan, weights, sample, seed, step, reference):
    path = previous/f'labels/seed_{seed}/{step:05d}.json'
    p = C.cached(previous,path)
    if not p: raise ValueError('Missing precomputed label; old results are read-only: '+str(path))
    short = p['binding']['slot'] < 2 and len(p['binding']['sampling']['native_frames']) <= 8
    binding = 'region-short-teacher-fix-v1.json'
    if short and p['files'].get(binding) != C.sha(previous/binding):
        raise ValueError('Short teacher label lacks repaired provenance')
    label, b = T.labels(previous,plan,weights,sample,seed,step,{})
    row = dict(step=step, **b, **{'label_'+k:T.S.tensor_sha(label[k])
                               for k in ('tracks','visibility','confidence')})
    if {k:row[k] for k in PAIR_KEYS} != {k:reference[k] for k in PAIR_KEYS}:
        raise ValueError('Original M0 input/label sequence mismatch')
    if not all(torch.isfinite(label[k]).all() for k in ('tracks','visibility','confidence')):
        raise ValueError('Nonfinite cached labels')
    return label, row


def optimize(task, adversarial, student, classifier, opt, disc_opt):
    loss = task + adversarial
    if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite objective')
    loss.backward()
    for params in (student,classifier):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise ValueError('Nonfinite gradient')
    student_norm = float(torch.nn.utils.clip_grad_norm_(student,RULE['student_clip']))
    disc_norm = float(torch.nn.utils.clip_grad_norm_(classifier,RULE['classifier_clip'])) if classifier else 0.
    opt.step()
    if disc_opt is not None: disc_opt.step()
    return student_norm, disc_norm


def train(root,previous,weights,audit,acdc,data,phase,arm,seed):
    checked(root,previous); gate(root,phase,arm,seed)
    plan = C.read(previous/'manifest.json')
    C.verify(weights,plan['weights']); C.verify(data,plan['files']['train-40'])
    C.verify(acdc,plan['external_files'])
    if C.sha(audit/'data-manifest.json') != plan['audit_sha256']: raise ValueError('Audit changed')
    _, reference, original_proof = old_training(previous,seed)
    folder = root/phase/f'train/seed_{seed}'/arm; folder.mkdir(parents=True,exist_ok=True)
    if cached(root,folder/'verified.json'):
        print(f'CACHE VERIFIED {phase}/{seed}/{arm}',flush=True); return
    model, _ = T.new_model(weights,seed,'M0'); partial = configure(model,arm)
    initial = T.S.state_hashes(model)
    if {n:v for n,v in initial.items() if n.startswith(PREFIX)} != original_proof['initial_sidecar']:
        raise ValueError('Initial Mamba differs from original M0')
    train_names = [n for n,p in model.named_parameters() if p.requires_grad]
    frozen = {n:v for n,v in initial.items() if n not in train_names}
    model.cuda(); side = [p for n,p in model.named_parameters() if n.startswith(PREFIX)]
    encoder = [p for n,p in model.named_parameters() if n.startswith(ENCODER_PREFIXES)] if partial else []
    params = side+encoder
    groups = [dict(params=side,lr=RULE['sidecar_lr'])]
    if encoder: groups.append(dict(params=encoder,lr=RULE['encoder_lr']))
    opt = torch.optim.AdamW(groups,weight_decay=RULE['weight_decay'])
    kind, maximum = ARMS[arm]; head = None; tap = None; disc_opt = None; classifier = []
    if kind != 'none':
        # Classifier initialization cannot consume the student's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(C.RULE['seed_base']+seed+991)
            head = DomainHead(256 if kind == 'sidecar' else model.latent_dim)
        head.cuda().train(); classifier = list(head.parameters())
        disc_opt = torch.optim.AdamW(classifier,lr=RULE['classifier_lr'],weight_decay=RULE['weight_decay'])
        tap = Tap(model,kind)
    records = []; checkpoint = folder/'resume.pth'; count = RULE[phase+'_steps']
    if checkpoint.exists():
        c = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if (c['plan'],c['phase'],c['arm'],c['seed']) != (C.sha(root/'plan.json'),phase,arm,seed):
            raise ValueError('Wrong resume identity')
        model.load_state_dict(c['model']); opt.load_state_dict(c['optimizer'])
        if head is not None:
            head.load_state_dict(c['domain_head']); disc_opt.load_state_dict(c['domain_optimizer'])
        records = c['records']
        if [r['step'] for r in records] != list(range(len(records))) or len(records)>count:
            raise ValueError('Broken resume step sequence')
        for r in records:
            if any(r[k]!=reference[r['step']][k] for k in PAIR_KEYS):
                raise ValueError('Resume contains unpaired input')
    audit_data = C.read(audit/'data-manifest.json')
    try:
        for step in range(len(records),count):
            start = time.perf_counter()
            s = T.sample(plan,audit_data,acdc,data,seed,step)
            label, row = input_labels(previous,plan,weights,s,seed,step,reference[step])
            T.seed_all(s['draw']+47)
            opt.zero_grad(set_to_none=True)
            if disc_opt is not None: disc_opt.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            if tap: tap.reset(len(s['video']))
            out = model(video=s['video'][None].cuda(),queries=s['queries'].cuda(),
                        iters=C.RULE['iterations'],is_train=True)
            target = {k:label[k].cuda() for k in ('tracks','visibility')}
            task = T.S.training_loss(out,target)
            strength = schedule(step,maximum); adversarial = task.new_zeros(())
            probability = None; embedding = None; diag = {}
            if head is not None:
                embedding = tap.embedding()
                adversarial, prob = domain_loss(head,embedding,s['domain'],strength)
                probability = float(prob.item())
                diag = dict(feature_rms=float(embedding.detach().square().mean().sqrt()),
                            feature_channel_std=float(embedding.detach().std()),
                            pooled_feature=embedding.detach().cpu().flatten().tolist(),
                            domain_correct=(probability>=.5)==(s['domain']=='acdc'))
            # Audited real task/GRL paths, not just requires_grad flags. Extra backward only
            # during six-step smoke and every 100 steps in larger experiments.
            if phase == 'smoke' or step%100==1:
                if encoder:
                    diag['encoder_task_grad_norm'] = gradient_norm(task,encoder)
                    if diag['encoder_task_grad_norm'] <= 0: raise ValueError('Task cannot update encoder')
                if head is not None and strength>0:
                    recipient = side if kind=='sidecar' else encoder
                    diag['adversarial_grad_norm'] = gradient_norm(adversarial,recipient)
                    if diag['adversarial_grad_norm'] <= 0: raise ValueError('GRL recipient has no gradient')
            sn, dn = optimize(task,adversarial,params,classifier,opt,disc_opt)
            if any(p.grad is not None for n,p in model.named_parameters() if n not in train_names):
                raise ValueError('Unexpected frozen gradient')
            torch.cuda.synchronize(); elapsed = time.perf_counter()-start
            row.update(point=float(task.detach()),domain_loss=float(adversarial.detach()),
                grl_strength=strength,domain_probability=probability,student_grad_norm=sn,
                classifier_grad_norm=dn,seconds=elapsed,max_memory_bytes=torch.cuda.max_memory_allocated(),**diag)
            records.append(row)
            if phase=='smoke' or (step+1)%25==0 or step+1==count:
                T.S.atomic_torch(checkpoint,dict(model=model.state_dict(),optimizer=opt.state_dict(),
                    domain_head=head.state_dict() if head is not None else None,
                    domain_optimizer=disc_opt.state_dict() if disc_opt is not None else None,
                    plan=C.sha(root/'plan.json'),phase=phase,arm=arm,seed=seed,records=records))
            if phase=='smoke' or step%25==0:
                print(f'TRAIN {phase} seed={seed} {arm} {step+1}/{count} point={row["point"]:.5f} domain={row["domain_loss"]:.4f} sec={elapsed:.2f}',flush=True)
            if elapsed>180: raise ValueError('Step runtime >180s; preserve partial work, do not scale up')
            del out, target, task, adversarial, embedding
            if tap: tap.values.clear()
    finally:
        if tap: tap.close()
    after = T.S.state_hashes(model)
    if any(after[n]!=v for n,v in frozen.items()): raise ValueError('Frozen tensors changed')
    if not any(after[n]!=initial[n] for n in train_names if n.startswith(PREFIX)):
        raise ValueError('Sidecar did not update')
    if partial and not any(after[n]!=initial[n] for n in train_names if n.startswith(ENCODER_PREFIXES)):
        raise ValueError('Encoder branch did not update')
    final = folder/'final.pth'
    T.S.atomic_torch(final,dict(model=model.cpu().state_dict(),feature_sidecar='mamba',
        backbone_architecture='base',freeze_base_for_inference=True,region_head=None,
        arm=arm,seed=seed,steps=count,plan=C.sha(root/'plan.json')))
    C.save(folder/'steps.json',records)
    C.save(folder/'proof.json',dict(frozen_tensors_exact=True,partial_encoder=partial,
        trainable_names=train_names,initial_sidecar=original_proof['initial_sidecar'],
        student_state_hashes=after,steps=count,
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_student_parameters=sum(p.numel() for p in params),
        training_only_classifier_parameters=sum(p.numel() for p in classifier),
        domain_counts={d:sum(r['domain']==d for r in records) for d in ('acdc','trackrad')}))
    # Classifier is excluded from final.pth; only resume.pth retains it.
    del model,opt,params,side,encoder,head,disc_opt,classifier,tap
    torch.cuda.empty_cache()
    from resources.model import setup_model
    from cotracker.models.core.cotracker.feature_sidecar import configure_sidecar_runtime
    loaded = setup_model(str(final),device='cuda').requires_grad_(False)
    configure_sidecar_runtime(loaded,backend='compact')
    if T.S.state_hashes(loaded)!=after: raise ValueError('Serialized student state changed')
    # Even after a complete resume, use a reproducible actual input for reload verification.
    s = T.sample(plan,audit_data,acdc,data,seed,count-1)
    with torch.no_grad():
        prediction = loaded(video=s['video'][None].cuda(),queries=s['queries'].cuda(),iters=2)[0]
    if not torch.isfinite(prediction).all(): raise ValueError('GPU reload inference failed')
    receipt(root,folder/'verified.json',[final,folder/'steps.json',folder/'proof.json'],
            phase=phase,arm=arm,seed=seed,steps=count,reload_gpu_forward=True)
    print(f'FINISHED TRAINING {phase}/{seed}/{arm}',flush=True)


def audit_training(root,previous,phase,require_inference=False):
    checked(root,previous); files=[]
    for seed in seeds_for(phase):
        _, reference, refproof = old_training(previous,seed); proofs={}
        for arm in arms_for(root,phase):
            folder=root/phase/f'train/seed_{seed}'/arm
            r=cached(root,folder/'verified.json')
            if not r or (r['phase'],r['arm'],r['seed'],r['steps']) != (phase,arm,seed,RULE[phase+'_steps']) or not r['reload_gpu_forward']:
                raise ValueError('Missing training handoff')
            rows=C.read(folder/'steps.json'); info=C.read(folder/'proof.json'); proofs[arm]=info
            if [x['step'] for x in rows]!=list(range(RULE[phase+'_steps'])): raise ValueError('Bad step history')
            if any(any(r[k]!=reference[i][k] for k in PAIR_KEYS) for i,r in enumerate(rows)):
                raise ValueError('Unpaired training')
            if not info['frozen_tensors_exact'] or info['initial_sidecar']!=refproof['initial_sidecar']:
                raise ValueError('Frozen/initialization audit failed')
            if info['domain_counts']!={'acdc':len(rows)//2,'trackrad':len(rows)//2}:
                raise ValueError('Unbalanced domains')
            if require_inference:
                f=root/phase/'validation-10'/f'seed_{seed}'/arm
                p=cached(root,f/'inference-verified.json')
                if not p or len(p['ids'])!=2: raise ValueError('Deployment smoke required')
                files.append(f/'inference-verified.json')
            files += [folder/n for n in ('verified.json','proof.json','steps.json','final.pth')]
        for zero,control in [('S0','M0'),('Fprobe','F0')]:
            if zero in proofs and proofs[zero]['student_state_hashes']!=proofs[control]['student_state_hashes']:
                raise ValueError('Zero-adversarial control changed student: '+zero)
    if phase!='smoke' or require_inference:
        receipt(root,root/phase/'audit.json',files,phase=phase,paired=True,
                zero_controls_checked=['S0/M0','Fprobe/F0'] if phase!='full' else [],
                zero_controls_prerequisite='smoke + pilot')
    print('AUDIT PASSED '+phase,flush=True)
