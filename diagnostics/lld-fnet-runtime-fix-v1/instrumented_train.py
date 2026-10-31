"""Fresh paired students; two optimizers; atomic records and optimizer resume."""
import time
import math
import torch
from protocol import *
import train as T
from lld_data import sample
from domain import configure, Tap, DomainHead, domain_loss, schedule, gradient_norm
from runtime import install, CONFIG
from measure import Phases, memory
from engine import execute


def validate_runtime_records(records):
    for row in records:
        seconds = row.get('seconds', float('nan'))
        if not math.isfinite(seconds) or seconds <= 0 or seconds > 180:
            raise ValueError('Recorded step failed the 180s runtime guard; do not skip it by resuming. Preserve and diagnose.')


def old_labels(previous, plan, weights, sample, seed, step, reference):
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
    if {k:row[k] for k in TRACKRAD_PAIR_KEYS} != {k:reference[k] for k in TRACKRAD_PAIR_KEYS}:
        raise ValueError('Original M0 input/label sequence mismatch')
    if not all(torch.isfinite(label[k]).all() for k in ('tracks','visibility','confidence')):
        raise ValueError('Nonfinite cached labels')
    return label, row


def input_labels(root, previous, plan, weights, sample, seed, step, reference):
    if sample['domain']=='trackrad':
        label,row=old_labels(previous,plan,weights,sample,seed,step,reference)
        row['label_origin']='cached_frozen_teacher'
    else:
        if sample['domain']!='lld_synthetic' or step%2!=1: raise ValueError('Wrong external domain')
        label=sample['label']
        row=dict(step=step,video=T.S.tensor_sha(sample['video']),queries=T.S.tensor_sha(sample['queries']),
            sampling=sample['sampling'],domain=sample['domain'],slot=-1,draw=sample['draw'],
            label_origin='known_synthetic_affine',
            **{'label_'+k:T.S.tensor_sha(label[k]) for k in ('tracks','visibility','confidence')})
        if not all(torch.isfinite(label[k]).all() for k in ('tracks','visibility','confidence')):
            raise ValueError('Nonfinite synthetic target')
        if not torch.equal(label['tracks'][:,0],sample['queries'][...,1:]): raise ValueError('Broken anchor')
    # Shared immutable input/label fingerprints across every arm and restart.
    C.freeze(root/f'inputs/seed_{seed}/{step:05d}.json',row)
    return label,row


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


def train(root,previous,weights,audit,lld,data,phase,arm,seed):
    study=checked(root,previous); gate(root,phase,arm,seed)
    plan = C.read(previous/'manifest.json')
    C.verify(weights,plan['weights']); C.verify(data,plan['files']['train-40'])
    if C.sha(audit/'data-manifest.json') != study['external_manifest_sha256']: raise ValueError('LLD audit changed')
    _, reference, original_proof = old_training(previous,seed)
    folder = root/phase/f'train/seed_{seed}'/arm; folder.mkdir(parents=True,exist_ok=True)
    if cached(root,folder/'verified.json'):
        print(f'CACHE VERIFIED {phase}/{seed}/{arm}',flush=True); return
    model, _ = T.new_model(weights,seed,'M0'); partial = configure(model,arm)
    if partial: install(model)
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
        validate_runtime_records(records)
        if [r['step'] for r in records] != list(range(len(records))) or len(records)>count:
            raise ValueError('Broken resume step sequence')
        for r in records:
            bound=C.read(root/f'inputs/seed_{seed}/{r["step"]:05d}.json')
            if any(r[k]!=bound[k] for k in PAIR_KEYS):
                raise ValueError('Resume contains unpaired input')
    try:
        for step in range(len(records),count):
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            def progress(phases):
                C.save(folder/'phase-profile.json',dict(phase=phase,arm=arm,seed=seed,step=step,
                    runtime=CONFIG if partial else None,**phases.report()))
            phases = Phases(progress)
            with phases.part('data_read_and_sampling'):
                s = sample(study,lld,data,seed,step)
            with phases.part('label_binding'):
                label, row = input_labels(root,previous,plan,weights,s,seed,step,reference[step])
            metrics, _ = execute(model,s,label,step,arm,params,classifier,opt,disc_opt,
                head,tap,phases,phase=='smoke' or step%100==1)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-start
            row.update(metrics,seconds=elapsed,max_memory_bytes=torch.cuda.max_memory_allocated(),
                **phases.report(),runtime_memory=memory())
            records.append(row)
            if phase=='smoke' or (step+1)%25==0 or step+1==count or elapsed>180:
                # A runtime stop always commits its completed optimizer step.
                # On resume the same record/input audit still runs.
                T.S.atomic_torch(checkpoint,dict(model=model.state_dict(),optimizer=opt.state_dict(),
                    domain_head=head.state_dict() if head is not None else None,
                    domain_optimizer=disc_opt.state_dict() if disc_opt is not None else None,
                    plan=C.sha(root/'plan.json'),phase=phase,arm=arm,seed=seed,records=records))
            if phase=='smoke' or step%25==0:
                print(f'TRAIN {phase} seed={seed} {arm} {step+1}/{count} point={row["point"]:.5f} domain={row["domain_loss"]:.4f} sec={elapsed:.2f} peak={row["max_memory_bytes"]/2**30:.2f}GiB',flush=True)
            if elapsed>180: raise ValueError('Step runtime >180s; preserve partial work, do not scale up')

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
        domain_counts={d:sum(r['domain']==d for r in records) for d in ('lld_synthetic','trackrad')}))
    # Classifier is excluded from final.pth; only resume.pth retains it.
    del model,opt,params,side,encoder,head,disc_opt,classifier,tap
    torch.cuda.empty_cache()
    from resources.model import setup_model
    from cotracker.models.core.cotracker.feature_sidecar import configure_sidecar_runtime
    loaded = setup_model(str(final),device='cuda').requires_grad_(False)
    configure_sidecar_runtime(loaded,backend='compact')
    if T.S.state_hashes(loaded)!=after: raise ValueError('Serialized student state changed')
    # Even after a complete resume, use a reproducible actual input for reload verification.
    s = sample(study,lld,data,seed,count-1)
    with torch.no_grad():
        prediction = loaded(video=s['video'][None].cuda(),queries=s['queries'].cuda(),iters=2)[0]
    if not torch.isfinite(prediction).all(): raise ValueError('GPU reload inference failed')
    receipt(root,folder/'verified.json',[final,folder/'steps.json',folder/'proof.json',folder/'phase-profile.json'],
            phase=phase,arm=arm,seed=seed,steps=count,reload_gpu_forward=True)
    print(f'FINISHED TRAINING {phase}/{seed}/{arm}',flush=True)


def audit_training(root,previous,phase,require_inference=False):
    checked(root,previous); files=[]
    for seed in seeds_for(phase):
        _, old_reference, refproof = old_training(previous,seed); proofs={}
        reference=C.read(root/phase/f'train/seed_{seed}/M0/steps.json')
        for arm in arms_for(root,phase):
            folder=root/phase/f'train/seed_{seed}'/arm
            r=cached(root,folder/'verified.json')
            if not r or (r['phase'],r['arm'],r['seed'],r['steps']) != (phase,arm,seed,RULE[phase+'_steps']) or not r['reload_gpu_forward']:
                raise ValueError('Missing training handoff')
            rows=C.read(folder/'steps.json'); info=C.read(folder/'proof.json'); proofs[arm]=info
            validate_runtime_records(rows)
            if [x['step'] for x in rows]!=list(range(RULE[phase+'_steps'])): raise ValueError('Bad step history')
            if any(any(r[k]!=reference[i][k] for k in PAIR_KEYS) for i,r in enumerate(rows)):
                raise ValueError('Unpaired training')
            for i,row in enumerate(rows):
                expected=C.read(root/f'inputs/seed_{seed}/{i:05d}.json')
                if any(row[k]!=expected[k] for k in PAIR_KEYS): raise ValueError('Input binding changed')
                if i%2==0 and any(row[k]!=old_reference[i][k] for k in TRACKRAD_PAIR_KEYS):
                    raise ValueError('TrackRAD sequence/teachers changed')
            if not info['frozen_tensors_exact'] or info['initial_sidecar']!=refproof['initial_sidecar']:
                raise ValueError('Frozen/initialization audit failed')
            if info['domain_counts']!={'lld_synthetic':len(rows)//2,'trackrad':len(rows)//2}:
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
