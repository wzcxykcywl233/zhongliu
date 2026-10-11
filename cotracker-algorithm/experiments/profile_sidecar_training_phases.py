"""Three REAL diagnostic updates, unchanged 1000-step schedule and verified trainer.

The trainer is instrumented in memory only. These weights are not experiment
results, and synchronized timing must not be read as production throughput.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ext'/'co-tracker'),str(Path(__file__).parent)]
from local_cotracker_source import activate_local_cotracker
activate_local_cotracker()
from audit_backbone_growth import file_sha, records
from cotracker.models.core.cotracker.backbone_growth import save_json
from training_phase_probe import PhaseProbe, instrument


def training_arguments(kind, output, dataset, checkpoints):
    # num_steps remains 1000 for OneCycleLR. Only the two loop stop tests
    # are instrumented to stop after the diagnostic update count.
    return ['--batch_size','1','--num_workers','0','--num_steps','1000',
        '--ckpt_path',str(output),'--model_name','cotracker_three',
        '--experiment_name','seed_0_'+kind,'--sequence_len','60',
        '--traj_per_sample','128','--train_iters','4','--save_freq','100000',
        '--save_every_n_steps','25','--keep_last_checkpoints','3','--save_every_n_epoch','1',
        '--trackrad_data_dir',str(dataset),'--skip_evaluation','--offline_model',
        '--restore_ckpt',str(checkpoints/'scaled_offline.pth'),
        '--teacher_types','cotracker2v1','online_cotracker_three','offline_cotracker_three',
        '--teacher_cotracker2_ckpt',str(checkpoints/'cotracker2v1.pth'),
        '--teacher_online_ckpt',str(checkpoints/'baseline_online.pth'),
        '--teacher_offline_ckpt',str(checkpoints/'baseline_offline.pth'),
        '--auxiliary_teacher_weight','0','--seed','20261008','--paired_step_seed','20261008',
        '--teacher_seed','20261008','--teacher_log_every','1','--confidence_target_mode','hard',
        '--lr','0.00005','--backbone_growth_study','--backbone_architecture','base','--feature_sidecar',kind]


def read_probe_runtime(path):
    if path is None: return None
    runtime=json.loads(path.read_text(encoding='utf-8-sig'))
    expected={'backend':'compact','point_chunk_size':0,'gain':1,'max_correction_pixels':0,'checkpoint_blocks':False}
    if runtime!=expected: raise ValueError('Compact training probe cannot change gains, clipping, chunks or block checkpoint')
    return runtime


def check_training_headroom(row):
    memory=row['memory_after']
    if memory.get('peak_allocated_mib',float('inf'))>20480 or memory.get('peak_reserved_mib',float('inf'))>20480 or memory.get('driver_free_mib',0)<2048:
        raise ValueError('Compact real training exceeded 20 GiB / 2 GiB free headroom; preserve partial report, do not continue')


def run(args):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Diagnostic requires exactly one visible CUDA device')
    if int(os.environ.get('WORLD_SIZE','1'))!=1: raise ValueError('Distributed timing is unsupported')
    if any(os.environ.get(n) for n in ('TRACKRAD_SIDECAR_RUNTIME','COTRACKER_CHECKPOINT')):
        raise ValueError('Unexpected model/runtime override')
    runtime=read_probe_runtime(args.runtime_config)
    trainer=ROOT/'ext'/'co-tracker'/'train_on_real_data.py'
    if file_sha(trainer)!=args.trainer_sha256.lower(): raise ValueError('Original trainer changed')
    args.output.mkdir(parents=True,exist_ok=True)
    if list(args.output.glob('*.pth')) or (args.output/'paired-steps.jsonl').exists():
        raise ValueError('Partial diagnostic training exists; preserve it and use a new results root. No resume mixing.')
    source=trainer.read_text(encoding='utf-8')
    compiled,anchors=instrument(source,str(trainer))
    probe=PhaseProbe(args.steps)
    report={'complete':False,'kind':args.kind,'diagnostic_steps':args.steps,'scheduler_num_steps':1000,
        'trainer_sha256':file_sha(trainer),'instrumentation_sha256':file_sha(Path(__file__).with_name('training_phase_probe.py')),
        'timing_anchors':anchors,'device':torch.cuda.get_device_name(),'torch':torch.__version__,
        'note':'Intrusive synchronized phase timing. Student exclusive time includes backbone and other student operations; backward_total includes gradients through frozen operations and cannot isolate sidecar backward. No official performance or throughput claim.'}
    config={'kind':args.kind,'profile':'seed_0_'+args.kind,
        'steps':args.steps,'scheduler_num_steps':1000,'trainer_sha256':file_sha(trainer),
        'sequence_len':60,'traj_per_sample':128,'train_iters':4,'lr':.00005,
        'training_seed':20261008,'paired_step_seed':20261008,'teacher_seed':20261008,
        'dataset':str(args.dataset),'confidence_target_mode':'hard','auxiliary_teacher_weight':0,'mask_loss_weight':0}
    if runtime is not None:
        config['sidecar_runtime']=runtime;report['sidecar_runtime']=runtime
        report['runtime_config_sha256']=file_sha(args.runtime_config)
    save_json(args.output/'run-config.json',config)
    save_json(args.output/'phase-profile.json',report)
    def save_progress():
        report.update({'rows':probe.rows,'outside_steps':probe.outside_steps,'timing_errors':probe.timing_errors})
        save_json(args.output/'phase-profile.json',report)
        if runtime is not None and probe.rows[-1]['complete'] and not probe.timing_errors:
            check_training_headroom(probe.rows[-1])
    probe.on_row=save_progress
    argv,workdir=sys.argv,Path.cwd()
    start=time.perf_counter()
    try:
        if runtime is not None: os.environ['TRACKRAD_SIDECAR_RUNTIME']=str(args.runtime_config.resolve())
        sys.argv=[str(trainer)]+training_arguments(args.kind,args.output,args.dataset,args.checkpoints)
        os.chdir(trainer.parent)
        with probe.installed():
            exec(compiled,{'__name__':'__main__','__file__':str(trainer),'__probe__':probe})
        paired=records(args.output/'paired-steps.jsonl',args.steps,'base')
        if probe.timing_errors or len(probe.rows)!=args.steps or any(not r['complete'] or r['optimizer_updates']!=1 for r in probe.rows):
            raise ValueError('Diagnostic update coverage differs')
        if [r['step'] for r in probe.rows]!=list(range(args.steps)): raise ValueError('Wrong step order')
        final=torch.load(args.output/'cotracker_three_final.pth',map_location='cpu',weights_only=True)
        if final['total_steps']!=args.steps: raise ValueError('Diagnostic stop did not execute')
        report['complete']=True
        report['paired_steps']=len(paired)
        report['final_checkpoint_sha256']=file_sha(args.output/'cotracker_three_final.pth')
    except BaseException as error:
        report['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        if runtime is not None: os.environ.pop('TRACKRAD_SIDECAR_RUNTIME',None)
        sys.argv=argv; os.chdir(workdir)
        report.update({'rows':probe.rows,'outside_steps':probe.outside_steps,
            'timing_errors':probe.timing_errors,'unassigned_data_seconds':probe.pending_data,'process_wall_seconds':time.perf_counter()-start})
        save_json(args.output/'phase-profile.json',report)
    print('REAL TRAINING PHASE PROFILE COMPLETE '+str(args.output/'phase-profile.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['mlp','mamba'],required=True)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--trainer-sha256',required=True)
    p.add_argument('--runtime-config',type=Path,help='Explicit compact-only execution config; original recipe remains unchanged')
    a=p.parse_args()
    if not 1<=a.steps<=5: p.error('Only 1..5 diagnostic updates are allowed')
    run(a)
