"""GPU training-step microbenchmark. No dataset, teacher cache or optimization run."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'ext' / 'co-tracker'),str(Path(__file__).parent)]
from local_cotracker_source import activate_local_cotracker
activate_local_cotracker()
from audit_backbone_growth import file_sha
from cotracker.models.core.cotracker.feature_sidecar import configure_sidecar_runtime
from cotracker.models.core.cotracker.backbone_growth import save_json
import resources.model as checkpoint_loader


def load_profile_model(path, kind, device, loader_sha256):
    expected_path = (ROOT / 'resources' / 'model.py').resolve()
    actual_path = Path(checkpoint_loader.__file__).resolve()
    if actual_path != expected_path or file_sha(actual_path) != loader_sha256.lower():
        raise ValueError('Stale checkpoint loader: ' + str(actual_path) +
                         '. Build the current code-copy training image in a NEW results root.')
    if os.environ.get('COTRACKER_CHECKPOINT'):
        raise ValueError('Profile must use named source checkpoints, not an environment override')
    model = checkpoint_loader.setup_model(str(path),device=device)
    if getattr(model,'feature_sidecar_kind',None)!=kind or not getattr(model,'frozen_pretrained',False):
        raise ValueError('Wrong loaded source architecture')
    return model


def parity(branch, model):
    torch.manual_seed(20261011)
    h = torch.randn(1,5,17,384,device='cuda')
    x = torch.randn(1,5,17,branch.input_proj.in_features-384,device='cuda')
    configure_sidecar_runtime(model)
    branch.train()
    expected = branch.residual(h,x)
    target = torch.randn_like(expected)
    (expected-target).square().mean().backward()
    gradients = {n:p.grad.detach().clone() for n,p in branch.named_parameters()}
    branch.zero_grad(set_to_none=True)
    configure_sidecar_runtime(model,backend='parallel',point_chunk_size=16,checkpoint_blocks=True)
    actual = branch.residual(h,x)
    (actual-target).square().mean().backward()
    torch.testing.assert_close(expected,actual,rtol=3e-5,atol=3e-6)
    for n,p in branch.named_parameters():
        torch.testing.assert_close(gradients[n],p.grad,rtol=2e-4,atol=3e-6)
    proof = {'output_max_abs':float((expected-actual).abs().max()),
             'gradient_max_abs':max(float((gradients[n]-p.grad).abs().max()) for n,p in branch.named_parameters()),
             'output_and_all_parameter_gradients_close':True}
    branch.zero_grad(set_to_none=True)
    return proof


def benchmark(model, hidden, rich, runtime, warmup=1, repeats=3, progress=None):
    branch = model.updateformer.feature_sidecar
    configure_sidecar_runtime(model,**runtime)
    branch.train()
    optimizer = torch.optim.AdamW(branch.parameters(),lr=5e-5)
    results = []
    for index in range(warmup+repeats):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free_before,total_memory = torch.cuda.mem_get_info()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        if progress:
            progress({'phase':'before_trial','trial':index,'warmup':index<warmup,
                'allocated_mib':allocated_before/1048576,'reserved_mib':reserved_before/1048576,
                'driver_free_mib':free_before/1048576,'driver_total_mib':total_memory/1048576})
        start = time.perf_counter()
        # Match four training iterations in the SAME graph, rather than hiding
        # activation-memory cost behind one forward/backward call.
        predictions = [branch.residual(hidden,rich) for _ in range(4)]
        loss = sum((p-.01).square().mean() for p in predictions)
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_end = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        end = time.perf_counter()
        if index >= warmup:
            results.append({'forward_seconds':forward_end-start,'backward_seconds':backward_end-forward_end,
                            'optimizer_seconds':end-backward_end,'total_seconds':end-start,
                            'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576,
                            'peak_reserved_mib':torch.cuda.max_memory_reserved()/1048576,
                            'allocated_before_mib':allocated_before/1048576,'reserved_before_mib':reserved_before/1048576,
                            'driver_free_before_mib':free_before/1048576,'driver_free_after_mib':torch.cuda.mem_get_info()[0]/1048576,
                            'driver_total_mib':total_memory/1048576})
        del predictions,loss
        if progress:
            progress({'phase':'completed_trial','trial':index,'warmup':index<warmup,
                'forward_seconds':forward_end-start,'backward_seconds':backward_end-forward_end,
                'optimizer_seconds':end-backward_end,'total_seconds':end-start,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576,
                'peak_reserved_mib':torch.cuda.max_memory_reserved()/1048576})
    return {'runtime':runtime,'trials':results,
            'median_step_seconds':statistics.median(r['total_seconds'] for r in results),
            'peak_allocated_mib':max(r['peak_allocated_mib'] for r in results),
            'peak_reserved_mib':max(r['peak_reserved_mib'] for r in results)}


def profile(train, output, minimum_speedup, loader_sha256):
    if not torch.cuda.is_available(): raise ValueError('GPU profile requires CUDA')
    if os.environ.get('COTRACKER_CHECKPOINT'): raise ValueError('Profile must use named source checkpoints, not an environment override')
    torch.manual_seed(20261011)
    records, evidence = [], {}
    selected = None
    for kind in ('mlp','mamba_short','mamba'):
        path = train / 'seed_0' / kind / 'cotracker_three_final.pth'
        print('CHECKPOINT LOADER '+str(checkpoint_loader.__file__),flush=True)
        model = load_profile_model(path,kind,'cuda',loader_sha256)
        branch = model.updateformer.feature_sidecar
        evidence[str(path.relative_to(train))] = file_sha(path)
        initial = {n:v.detach().clone() for n,v in branch.state_dict().items()}
        if kind.startswith('mamba'): correctness = parity(branch,model)
        else: correctness = {'not_applicable':'MLP has no scan'}
        hidden = torch.randn(1,128,60,384,device='cuda')
        rich = torch.randn(1,128,60,branch.input_proj.in_features-384,device='cuda')
        settings = [{'backend':'reference','point_chunk_size':0,'gain':1.,'max_correction_pixels':0.,'checkpoint_blocks':False}]
        if kind.startswith('mamba'):
            settings += [{'backend':'parallel','point_chunk_size':chunk,'gain':1.,'max_correction_pixels':0.,'checkpoint_blocks':checkpoint}
                         for checkpoint in (False,True) for chunk in (16,32)]
        successes = []
        for runtime in settings:
            print('PROFILE '+kind+' '+json.dumps(runtime),flush=True)
            branch.load_state_dict(initial,strict=True)
            try:
                result = benchmark(model,hidden,rich,runtime)
                result.update({'kind':kind,'correctness':correctness,'complete':True})
                successes.append(result)
            except torch.cuda.OutOfMemoryError:
                # Record failure explicitly. Never silently use the slower engine.
                result = {'kind':kind,'runtime':runtime,'complete':False,'error':'CUDA out of memory'}
                branch.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            records.append(result)
            print(json.dumps(result),flush=True)
        if kind == 'mamba':
            reference = next((r for r in successes if r['runtime']['backend']=='reference'),None)
            fast = [r for r in successes if r['runtime']['backend']=='parallel']
            if reference and fast:
                best = min(fast,key=lambda r:r['median_step_seconds'])
                ratio = reference['median_step_seconds']/best['median_step_seconds']
                if ratio >= minimum_speedup:
                    selected = {'runtime':best['runtime'],'measured_speedup':ratio}
        del model,branch,hidden,rich,initial
        gc.collect()
        torch.cuda.empty_cache()
    summary = {'complete':True,'device':torch.cuda.get_device_name(),'torch':torch.__version__,
               'shape':{'frames':60,'points':128,'iterations_in_one_graph':4},'minimum_speedup':minimum_speedup,
               'selected':selected,'records':records,'source_checkpoints_sha256':evidence,
               'checkpoint_loader':{'path':str(checkpoint_loader.__file__),'sha256':loader_sha256.lower()},
               'note':'Synthetic rich-feature TRAINING microbenchmark: forward+backward+AdamW. Excludes teacher, CNN/Transformer, I/O, logging and checkpoint saves. Not an end-to-end speed guarantee or official Mamba kernel benchmark.'}
    save_json(output,summary)
    print('PROFILE COMPLETE '+str(output),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minimum-speedup',type=float,default=2.)
    p.add_argument('--loader-sha256',required=True)
    a = p.parse_args()
    profile(a.train,a.output,a.minimum_speedup,a.loader_sha256)
