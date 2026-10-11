"""ONE configuration per fresh CUDA process. No dataset or persistent training."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).parent))
from profile_sidecar_speed import load_profile_model, benchmark, file_sha
from cotracker.models.core.cotracker.backbone_growth import save_json


def run(args):
    if not torch.cuda.is_available(): raise ValueError('CUDA required')
    if os.environ.get('TRACKRAD_SIDECAR_RUNTIME'): raise ValueError('Unexpected runtime environment override')
    runtime={'backend':args.backend,'point_chunk_size':args.chunk,'gain':1.,'max_correction_pixels':0.,'checkpoint_blocks':args.checkpoint}
    partial={'complete':False,'kind':args.kind,'runtime':runtime,'pid':os.getpid(),'phase':'loading','events':[]}
    partial_path=args.output.with_name('progress.json')
    save_json(partial_path,partial)
    path=args.train/'seed_0'/args.kind/'cotracker_three_final.pth'
    torch.manual_seed(20261011)
    model=load_profile_model(path,args.kind,'cuda',args.loader_sha256)
    branch=model.updateformer.feature_sidecar
    # Verify EACH requested backend/checkpoint setting, not a shared proof
    # copied to every record. Still a small 17-frame proof, explicitly labeled.
    from cotracker.models.core.cotracker.feature_sidecar import configure_sidecar_runtime
    hidden=torch.randn(1,5,17,384,device='cuda')
    rich=torch.randn(1,5,17,branch.input_proj.in_features-384,device='cuda')
    branch.train()
    configure_sidecar_runtime(model)
    expected=branch.residual(hidden,rich)
    target=torch.randn_like(expected)
    (expected-target).square().mean().backward()
    gradients={n:p.grad.detach().clone() for n,p in branch.named_parameters()}
    branch.zero_grad(set_to_none=True)
    configure_sidecar_runtime(model,**runtime)
    actual=branch.residual(hidden,rich)
    (actual-target).square().mean().backward()
    torch.testing.assert_close(expected,actual,rtol=3e-5,atol=3e-6)
    for n,p in branch.named_parameters(): torch.testing.assert_close(gradients[n],p.grad,rtol=2e-4,atol=3e-6)
    proof={'frames':17,'points':5,'output_and_all_parameter_gradients_close':True,
           'output_max_abs':float((expected-actual).abs().max()),
           'gradient_max_abs':max(float((gradients[n]-p.grad).abs().max()) for n,p in branch.named_parameters())}
    branch.zero_grad(set_to_none=True)
    del expected,actual,target,gradients,hidden,rich
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    # Same seeds/features across configurations within each kind. Initial
    # weights were never optimized by the parity check.
    torch.manual_seed(20261012)
    hidden=torch.randn(1,128,60,384,device='cuda')
    rich=torch.randn(1,128,60,branch.input_proj.in_features-384,device='cuda')
    partial['correctness']=proof
    def progress(event):
        partial['phase']=event['phase'];partial['events'].append(event)
        save_json(partial_path,partial)
        print(json.dumps(event),flush=True)
    result=benchmark(model,hidden,rich,runtime,warmup=1,repeats=2,progress=progress)
    result.update({'complete':True,'kind':args.kind,'correctness':proof,'pid':os.getpid(),
        'fresh_process_per_configuration':True,'source_checkpoint_sha256':file_sha(path),
        'torch':torch.__version__,'device':torch.cuda.get_device_name(),
        'checkpoint_loader_sha256':args.loader_sha256.lower(),
        'note':'Synthetic sidecar only; excludes teachers/backbone/I/O. Allocator and driver memory are not Windows shared-memory measurements.'})
    save_json(args.output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['mlp','mamba_short','mamba'],required=True)
    p.add_argument('--backend',choices=['reference','parallel'],required=True)
    p.add_argument('--chunk',type=int,default=0)
    p.add_argument('--checkpoint',action='store_true')
    p.add_argument('--loader-sha256',required=True)
    a=p.parse_args()
    if a.backend=='reference' and (a.chunk or a.checkpoint): p.error('Reference control uses no chunk/checkpoint')
    if a.backend=='parallel' and a.chunk not in (16,32): p.error('Parallel probe uses chunk 16 or 32')
    run(a)
