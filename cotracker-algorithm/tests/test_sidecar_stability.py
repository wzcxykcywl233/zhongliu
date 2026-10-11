import ast
import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'ext/co-tracker')]
from summarize_sidecar_stability import stability,compare_tree
from training_phase_probe import PhaseProbe,InstrumentTrainer,instrument
from cotracker.models.core.cotracker.feature_sidecar import FeatureSidecar,configure_sidecar_runtime
from profile_sidecar_training_phases import training_arguments


def measurements():
    return [dict(step=i,complete=True,optimizer_updates=1,gradients_finite=True,
        wall_seconds_including_data=2.,loss=.02,gradient_norm=.01,residual_rms_max=.001,learning_rate=.00005,
        memory_after=dict(peak_allocated_mib=12000.,peak_reserved_mib=13000.,driver_free_mib=10000.,allocated_mib=700.)) for i in range(100)]


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_order_coverage_finite_memory_and_growth_gates(self):
        rows=measurements();self.assertTrue(stability(rows)['passed'])
        with self.assertRaises(ValueError): stability(rows[:-1])
        for field,value in [('loss',float('nan')),('gradient_norm',float('inf')),('gradients_finite',False)]:
            bad=copy.deepcopy(rows);bad[60][field]=value
            with self.assertRaises(ValueError): stability(bad)
        for field,value in [('peak_allocated_mib',21000),('driver_free_mib',0)]:
            bad=copy.deepcopy(rows);bad[60]['memory_after'][field]=value
            with self.assertRaises(ValueError): stability(bad)
        bad=copy.deepcopy(rows)
        for r in bad[80:]: r['wall_seconds_including_data']=5
        self.assertFalse(stability(bad)['passed'])
        bad=copy.deepcopy(rows)
        for r in bad[80:]: r['memory_after']['allocated_mib']=1100
        self.assertFalse(stability(bad)['passed'])

    def test_resume_state_strict_keys_rng_and_tolerances(self):
        state={'state':{0:{'exp_avg':torch.tensor([.1,.2]),'step':torch.tensor(100.)}},'groups':[{'lr':.00001}]}
        self.assertEqual(compare_tree(state,copy.deepcopy(state)),[])
        close=copy.deepcopy(state);close['state'][0]['exp_avg'][0]+=1e-8
        self.assertTrue(compare_tree(state,close,tolerant=True))
        close['state'][0]['exp_avg'][0]+=.01
        with self.assertRaisesRegex(ValueError,'tolerance'): compare_tree(state,close,tolerant=True)
        with self.assertRaises(ValueError): compare_tree({'rng':torch.tensor([1,2])},{'rng':torch.tensor([1,3])})
        with self.assertRaises(ValueError): compare_tree(state,{})
        with self.assertRaises(ValueError): compare_tree({'lr':.1},{'lr':.2},tolerant=True)

    def test_resume_skip_outside_timed_steps_and_health_anchor(self):
        source=(ROOT/'ext/co-tracker/train_on_real_data.py').read_text()
        _,counts=instrument(source,'trainer.py',resume_safe=True)
        self.assertEqual(counts['health'],1)
        tree=InstrumentTrainer(resume_safe=True).visit(ast.parse(source))
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='(i_batch, batch)')
        self.assertIsInstance(loop.body[0],ast.If)
        self.assertEqual(ast.unparse(loop.body[0].test),'epoch == resume_epoch and i_batch < resume_batch')
        self.assertIsInstance(loop.body[1],ast.With)
        original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='fetch_optimizer')
        transformed=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fetch_optimizer')
        self.assertEqual(ast.dump(original),ast.dump(transformed))
        recipe=training_arguments('mamba',Path('/a'),Path('/b'),Path('/c'))
        self.assertEqual(recipe[recipe.index('--num_steps')+1],'1000')

    def test_health_reads_do_not_change_gradients_or_parameters(self):
        module=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(module.parameters())
        loss=module(torch.ones(1,3)).square().sum();loss.backward()
        gradients=[p.grad.clone() for p in module.parameters()];state=copy.deepcopy(module.state_dict())
        probe=PhaseProbe(100,monitor_health=True)
        with patch.object(probe,'sync'),patch.object(probe,'memory',return_value={}):
            with probe.step(0): probe.health(loss,optimizer);probe.mark_update(1)
        self.assertTrue(probe.rows[0]['gradients_finite'])
        for p,g in zip(module.parameters(),gradients): self.assertTrue(torch.equal(p.grad,g))
        compare_tree(state,module.state_dict())

    def test_resume_skips_do_not_create_ghost_updates(self):
        source='''
def run():
    total_steps=50
    epoch=resume_epoch=1
    resume_batch=10
    for i_batch,batch in enumerate(range(12)):
        if epoch == resume_epoch and i_batch < resume_batch:
            continue
        total_steps += 1
'''
        tree=InstrumentTrainer(resume_safe=True).visit(ast.parse(source))
        probe=PhaseProbe(100)
        namespace={'__probe__':probe}
        exec(compile(ast.fix_missing_locations(tree),'toy.py','exec'),namespace)
        with patch.object(probe,'sync'),patch.object(probe,'memory',return_value={}): namespace['run']()
        self.assertEqual([r['step'] for r in probe.rows],[50,51])
        self.assertTrue(all(r['complete'] and r['optimizer_updates']==1 for r in probe.rows))

    def test_compact_sidecar_adamw_checkpoint_restart(self):
        torch.manual_seed(7)
        base=FeatureSidecar('mamba',input_dim=8,hidden_dim=12,width=8,depth=1)
        configure_sidecar_runtime(SimpleNamespace(updateformer=SimpleNamespace(feature_sidecar=base)),backend='compact')
        a,b=copy.deepcopy(base),copy.deepcopy(base)
        oa,ob=torch.optim.AdamW(a.parameters(),lr=.001),torch.optim.AdamW(b.parameters(),lr=.001)
        h,r=torch.randn(1,2,9,12),torch.randn(1,2,9,8)
        def update(model,opt):
            opt.zero_grad();out=model.residual(h,r);(out-.01).square().mean().backward();opt.step()
        for _ in range(4): update(a,oa)
        for _ in range(2): update(b,ob)
        saved_m,saved_o=copy.deepcopy(b.state_dict()),copy.deepcopy(ob.state_dict())
        b=copy.deepcopy(base);b.load_state_dict(saved_m);ob=torch.optim.AdamW(b.parameters(),lr=.001);ob.load_state_dict(saved_o)
        for _ in range(2): update(b,ob)
        compare_tree(a.state_dict(),b.state_dict());compare_tree(oa.state_dict(),ob.state_dict())


if __name__=='__main__': unittest.main()
