import copy
import os
from pathlib import Path
import tempfile
import unittest
import torch
import analyze as A

LIMITS=dict(output_atol=2e-4,output_rtol=1e-5,grad_atol=2e-6,grad_rtol=1e-3,state_atol=2e-6,state_rtol=1e-5)
KEY='fnet.layer4.0.conv1.weight'


def packet():
    return dict(outputs=[torch.ones(2)],gradients={KEY:torch.tensor([1e-9,2e-9])},classifier_gradients={},
                student={KEY:torch.tensor([.0268,.02]),'fnet.conv1.weight':torch.ones(2)},classifier={})


class Tests(unittest.TestCase):
    def test_all_failures_counted_not_only_first_tensor(self):
        a=packet(); b=copy.deepcopy(a)
        b['student'][KEY][0]+=2.5e-6
        b['student']['fnet.conv1.weight'][0]+=.001
        b['outputs'][0][0]+=.01
        groups,rows=A.compare_all(a,b,LIMITS,1)
        self.assertEqual(groups['student']['mismatched_elements'],2)
        self.assertEqual(groups['student']['mismatched_tensors'],2)
        self.assertEqual(groups['outputs']['mismatched_elements'],1)
        self.assertTrue(groups['gradients']['within_existing_tolerance'])

    def test_stats_native_isclose_and_worst_ratio(self):
        a=torch.tensor([100.,0.,1e-9]); b=torch.tensor([100.0005,3e-6,-1e-9])
        r=A.tensor_stats(a,b,2e-6,1e-5)
        self.assertEqual(r['mismatched_elements'],int((~torch.isclose(a,b,atol=2e-6,rtol=1e-5)).sum()))
        self.assertEqual(r['worst_flat'],0)
        self.assertEqual(r['worst_ratio_flat'],1)
        self.assertEqual(r['sign_flips'],1)

    def test_zero_norm_and_nonfinite(self):
        r=A.tensor_stats(torch.zeros(2),torch.ones(2),2e-6,1e-5)
        self.assertIsNone(r['relative_l2'])
        for x in (float('nan'),float('inf')):
            with self.assertRaises(ValueError): A.tensor_stats(torch.tensor([x]),torch.zeros(1),1e-6,1e-5)
        with self.assertRaises(ValueError): A.tensor_stats(torch.ones(1),torch.ones(2),1e-6,1e-5)

    def test_formula_matches_scalar_cpu_adamw(self):
        p=torch.nn.Parameter(torch.tensor([.0268],dtype=torch.float64))
        opt=torch.optim.AdamW([p],lr=5e-6,weight_decay=1e-5,foreach=False)
        grads=[]
        for g in (1e-10,-3e-9,2e-8):
            previous=p.item(); grads.append(g)
            p.grad=torch.tensor([g],dtype=torch.float64); opt.step()
            predicted=A.adam_estimate(previous,grads,5e-6,1e-5)
            self.assertAlmostEqual(predicted['predicted_weight'],p.item(),places=14)

    def test_trace_detects_increment_not_only_prior_drift(self):
        a=[packet(),packet()]; b=copy.deepcopy(a)
        b[0]['student'][KEY][0]+=1e-7
        b[1]['student'][KEY][0]+=2.5e-6
        trace=A.trace_coordinate(KEY,0,a,b,5e-6,1e-5)
        self.assertTrue(trace['available'])
        self.assertGreater(trace['incremental_update_difference'],2e-6)
        del b[0]['gradients'][KEY]
        self.assertFalse(A.trace_coordinate(KEY,0,a,b,5e-6,1e-5)['available'])

    def fixture(self,root):
        checkout=root/'repo'; checkout.mkdir()
        source=root/'source'; source.mkdir()
        original=source/'original/F0'; candidate=source/'lowmem/F0'
        original.mkdir(parents=True); candidate.mkdir(parents=True)
        A.save(source/'manifest.json',dict(original_sources={},repair_sources={},limits=LIMITS))
        A.save(source/'environment.json',{'fixture':True})
        rows=[dict((k,i if k=='step' else 'paired') for k in A.PAIR_KEYS) for i in range(3)]
        for row in rows: row.update(measured_seconds=10.,peak_allocated=1024,domain='trackrad')
        for i in range(3): torch.save(packet(),original/f'step-{i}.pth')
        A.save(original/'steps.json',rows)
        files=[original/'steps.json']+[original/f'step-{i}.pth' for i in range(3)]
        A.save(original/'verified.json',dict(complete=True,mode='original',arm='F0',manifest=A.sha(source/'manifest.json'),files={p.name:A.sha(p) for p in files}))
        A.save(candidate/'steps.json',rows[:1]); torch.save(packet(),candidate/'step-0.pth')
        fail=packet(); fail['student'][KEY][0]+=2.5e-6
        torch.save(fail,candidate/'failed-evidence.pth')
        A.save(candidate/'equivalence-failure.json',dict(complete=False,arm='F0',mode='lowmem',step=1,runtime_record=rows[1]))
        return source,checkout

    def test_end_to_end_readonly_no_gate(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root=Path(d); source,checkout=self.fixture(root)
            before={p.relative_to(source).as_posix():A.sha(p) for p in source.rglob('*') if p.is_file()}
            r=A.analyze(source,root/'out',checkout)
            self.assertTrue(r['complete']); self.assertFalse(r['training_authorized']); self.assertFalse(r['gate_changed'])
            self.assertTrue(r['frozen_student_exact'])
            self.assertEqual(r['comparisons'][1]['groups']['student']['mismatched_elements'],1)
            self.assertFalse((root/'out/verified.json').exists())
            A.verify(source,before)
            A.analyze(source,root/'out',checkout)
            A.save(source/'environment.json',{'changed':True})
            with self.assertRaises(ValueError): A.analyze(source,root/'out',checkout)

    def test_bad_binding_and_reference_hash_stop(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root=Path(d); source,checkout=self.fixture(root)
            path=source/'lowmem/F0/equivalence-failure.json'; f=A.read(path)
            f['runtime_record']['queries']='changed'; A.save(path,f)
            with self.assertRaises(ValueError): A.analyze(source,root/'out',checkout)
            f['runtime_record']['queries']='paired'; A.save(path,f)
            torch.save(packet(),source/'original/F0/step-0.pth')
            # Serialization can be identical; alter a tensor to guarantee tamper.
            bad=packet(); bad['outputs'][0][0]=3.; torch.save(bad,source/'original/F0/step-0.pth')
            with self.assertRaises(ValueError): A.analyze(source,root/'out',checkout)

    def test_paths_and_schema_refused(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root=Path(d)
            for name in ('../x','C:/x','x\\y'):
                with self.assertRaises(ValueError): A.safe(root,name)
        a=packet(); b=packet(); b['gradients']={}
        with self.assertRaises(ValueError): A.compare_all(a,b,LIMITS,0)


if __name__=='__main__': unittest.main()
