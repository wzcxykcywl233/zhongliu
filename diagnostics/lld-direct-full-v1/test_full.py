import copy
import csv
from contextlib import ExitStack, nullcontext
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import full_context as X
import training as T
import full_evaluation as E

C,P,H,V,S=X.C,X.P,X.H,X.V,X.S
torch.set_num_threads(2)


def row(i):
    r={k:'hash-'+str(i) for k in P.PAIR_KEYS}
    r.update(step=i,domain='trackrad' if i%2==0 else 'lld_synthetic',point=.1,domain_loss=.2,
        student_grad_norm=.1,classifier_grad_norm=.1,grl_strength=0.,seconds=1.,
        peak_allocated=100,device_total=1000,max_memory_bytes=100,
        runtime_memory=dict(peak_allocated=100,device_total=1000))
    return r


def probe(n,i):
    return dict(row(i),after_updates=n,sample_index=i,point_loss=.1,coordinate_shift_rms_px=.1,
        visibility_mean_abs_change=.1,confidence_mean_abs_change=.1,
        visibility_mean=.9,confidence_mean=.9,out_of_bounds_fraction=0.)


def stable_fixture(root):
    C.save(root/'plan.json',dict(rule=H.RULE,sources=H.sources(),evidence={'old':'bound'},upstream={'base':'bound'}))
    C.save(root/'environment.json',{'fixture':1})
    C.save(root/'graph/summary.json',dict(complete=True,weights_unchanged=True,optimizer_unchanged=True,
        rows=[dict(domain=d,zero_student_gradient_paths=0,zero_embedding_gradient_is_none=True) for d in ('trackrad','lld_synthetic')]))
    H.publish(root,root/'graph',['summary.json'],'graph')
    arms={}
    for a in H.ARMS:
        f=root/'train'/a; C.save(f/'resume.pth',{'not_imported_as_weights':True})
        C.save(f/'steps.json',[row(i) for i in range(100)])
        C.save(f/'probes.json',[probe(n,i) for n in H.RULE['probe_after_updates'] for i in H.RULE['fixed_probe_sample_indices']])
        C.save(f/'proof.json',dict(arm=a,updates=100,frozen_exact=True,sidecar_updated=True,partial_encoder_updated=True))
        H.publish(root,f,['resume.pth','steps.json','probes.json','proof.json'],a)
        arms[a]=dict(technical_checks_passed=True,frozen_exact=True,updates=100,needs_loss_review=False,fixed_probe_loss_review_alerts=[])
    C.save(root/'summary.json',dict(complete=True,graph_isolation_passed=True,paired_inputs_and_labels=True,arms=arms,
        previous_strict_gate_changed=False,numerical_equivalence_claimed=False,full_training_authorized=False,validation_or_test_accessed=False))


class ProtocolTests(unittest.TestCase):
    def test_five_fixed_full_arms_without_mutating_old_protocol(self):
        self.assertEqual(list(X.ARMS),['M0','F0','Fprobe','F001','F010'])
        self.assertEqual(X.RULE['full_steps'],2000); self.assertEqual(X.RULE['seeds'],[0,1,2])
        self.assertFalse(X.RULE['previous_numerical_gate_changed'])
        self.assertFalse(X.RULE['numerical_equivalence_claimed'])
        self.assertNotIn('direct_full',P.RULE)
        self.assertEqual(X.arms_for(None,'full'),list(X.ARMS))
        with self.assertRaises(ValueError): X.arms_for(None,'pilot')

    def test_evidence_readonly_provenance_and_stability_tamper(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            folder=Path(d); stable=folder/'stable'; evidence=folder/'evidence'; out=folder/'out'
            stable_fixture(stable); C.save(evidence/'environment.json',{'fixture':1})
            before={p.relative_to(stable).as_posix():C.sha(p) for p in stable.rglob('*') if p.is_file()}
            old=dict(previous_files={},ids={'train-40':[],'validation-10':[],'test-38':[]},images={},external={},external_manifest_sha256='x')
            with patch.object(H,'evidence_binding',return_value={'old':'bound'}), patch.object(X.B,'identity',return_value={'base':'bound'}), patch.object(P,'checked',return_value=old), patch.object(C,'checked'):
                X.prepare(out,folder,folder,folder,evidence,stable)
                X.provenance(out,folder,folder,folder,evidence,stable)
                C.verify(stable,before)
                X.gate(out,'full','F010',2)
                with self.assertRaises(ValueError): X.gate(out,'full','S010',0)
                with self.assertRaises(ValueError): X.test_gate(out)
                # Do not silently reinterpret the old fixed false authorization flag.
                summary=C.read(stable/'summary.json'); summary['full_training_authorized']=True; C.save(stable/'summary.json',summary)
                with self.assertRaises(ValueError): X.stability_binding(stable,evidence,folder,folder,folder)

    def test_hidden_loss_alert_not_accepted_from_summary_only(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root=Path(d); stable_fixture(root); f=root/'train/F001'
            values=C.read(f/'probes.json'); values[-1]['point_loss']=.4; C.save(f/'probes.json',values)
            H.publish(root,f,['resume.pth','steps.json','probes.json','proof.json'],'F001')
            with patch.object(H,'evidence_binding',return_value={'old':'bound'}), patch.object(X.B,'identity',return_value={'base':'bound'}):
                with self.assertRaisesRegex(ValueError,'loss alert'): X.stability_binding(root,root,root,root,root)

    def test_validation_requires_gain_against_all_three_controls(self):
        metrics={(a,s):dict(DSC=.8,HD95=2.,MASD=1.,CD=1.,D98=.9) for a in ['pretrained']+list(X.ARMS) for s in [0,1,2]}
        for s in [0,1,2]:
            metrics['F001',s]['DSC']+=.001; metrics['F010',s]['DSC']+=.002
            metrics['F010',s]['D98']-=.01
        self.assertEqual(E.selection(metrics,[0,1,2],list(X.ARMS))['promoted'],['F001'])
        for s in [0,1,2]: metrics['Fprobe',s]['DSC']=.803
        self.assertEqual(E.selection(metrics,[0,1,2],list(X.ARMS))['promoted'],[])

    def test_runtime_nonfinite_and_memory_guards(self):
        T.validate([row(0),row(1)],2000)
        for k,v in [('seconds',181.),('seconds',float('nan')),('point',float('nan'))]:
            r=row(0); r[k]=v
            with self.assertRaises(ValueError): T.validate([r],2000)
        r=row(0); r['runtime_memory']['peak_allocated']=900
        with self.assertRaises(ValueError): T.validate([r],2000)


def toy(seed):
    torch.manual_seed(35+seed)
    m=torch.nn.Module(); m.latent_dim=2
    m.fnet=torch.nn.Module(); m.fnet.layer4=torch.nn.Linear(2,2)
    m.updateformer=torch.nn.Module(); m.updateformer.feature_sidecar=torch.nn.Linear(2,2)
    m.register_buffer('frozen_buffer',torch.ones(1)); return m


class TrainingTests(unittest.TestCase):
    def test_model_configuration_partial_frozen_lr_and_seed(self):
        with patch.object(X.T,'new_model',side_effect=lambda w,s,a:(toy(s),None)), patch.object(torch.nn.Module,'cuda',lambda m:m), patch.object(V,'install') as lowmem:
            models={}
            for a in X.ARMS:
                m,opt,head,disc,tap,params=T.make_model(None,a,2)
                models[a]=X.T.S.state_hashes(m)
                self.assertEqual(m.fnet.layer4.weight.requires_grad,a.startswith('F'))
                self.assertEqual([g['lr'] for g in opt.param_groups],[5e-5,5e-6] if a.startswith('F') else [5e-5])
                self.assertEqual(head is None,a in ('M0','F0'))
                if head is not None: self.assertEqual(disc.param_groups[0]['lr'],1e-4)
                if tap: tap.close()
            self.assertEqual(lowmem.call_count,4)
            self.assertTrue(all(v==models['M0'] for v in models.values()))

    def test_train_resume_export_audit_reports_and_fixed_test_roster(self):
        class Phases:
            def __init__(self,callback): callback(self)
            def report(self): return dict(phase_seconds={},phase_memory={},measured_seconds=.001)
            def part(self,name): return nullcontext()
        def make(w,a,s):
            m=toy(s); V.configure(m,a); params=[p for p in m.parameters() if p.requires_grad]
            return m,torch.optim.AdamW(params,lr=5e-5),None,None,None,params
        def old_training(previous,seed):
            initial=X.T.S.state_hashes(toy(seed))
            return None,[row(i) for i in range(4)],dict(initial_sidecar={n:h for n,h in initial.items() if n.startswith(P.PREFIX)})
        def labels(root,previous,old,w,s,seed,step,ref):
            r={k:row(step)[k] for k in P.PAIR_KEYS}; C.freeze(root/f'inputs/seed_{seed}/{step:05d}.json',r)
            return {},r
        calls=[]
        def execute(m,s,label,step,arm,params,classifier,opt,disc,head,tap,phases,audit,evidence):
            calls.append((arm,step)); opt.zero_grad(set_to_none=True)
            m.updateformer.feature_sidecar(m.fnet.layer4(torch.ones(1,2))).square().sum().backward(); opt.step()
            return {k:row(step)[k] for k in ('point','domain_loss','student_grad_norm','classifier_grad_norm','grl_strength')},None
        def reload(final,after,*args):
            saved=H.load(final)
            self.assertEqual(set(saved['model']),set(after)); self.assertNotIn('domain_head',saved)
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d, ExitStack() as stack:
            root=Path(d)/'out'; previous=Path(d)/'previous'; audit=Path(d)/'audit'
            C.save(audit/'data-manifest.json',{}); C.save(previous/'manifest.json',dict(weights={},files={'train-40':{}}))
            plan=dict(external_manifest_sha256=C.sha(audit/'data-manifest.json'),ids={'validation-10':['A_001'],'test-38':['X_001']})
            C.save(root/'plan.json',plan); C.save(root/'environment.json',{})
            replacements=[(X,'RULE',dict(X.RULE,full_steps=4,checkpoint_every=1)),(X,'checked',lambda *a:plan),(X,'gate',lambda *a:None),
                (T,'make_model',make),(P,'old_training',old_training),(V,'sample',lambda *a:{}),
                (B:=X.B.BASE,'input_labels',labels),(V,'execute',execute),(V,'Phases',Phases),
                (V,'memory',lambda:dict(peak_allocated=100,device_total=1000)),(T,'reload_check',reload)]
            for obj,n,value in replacements: stack.enter_context(patch.object(obj,n,value))
            for n in ('synchronize','reset_peak_memory_stats','empty_cache'): stack.enter_context(patch.object(torch.cuda,n,lambda:None))
            stack.enter_context(patch.object(torch.cuda,'max_memory_allocated',lambda:100))
            def interrupted(*args,**kwargs):
                if args[3]==2: raise RuntimeError('simulated interruption')
                return execute(*args,**kwargs)
            with patch.object(V,'execute',interrupted):
                with self.assertRaises(RuntimeError): T.train(root,previous,root,audit,root,root,'M0',0)
            calls.clear()
            for seed in [0,1,2]:
                for a in X.ARMS: T.train(root,previous,root,audit,root,root,a,seed)
            self.assertEqual(calls[0],('M0',2))
            T.audit_training(root,previous)
            with patch.object(V,'execute',side_effect=AssertionError('cache not reused')):
                T.train(root,previous,root,audit,root,root,'M0',0)
            def endpoints(split):
                case=plan['ids'][split][0]
                for a in ['pretrained']+list(X.ARMS):
                    for seed in ([0] if a=='pretrained' else [0,1,2]):
                        f=root/'full'/split/f'seed_{seed}'/a
                        vals=dict(DSC=.8,HD95=2.,MASD=1.,CD=1.,D98=.9)
                        vals['DSC']+=dict(pretrained=.01,M0=0,F0=.001,Fprobe=.001,F001=.002,F010=.003)[a]
                        if a=='F010': vals['D98']-=.01
                        if split=='test-38' and a=='F010': vals['DSC']=.99
                        ag={C.FIELDS[k]:v for k,v in vals.items()}; ag['total_time']=1.
                        C.save(f/'metrics.json',dict(aggregates=ag,results=[dict(case_id=case,**{C.FIELDS[k]:v for k,v in vals.items()})]))
                        P.receipt(root,f/'inference-verified.json',[],ids=[case])
                        P.receipt(root,f/'metrics-verified.json',[f/'metrics.json'])
            stack.enter_context(patch.object(E,'checked',lambda *a:plan))
            endpoints('validation-10'); E.report(root,previous,'full','validation-10')
            selection=root/'reports/full/validation-10/selection.json'; frozen=C.sha(selection)
            self.assertEqual(C.read(selection)['promoted'],['F001'])
            self.assertEqual(X.test_gate(root),list(X.ARMS))
            with (root/'reports/full/validation-10/paired-deltas.csv').open(encoding='utf-8-sig') as stream:
                pairs=list(csv.DictReader(stream))
            self.assertEqual(len([r for r in pairs if r['Candidate']=='F010' and r['Reference']=='Fprobe']),3)
            endpoints('test-38'); E.report(root,previous,'full','test-38')
            self.assertEqual(C.sha(selection),frozen)
            self.assertFalse((root/'reports/full/test-38/selection.json').exists())
            p=root/'full/train/seed_0/F001/steps.json'; data=C.read(p); data[0]['queries']='changed'; C.save(p,data)
            with self.assertRaises(ValueError): T.audit_training(root,previous)


if __name__=='__main__': unittest.main()
