from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import RETUNE_CATALOG as C, RETUNE_EXPERIMENTS as E, ExperimentConfig
from parameter_retune import PAIRS

spec=importlib.util.spec_from_file_location('retune_plan',Path(__file__).resolve().parents[1]/'experiments'/'plan_parameter_retune.py')
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


class RetuneTests(unittest.TestCase):
    def test_staged_cli_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def invoke(stage):
                with patch('sys.argv',['planner','--root',str(root),'--stage',stage]): p.main()
            def populate(stage):
                for name in p.read(root/(stage+'.json'))['profiles']:
                    job=root/stage/'validation-10'/name
                    alias='rt_control' if name=='rt_repeat' else name
                    score=.9 if alias=='rt_control' else .901+int(p.hashlib.sha256(alias.encode()).hexdigest()[:4],16)/1e8
                    values={field:(score if key=='DSC' else 1.) for key,field in p.FIELDS.items()}
                    results=[]
                    for i in range(10):
                        case='case'+str(i)
                        directory=job/'checkpoint'/'jobs'/case
                        output=directory/'output'/'images'/'mri-linac-series-targets'/'output.mha'
                        output.parent.mkdir(parents=True)
                        output.write_bytes((alias+case).encode())
                        (directory/'.complete').touch()
                        p.freeze(directory/'diagnostics.json',{'profile':name,'config':C[name]['config'],
                            'prediction':{'array_sha256':alias+case,'output_file_sha256':p.digest(output)}})
                        results.append({'case_id':case,**values})
                    p.freeze(job/'metrics.json',{'aggregates':values,'results':results})
            invoke('single'); invoke('single')
            self.assertEqual(len(p.read(root/'single.json')['profiles']),63)
            with self.assertRaises(FileNotFoundError): invoke('combination')
            populate('single'); invoke('combination'); populate('combination')
            invoke('final'); invoke('final'); populate('final'); invoke('verify-final')
            self.assertTrue((root/'final'/'validation-10'/'retune-matched-comparisons.csv').exists())
            file=next((root/'final'/'validation-10').rglob('output.mha'))
            file.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'Invalid committed'): invoke('verify-final')

    def test_valid_and_single_variable(self):
        for name,entry in C.items():
            self.assertEqual(asdict(E[name]),entry['config'])
            ExperimentConfig(**entry['config'])
            if entry['stage']=='single':
                reference=C[entry['reference']]['config']
                diff={k:v for k,v in entry['config'].items() if v!=reference[k]}
                self.assertEqual(diff,entry['changes'],name)
                self.assertEqual(len(diff),1,name)
        self.assertEqual(E['rt_control'],E['rt_repeat'])
        self.assertTrue(all(v.border_points!=1500 for v in E.values()))

    def test_pairs_have_matched_removal_controls(self):
        for name,e in C.items():
            if e['stage']!='combination': continue
            for field in e['changes']:
                target=dict(e['config']); target[field]=C[e['reference']]['config'][field]
                self.assertTrue(any(v['config']==target and v['stage']!='combination' for v in C.values()),name)

    def rows(self):
        return {name:dict(DSC=.9+i/100000,HD95=4.,MASD=1.5,CD=2.,D98=.9,TimeSec=10.,hashes=[name]*10)
                for i,(name,e) in enumerate(C.items()) if e['stage']!='combination'}

    def test_combinations_bounded_and_no_test_inputs(self):
        selected,axes=p.choose_combinations(self.rows())
        self.assertEqual(len(selected),4*len(PAIRS))
        self.assertTrue(all(C[n]['stage']=='combination' for n in selected))
        self.assertTrue(all(len(v)<=2 for v in axes.values()))

    def test_no_effect_axes_not_promoted(self):
        rows=self.rows()
        for n,e in C.items():
            if n in rows: rows[n]['hashes']=rows[e['reference']]['hashes']
        selected,_=p.choose_combinations(rows)
        self.assertEqual(selected,[])

    def test_shortlist_at_most_three(self):
        self.assertLessEqual(len(p.shortlist(self.rows())),3)
        self.assertNotIn('rt_repeat',p.shortlist(self.rows()))

    def test_freeze_rejects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'plan.json'
            p.freeze(file,{'x':1}); p.freeze(file,{'x':1})
            with self.assertRaises(ValueError): p.freeze(file,{'x':2})

    def test_incomplete_or_wrong_config_stops_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); p.freeze(root/'single.json',{'profiles':['rt_control']})
            path=root/'single'/'validation-10'/'rt_control'
            path.mkdir(parents=True)
            p.freeze(path/'metrics.json',{'results':[], 'aggregates':{}})
            with self.assertRaises(ValueError): p.load_stage(root,'single')

    def test_invalid_tau(self):
        for v in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): ExperimentConfig(query_state_decay_tau=v)


if __name__=='__main__': unittest.main()
