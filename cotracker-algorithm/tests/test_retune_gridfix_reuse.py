import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

DIRECTORY=Path(__file__).resolve().parents[1]/'experiments'
sys.path.insert(0,str(DIRECTORY))
import reuse_retune_gridfix as reuse
from plan_parameter_retune import freeze, digest, CATALOG, FIELDS


class ReuseTests(unittest.TestCase):
    def protocol(self):
        source=[{'File':n,'SHA256':sorted(v)[0]} for n,v in reuse.OLD_HELPERS.items()]
        source.append({'File':'cotracker-algorithm/model.py','SHA256':'unchanged'})
        return {'Manifest':{'profiles':['rt_control','rt_repeat','rt_decay','rt_points_0','rt_grid_1']},
                'CheckpointSHA256':'same-weight',
                'Dataset':[{'Split':'validation-10','Case':'c'+str(i)} for i in range(10)],'Source':source}

    def test_rejects_unrelated_runtime_weights_and_unknown_old_source(self):
        import copy
        old=self.protocol()
        for mutate in (lambda x:x.update(CheckpointSHA256='other'),
                       lambda x:x['Source'][-1].update(SHA256='other')):
            new=copy.deepcopy(old); mutate(new)
            with self.assertRaises(ValueError): reuse.verify_protocol(old,new)
        old['Source'][0]['SHA256']='unknown'
        with self.assertRaises(ValueError): reuse.verify_protocol(old,old)

    def test_copy_resume_provenance_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); old=root/'old'/'single'; new=root/'new'
            old.mkdir(parents=True);new.mkdir()
            protocol=self.protocol()
            for stage in (old,new):
                freeze(stage/'frozen-run.json',protocol)
                (stage/'validation-10').mkdir()
                freeze(stage/'validation-10'/'frozen-images.json',{'image':stage.name})
            def populate(stage,profile):
                directory=stage/'validation-10'/profile
                rows=[]
                for i in range(10):
                    case='c'+str(i); job=directory/'checkpoint'/'jobs'/case
                    output=job/'output'/'images'/'mri-linac-series-targets'/'output.mha'
                    output.parent.mkdir(parents=True);output.write_bytes((profile+case).encode())
                    (job/'.complete').touch();freeze(job/'prediction.json',{})
                    freeze(job/'diagnostics.json',{'profile':profile,'config':CATALOG[profile]['config'],
                        'prediction':{'array_sha256':profile+case,'output_file_sha256':digest(output)}})
                    rows.append({'case_id':case,**{v:1. for v in FIELDS.values()}})
                freeze(directory/'metrics.json',{'results':rows})
            for profile in reuse.CONTROLS:
                populate(old,profile);populate(new,profile)
            populate(old,'rt_points_0')
            original={str(f.relative_to(old)):digest(f) for f in old.rglob('*') if f.is_file()}
            def invoke():
                with patch('sys.argv',['reuse','--old',str(old.parent),'--new',str(new)]): reuse.main()
            invoke();invoke()
            self.assertEqual(original,{str(f.relative_to(old)):digest(f) for f in old.rglob('*') if f.is_file()})
            self.assertTrue((new/'validation-10'/'rt_points_0'/'reuse-origin.json').is_file())
            self.assertFalse((new/'validation-10'/'rt_grid_1').exists())
            control=next((new/'validation-10'/'rt_control').rglob('diagnostics.json'))
            original_control=control.read_bytes()
            changed=json.loads(original_control)
            changed['prediction']['array_sha256']='different-output'
            control.write_text(json.dumps(changed),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'control output changed'): invoke()
            control.write_bytes(original_control)
            corrupt=next((new/'validation-10'/'rt_points_0').rglob('output.mha'))
            corrupt.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Corrupt'): invoke()


if __name__=='__main__': unittest.main()
