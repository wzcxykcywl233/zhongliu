import copy
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import numpy as np
import dataset as D
import synthetic as S
import protocol as P
import lld_data as L
import train_domain as T

torch.set_num_threads(2)


class DataTests(unittest.TestCase):
    def test_two_official_patient_namings_same_group(self):
        a=D.identity('images/MR-391135_1_C+A_0000.nii.gz')
        b=D.identity('images/MR391135_6_T2WI_0000.nii.gz')
        self.assertEqual(a['patient'],b['patient'])
        self.assertEqual(a['label'],'labels/MR-391135_1_C+A.nii.gz')
        with self.assertRaises(ValueError): D.identity('images/unknown_0000.nii.gz')

    def test_patient_and_transitive_duplicate_groups_do_not_leak(self):
        rows=[dict(patient=f'p{i}',decoded_image_sha256=str(i)) for i in range(10)]
        rows += [dict(patient='p1',decoded_image_sha256='0'),dict(patient='p2',decoded_image_sha256='1')]
        a,dups=D.split_groups(rows); b,_=D.split_groups(list(reversed(rows)))
        self.assertEqual(a,b); self.assertEqual(a['p0'],a['p1']); self.assertEqual(a['p1'],a['p2'])
        self.assertTrue(dups); self.assertEqual(set(a.values()),{'external-train','external-dev'})

    def test_external_sampling_is_patient_balanced_and_no_dev(self):
        plan=dict(ids={'train-40':[f'A_{i:03d}' for i in range(40)]},external={'patients':{'a':'external-train','b':'external-train','c':'external-dev'}})
        for seed in range(3):
            external=[L.choose(plan,seed,i)[1] for i in range(1,2000,2)]
            self.assertEqual(set(external),{'a','b'}); self.assertEqual(external.count('a'),500)
            for i in range(0,2000,2):
                domain,name,draw=L.choose(plan,seed,i)
                epoch,index=divmod(i//2,40)
                order=np.random.default_rng(P.C.RULE['seed_base']+seed+epoch*97).permutation(40)
                self.assertEqual(name,plan['ids']['train-40'][order[index]])
                self.assertEqual(draw,P.C.RULE['seed_base']+seed*1000003+i*101)
                self.assertEqual(domain,'trackrad')

    def test_forward_inverse_translation_and_out_of_bounds(self):
        image=torch.zeros(3,20,24); image[:,8,10]=255
        mask=(image[:1]>0).float(); q=torch.tensor([[[0.,10.,8.],[0.,23.,8.]]])
        a=torch.eye(3).repeat(2,1,1); a[1,0,2]=3; a[1,1,2]=2
        video,m,labels=S.warp(image,mask,q,a)
        torch.testing.assert_close(video[1,:,10,13],torch.full((3,),255.))
        self.assertEqual(m[1,0,10,13],1)
        torch.testing.assert_close(labels['tracks'][0,1,0],torch.tensor([13.,10.]))
        self.assertEqual(labels['visibility'][0,1,1],0)
        self.assertTrue(torch.equal(labels['tracks'][:,0],q[...,1:]))

    def test_reproducible_nontrivial_synthetic_motion(self):
        a=S.matrices(123,32,48)
        self.assertTrue(torch.equal(a,S.matrices(123,32,48)))
        self.assertFalse(torch.equal(a,S.matrices(124,32,48)))
        self.assertTrue(torch.equal(a[0],torch.eye(3)))
        image=torch.rand(3,32,48)*255; mask=torch.zeros(1,32,48); mask[:,10:20,10:30]=1
        q=torch.tensor([[[0.,10.,12.],[0.,15.,18.]]])
        v,m,l=S.warp(image,mask,q,a)
        self.assertEqual(v.shape,(16,3,32,48)); self.assertEqual(m.shape,(16,1,32,48))
        self.assertTrue(torch.equal(v[0],image)); self.assertTrue(torch.equal(m[0],mask))
        self.assertTrue((v>=0).all() and (v<=255).all()); self.assertTrue(torch.isfinite(l['tracks']).all())
        self.assertGreater((l['tracks'][:,1:]-l['tracks'][:,:1]).abs().max(),0)

    def test_affine_rotation_scale_tracks(self):
        a=torch.eye(3).repeat(2,1,1); a[1,:2,:2]=torch.tensor([[0.,-1.1],[1.1,0.]])
        a[1,:2,2]=torch.tensor([20.,3.])
        q=torch.tensor([[[0.,8.,10.]]]); image=torch.rand(3,32,32)*255
        _,_,l=S.warp(image,torch.ones(1,32,32),q,a)
        torch.testing.assert_close(l['tracks'][0,1,0],torch.tensor([9.,11.8]))

    def test_invalid_warp_fails_closed(self):
        image=torch.ones(3,20,24); mask=image[:1]; q=torch.tensor([[[0.,10.,8.]]]); a=torch.eye(3).repeat(2,1,1)
        for bad in (image*float('nan'),image*256):
            with self.assertRaises(ValueError): S.warp(bad,mask,q,a)
        q[0,0,0]=1
        with self.assertRaises(ValueError): S.warp(image,mask,q,a)

    def test_synthetic_labels_never_call_teacher_and_bind_changes(self):
        label=dict(tracks=torch.ones(1,2,1,2),visibility=torch.ones(1,2,1),confidence=torch.ones(1,2,1))
        s=dict(domain='lld_synthetic',draw=123,sampling={'time_is_synthetic':True},video=torch.ones(2,3,3,3),
               queries=torch.tensor([[[0.,1.,1.]]]),label=label)
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d, patch.object(T,'old_labels') as teacher:
            _,r=T.input_labels(Path(d),Path('.'),{},Path('.'),s,0,1,{})
            self.assertEqual(r['label_origin'],'known_synthetic_affine'); self.assertEqual(r['slot'],-1)
            teacher.assert_not_called()
            T.input_labels(Path(d),Path('.'),{},Path('.'),s,0,1,{})
            s['label']['tracks'][0,1,0,0]+=1
            with self.assertRaises(ValueError): T.input_labels(Path(d),Path('.'),{},Path('.'),s,0,1,{})

    def test_download_resume_and_sha(self):
        payload=b'0123456789'; entry=dict(size=10,sha256=hashlib.sha256(payload).hexdigest(),git_blob=None)
        class Response(io.BytesIO):
            status=206
            headers={'Content-Range':'bytes 4-9/10'}
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root=Path(d); (root/'f.partial').write_bytes(payload[:4])
            def opener(req,timeout):
                self.assertEqual(req.headers['Range'],'bytes=4-'); return Response(payload[4:])
            D.fetch_file(root,'f',entry,opener)
            self.assertEqual((root/'f').read_bytes(),payload)
            (root/'f').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): D.fetch_file(root,'f',entry,opener)

    def test_download_no_range_support_restarts_partial(self):
        payload=b'0123456789'; entry=dict(size=10,sha256=hashlib.sha256(payload).hexdigest())
        class Response(io.BytesIO): status=200; headers={}
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root=Path(d); (root/'f.partial').write_bytes(b'bad')
            D.fetch_file(root,'f',entry,lambda *a,**k:Response(payload))
            self.assertEqual((root/'f').read_bytes(),payload)

    def test_terms_and_path_guards_before_network(self):
        with patch.object(D.urllib.request,'urlopen') as net:
            with self.assertRaises(ValueError): D.download(Path('/does-not-exist'),False)
            net.assert_not_called()
        for name in ('../escape','C:/data','a\\b','/absolute'):
            with self.assertRaises(ValueError): D.safe(Path('input'),name)

    def test_geometry_audit(self):
        # Real voxel/geometry checks without requiring local SimpleITK installation.
        import sys,types
        class Image:
            def __init__(self,array,origin=(0,0,0)): self.array=array; self.origin=origin
            def GetDimension(self): return 3
            def GetSize(self): return self.array.shape[::-1]
            def GetSpacing(self): return (1,1,3)
            def GetOrigin(self): return self.origin
            def GetDirection(self): return (1,0,0,0,1,0,0,0,1)
        a=Image(np.arange(2*16*16).reshape(2,16,16)); b=Image(np.ones((2,16,16)))
        sitk=types.SimpleNamespace(GetArrayFromImage=lambda i:i.array)
        with patch.dict(sys.modules,{'SimpleITK':sitk}):
            self.assertEqual(D.scan_arrays(a,b)['usable_slices'],[0,1])
            with self.assertRaises(ValueError): D.scan_arrays(a,Image(b.array,origin=(1,0,0)))
            with self.assertRaises(ValueError): D.scan_arrays(a,Image(np.zeros((2,16,16))))
            with self.assertRaises(ValueError): D.scan_arrays(a,Image(np.ones((2,16,16))*.5))


if __name__=='__main__': unittest.main()
