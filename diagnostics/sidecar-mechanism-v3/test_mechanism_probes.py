import inspect
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import SimpleITK as sitk
import mechanism_worker as worker
from probe_common import (read,save,sha,split_displacement,native_xy,frozen_output,PROFILE,MASK,CASES,input_manifest)

class TestProbes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        worker.activate()

    def test_displacement_is_exact_and_local_zero_mean(self):
        rng=np.random.default_rng(8)
        reference=rng.random((1,6,19,2)).astype(np.float64)
        candidate=reference+rng.random(reference.shape)
        glob,local=split_displacement(reference,candidate)
        np.testing.assert_allclose(glob+local,candidate-reference,atol=1e-14)
        np.testing.assert_allclose(local.mean(2),0,atol=1e-14)
        with self.assertRaises(ValueError): split_displacement(reference,candidate[:,:,:2])
        candidate[0,1,1,0]=np.nan
        with self.assertRaises(ValueError): split_displacement(reference,candidate)

    def test_native_axes_are_not_conflated(self):
        p=np.array([[[[4.,8.]]]])
        zero=np.zeros_like(p)
        direct=native_xy(p,(100,300))-native_xy(zero,(100,300))
        swapped=native_xy(p,(100,300),True)-native_xy(zero,(100,300),True)
        np.testing.assert_allclose(direct,[[[[4*300/512,8*100/384]]]])
        np.testing.assert_allclose(swapped,[[[[8*300/384,4*100/512]]]])

    def test_observation_preserves_output_and_exact_residual(self):
        from cotracker.models.core.cotracker.feature_sidecar import FeatureSidecar
        class Dummy(torch.nn.Module):
            def __init__(self,kind):
                super().__init__()
                self.feature_sidecar=FeatureSidecar(kind,input_dim=4,hidden_dim=6,width=8,depth=1)
                self.flow_head=torch.nn.Linear(6,2)
            def forward(self,h,r):
                residual=self.feature_sidecar.residual(h,r)
                self.last=self.feature_sidecar.correction(residual,self.flow_head.weight)
                return self.flow_head(h)+self.last
        class Model: stride=4
        for kind in ('mlp','mamba'):
            for nonzero in (False,True):
                torch.manual_seed(22)
                model=Model();model.updateformer=Dummy(kind).eval()
                if nonzero: torch.nn.init.normal_(model.updateformer.feature_sidecar.output.weight,std=.03)
                hidden=torch.randn(1,3,5,6);rich=torch.randn(1,3,5,4)
                before=worker.full_tensor_hash(model.updateformer.state_dict())
                with torch.no_grad(): baseline=model.updateformer(hidden,rich)
                rows=[];handles=worker.observe_corrections(model,3,5,rows)
                with torch.no_grad(): observed=model.updateformer(hidden,rich)
                for handle in handles:handle.remove()
                self.assertTrue(torch.equal(baseline,observed))
                self.assertEqual(before,worker.full_tensor_hash(model.updateformer.state_dict()))
                np.testing.assert_array_equal(rows[0],(model.updateformer.last.permute(0,2,1,3)*4).detach().numpy())
                self.assertEqual(bool(np.any(rows[0])),nonzero)

    def test_real_reader_nonsquare_sentinel(self):
        from cotracker.datasets.trackrad_video_dataset import TrackRADVideoDataset
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);training=root/'training';training.mkdir();output=root/'output';output.mkdir()
            reader=Path(inspect.getfile(TrackRADVideoDataset))
            save(training/'frozen-run.json',{'Source':[{'File':'cotracker-algorithm/ext/co-tracker/cotracker/datasets/trackrad_video_dataset.py','SHA256':sha(reader)}]})
            result=worker.inspect_surface(training,root,output,'training')
            self.assertTrue(result['source_verified'])
            self.assertTrue(result['training_transposed'])
            self.assertEqual(result['actual_shape'],[4,7,5])

    def test_routes_are_paired_but_training_axes_differ(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);case='B_006'
            (root/case/'images').mkdir(parents=True);(root/case/'targets').mkdir()
            raw=np.arange(20*30*6,dtype=np.float32).reshape(20,30,6)
            target=np.zeros((20,30,6),np.uint8);target[3:10,5:15,:]=1
            sitk.WriteImage(sitk.GetImageFromArray(raw),str(root/case/'images'/f'{case}_frames.mha'))
            sitk.WriteImage(sitk.GetImageFromArray(target),str(root/case/'targets'/f'{case}_labels.mha'))
            sitk.WriteImage(sitk.GetImageFromArray(target[:,:,:1]),str(root/case/'targets'/f'{case}_first_label.mha'))
            args=(root,case)
            raw_route=worker.prepare_route(*args,'deployment_raw',{'training_transposed':True},5,16)
            norm_route=worker.prepare_route(*args,'deployment_normalized',{'training_transposed':True},5,16)
            train_route=worker.prepare_route(*args,'training_exact',{'training_transposed':True},5,16)
            self.assertTrue(torch.equal(raw_route[1],norm_route[1]))
            self.assertFalse(torch.equal(norm_route[1],train_route[1]))
            self.assertFalse(torch.equal(raw_route[0],norm_route[0]))
            self.assertEqual(raw_route[2].shape,(5,30,20))
            self.assertTrue(train_route[4])

    def test_manifest_cannot_change_or_overwrite_unknown_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);frozen_output(root,{'input':'one'})
            frozen_output(root,{'input':'one'})
            with self.assertRaises(ValueError): frozen_output(root,{'input':'two'})
            sub=root/'unknown';sub.mkdir();save(sub/'other.json',{})
            with self.assertRaises(ValueError): frozen_output(sub,{})

    def test_completed_jobs_are_hash_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);save(root/'manifest.json',{'fixed':True})
            job=root/'job';job.mkdir();save(job/'complete.json',{'complete':True})
            worker.commit_job(job,root/'manifest.json')
            self.assertTrue(worker.verify_job(job,root/'manifest.json'))
            save(job/'complete.json',{'complete':False})
            with self.assertRaises(ValueError):worker.verify_job(job,root/'manifest.json')

    def test_actual_offline_iteration_observer_and_hooks(self):
        from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
        from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar
        torch.manual_seed(4)
        model=CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60)
        attach_sidecar(model,'mlp');model.eval()
        torch.nn.init.normal_(model.updateformer.feature_sidecar.output.weight,std=.001)
        video=torch.rand(1,5,3,32,32)*255
        queries=torch.tensor([[[0.,10.,10.],[0.,15.,10.],[0.,15.,15.]]])
        with torch.no_grad():baseline=model(video,queries,iters=4)
        corrections=[];positions=[]
        handles=worker.observe_corrections(model,3,5,corrections)
        with torch.no_grad():observed=model(video,queries,iters=4,iteration_observer=lambda p,f:positions.extend(p))
        for h in handles:h.remove()
        self.assertEqual(len(corrections),4);self.assertEqual(len(positions),4)
        self.assertEqual(corrections[0].shape,(1,5,3,2))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(baseline[:3],observed[:3])))

    def test_protocol_data_weights_and_cache_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);training=root/'training';study=root/'study';dataset=root/'dataset';weights=root/'weights'
            for path in (training,study,dataset,weights):path.mkdir()
            audit={'complete':True,'steps':1000,'seeds':[0,1,2],'paired_teacher_case_query_video_sequences':True,'final_checkpoints_sha256':{}}
            for seed in (0,1,2):
                for kind in ('mlp','mamba'):
                    rel=f'seed_{seed}/{kind}/cotracker_three_final.pth'
                    p=training/'train'/rel;save(p,{'test_fixture_only':True})
                    audit['final_checkpoints_sha256'][rel]=sha(p)
                    if seed==0:save(p.parent/'meta.json',{})
            save(training/'train/training-audit.json',audit)
            save(training/'pretrained-frozen.pth',{'test_fixture_only':True})
            save(study/'study-spec.json',{});save(study/'local-runtime-images.json',{})
            validation=list(CASES)+['A_019','A_024','B_021','B_024','C_004','C_010']
            rows=[]
            for split,cases in (('train-40',[f'T_{i:03}' for i in range(40)]),('validation-10',validation),('test-38',[f'X_{i:03}' for i in range(38)])):
                for case in cases:
                    if split=='validation-10':
                        p=dataset/case/'frame-rate.json';save(p,{'rate':4})
                        checksum=sha(p)
                    else:checksum='0'*64
                    rows.append({'Split':split,'Case':case,'File':f'C:\\dataset\\{case}\\frame-rate.json','SHA256':checksum})
            source=weights/'scaled_offline.pth';save(source,{'test_fixture_only':True})
            save(training/'frozen-run.json',{'Dataset':rows,'Weights':[{'Name':'scaled_offline.pth','SHA256':sha(source)}]})
            save(study/'frozen-study.json',{'OldProtocolSHA256':sha(training/'frozen-run.json'),'OldTrainingAuditSHA256':sha(training/'train/training-audit.json')})
            for case in CASES:
                folders=[study/'validation-10/pretrained'/PROFILE/'checkpoint/jobs'/case]
                folders += [study/'validation-10'/f'seed_{seed}'/(kind+'_full')/PROFILE/'checkpoint/jobs'/case for seed in (0,1,2) for kind in ('mlp','mamba')]
                for folder in folders:
                    for leaf in (MASK,'diagnostics.json','output/sidecar-tracks.npz','.complete'):save(folder/leaf,{})
            _,hashes=input_manifest(training,study,dataset,weights)
            self.assertIn('weights/scaled_offline.pth',hashes)
            save(dataset/'B_006/frame-rate.json',{'rate':5})
            with self.assertRaises(ValueError):input_manifest(training,study,dataset,weights)

    def test_cached_replay_all_pairs_and_fail_on_changed_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);study=root/'study';dataset=root/'dataset';output=root/'cached'
            study.mkdir();dataset.mkdir();output.mkdir()
            save(root/'manifest.json',{'inputs':{}})
            runtime={'backend':'reference','point_chunk_size':0,'gain':1.,'max_correction_pixels':0.,'checkpoint_blocks':False}
            save(study/'study-spec.json',{'arms':[{'name':kind+'_full','kind':kind,'runtime':runtime} for kind in ('mlp','mamba')]})
            cfg={'visibility_threshold':None,'confidence_threshold':None,'lock_first_mask':True}
            points=np.array([[[[100.,100.],[200.,100.],[200.,200.],[100.,200.]]]*3],dtype=np.float32)
            queries=np.concatenate((np.zeros((1,4,1)),points[:,0]),axis=-1).astype(np.float32)
            v=np.ones((1,3,4),np.float32)
            target=np.zeros((3,40,60),bool);target[:,10:22,10:24]=True
            last=None
            for case in CASES:
                (dataset/case/'targets').mkdir(parents=True)
                for suffix,value in (('labels',target),('first_label',target[:1])):
                    sitk.WriteImage(sitk.GetImageFromArray(value.transpose(2,1,0).astype(np.uint8)),str(dataset/case/'targets'/f'{case}_{suffix}.mha'))
                folders=[study/'validation-10/pretrained'/PROFILE/'checkpoint/jobs'/case]
                folders += [study/'validation-10'/f'seed_{seed}'/(kind+'_full')/PROFILE/'checkpoint/jobs'/case for seed in (0,1,2) for kind in ('mlp','mamba')]
                for index,job in enumerate(folders):
                    (job/'output/images/mri-linac-series-targets').mkdir(parents=True)
                    p=points.copy()
                    if index:p[:,1:,:,0]+=index*.1
                    mask=worker.render_tracks(p,target.shape[1:],target[0],cfg,visibility=v,confidence=v)
                    sitk.WriteImage(sitk.GetImageFromArray(mask.transpose(2,1,0).astype(np.uint8)),str(job/MASK))
                    np.savez_compressed(job/'output/sidecar-tracks.npz',queries=queries,trajectories=p,visibility=v,confidence=v)
                    save(job/'diagnostics.json',{'config':cfg,'model':{'sidecar_runtime':runtime},'prediction':{'output_file_sha256':sha(job/MASK)}})
                    last=job
            worker.cached(study,root,dataset,output,root)
            self.assertTrue(read(output/'cached-complete.json')['all_reference_and_full_replays_exact'])
            changed=np.zeros((60,40,3),np.uint8)
            sitk.WriteImage(sitk.GetImageFromArray(changed),str(last/MASK))
            with self.assertRaises(ValueError):worker.cached(study,root,dataset,output,root)

if __name__=='__main__':unittest.main(verbosity=2)
