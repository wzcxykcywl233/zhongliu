import unittest
import torch
from test_forward_pass import resource_model
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline


class GridMemoryTests(unittest.TestCase):
    def test_real_model_multisegment_grids_and_unchanged_support_features(self):
        previous=torch.get_num_threads(); torch.set_num_threads(1)
        try:
            with torch.inference_mode():
                torch.manual_seed(42)
                model=CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60).eval()
                video=torch.rand(1,5,1,64,64)*255
                queries=torch.tensor([[[0.,16.,16.],[0.,32.,32.]]])
                for grid in (0,3,5,7):
                    diagnostics={}
                    result=resource_model.hierarchical_forward_pass(
                        model,video,queries,span=2,support_grid_size=grid,n_iterations=1,
                        original_feature_weight=.5,dual_anchor_weight=.5,
                        query_memory_mode='topk_confidence_diversity',query_memory_slots=4,
                        diagnostics=diagnostics,device='cpu')
                    self.assertEqual(result.trajectories.shape,(1,5,2,2))
                    self.assertTrue(torch.isfinite(result.trajectories).all())
                    self.assertGreater(diagnostics['query_memory_fusions'],0)
                    first, memory,_=resource_model._run_single_clip(
                        model,video[:,:3],queries,support_grid_size=grid,n_iterations=1,
                        return_query_feature_memory=True,device='cpu')
                    for pyramid in memory:
                        for feature in pyramid: self.assertEqual(feature.shape[-2],2)
                    observed={}
                    def capture(level,name,value): observed[(level,name)]=value.clone()
                    resource_model._run_single_clip(
                        model,video[:,2:],queries,support_grid_size=grid,n_iterations=1,
                        query_feature_memory=memory,query_feature_weight=.5,
                        feature_observer=capture,device='cpu')
                    if grid:
                        for level in range(model.corr_levels):
                            for kind in ('track','support'):
                                torch.testing.assert_close(observed[(level,'current_'+kind)][...,2:,:],
                                    observed[(level,'input_'+kind)][...,2:,:],rtol=0,atol=0)
                    else:
                        plain=model(video[:,:3].expand(-1,-1,3,-1,-1),queries,iters=1,
                                    query_feature_memory=memory,query_feature_weight=.5)
                        explicit=model(video[:,:3].expand(-1,-1,3,-1,-1),queries,iters=1,
                                    query_feature_memory=memory,query_feature_weight=.5,query_feature_memory_points=2)
                        for a,b in zip(plain[:3],explicit[:3]): torch.testing.assert_close(a,b,rtol=0,atol=0)
        finally: torch.set_num_threads(previous)


if __name__=='__main__': unittest.main()
