"""Training-only recomputation; original files, tensor names and math unchanged."""
import ast
import inspect
import textwrap
import types
import torch
from torch.utils.checkpoint import checkpoint

CONFIG = dict(version=1, fnet_checkpoint=True, correlation_checkpoint=True,
              use_reentrant=False, preserve_rng_state=True,
              change_frames=False, change_queries=False, change_precision=False)


def correlation(model, fmaps, coords, support):
    b, t = fmaps.shape[:2]
    n = coords.shape[1]
    r = 2 * model.corr_radius + 1
    corr_feat = model.get_correlation_feat(fmaps, coords)
    track_support = support.view(b, 1, r, r, n, model.latent_dim).squeeze(1).permute(0, 3, 1, 2, 4)
    volume = torch.einsum('btnhwc,bnijc->btnhwij', corr_feat, track_support)
    return model.corr_mlp(volume.reshape(b * t * n, r * r * r * r))


def _runtime_correlation(self, fmaps, coords, support):
    def run(f, c, s):
        return correlation(self, f, c, s)
    return checkpoint(run, fmaps, coords, support, use_reentrant=False,
                      preserve_rng_state=True)


def transformed_forward(original):
    """Replace exactly the four local-correlation statements, nothing else.

    Guard the AST, rather than depending on whitespace / line-ending conventions.
    The imported original sources are also bound by the original study manifest.
    No disk source or class-global forward is patched.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    changed = 0
    class Rewrite(ast.NodeTransformer):
        def visit_For(self, node):
            nonlocal changed
            self.generic_visit(node)
            names = [x.targets[0].id if isinstance(x, ast.Assign) and len(x.targets) == 1
                     and isinstance(x.targets[0], ast.Name) else None for x in node.body]
            if names != ['corr_feat', 'track_feat_support', 'corr_volume', 'corr_emb', None]:
                return node
            expected = ast.parse(textwrap.dedent('''
                corr_feat = self.get_correlation_feat(fmaps_pyramid[i], coords_init / 2**i)
                track_feat_support = track_feat_support_pyramid[i].view(B, 1, r, r, N, self.latent_dim).squeeze(1).permute(0, 3, 1, 2, 4)
                corr_volume = torch.einsum("btnhwc,bnijc->btnhwij", corr_feat, track_feat_support)
                corr_emb = self.corr_mlp(corr_volume.reshape(B * T * N, r * r * r * r))
                corr_embs.append(corr_emb)
            ''')).body
            if [ast.dump(x) for x in node.body] != [ast.dump(x) for x in expected]:
                raise ValueError('Correlation implementation changed; do not guess a patch')
            node.body = ast.parse('corr_emb = self._runtime_correlation(fmaps_pyramid[i], coords_init / 2**i, track_feat_support_pyramid[i])\ncorr_embs.append(corr_emb)').body
            changed += 1
            return node
    Rewrite().visit(tree)
    if changed != 1:
        raise ValueError('Expected exactly one original correlation loop')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, __file__ + ':guarded_forward', 'exec'), namespace)
    return namespace[original.__name__]


def install(model):
    if getattr(model, '_lowmem_installed', False):
        raise ValueError('Runtime already installed')
    from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
    if type(model) is not CoTrackerThreeOffline:
        raise ValueError('Only the audited offline tracker is supported')
    if model.training or model.fnet.training:
        raise ValueError('Frozen normalization/eval mode must be preserved')
    forward = model.forward
    encoder = model.fnet.forward
    patched = transformed_forward(forward.__func__)
    def encode(x):
        if not torch.is_grad_enabled():
            return encoder(x)
        # Recompute the bound forward, NOT Module.__call__: the domain hook must
        # capture once in the real forward, never again during backward.
        return checkpoint(encoder, x, use_reentrant=False, preserve_rng_state=True)
    def dispatch(self, *args, **kwargs):
        is_train = kwargs.get('is_train', args[3] if len(args) > 3 else False)
        if not is_train or not torch.is_grad_enabled():
            return forward(*args, **kwargs)
        return patched(self, *args, **kwargs)
    model.fnet.forward = encode
    model._runtime_correlation = types.MethodType(_runtime_correlation, model)
    model.forward = types.MethodType(dispatch, model)
    model._lowmem_installed = True
    return CONFIG
