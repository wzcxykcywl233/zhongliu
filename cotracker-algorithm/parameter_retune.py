"""Pre-registered, bounded inference parameter search. No label access."""
from dataclasses import asdict, replace
from itertools import product

AXES = {
    'span': ('hierarchical_span', [3, 5, 8, 10, 15, 20, 30]),
    'iters': ('n_iterations', [1, 2, 3, 4, 6]),
    'slots': ('query_memory_slots', [2, 3, 4, 6, 8]),
    'floor': ('query_memory_original_floor', [0., .15, .30, .50, .70, 1.]),
    'diversity': ('query_memory_diversity_weight', [0., .10, .25, .50, 1.]),
    'feature': ('original_feature_weight', [.10, .25, .50, .75, 1.]),
    'reliability': ('query_memory_min_reliability', [0., .50, .90, .97, .99]),
    'similarity': ('query_memory_min_similarity', [-1., .30, .50, .70, .90]),
    'global': ('dual_anchor_weight', [0., .25, .50, .75, 1.]),
    'tau': ('query_state_decay_tau', [.25, .50, 1., 2., 4.]),
    'points': ('border_points', [250, 500, 750, 1000, 1250]),
    'grid': ('support_grid_size', [0, 3, 5, 7]),
    'occ_visibility': ('occlusion_visibility_threshold', [.3, .5, .7]),
    'occ_fraction': ('occlusion_point_fraction', [.25, .5, .75]),
}
PAIRS = [('span','iters'), ('slots','diversity'), ('feature','floor'),
         ('reliability','similarity'), ('global','tau')]


def build_retune_catalog(base):
    profiles = {'rt_control': base, 'rt_repeat': base,
                'rt_decay': replace(base, query_state_inheritance='vc_decay')}
    catalog = {name: {'stage':'control', 'reference':'rt_control', 'changes':{}}
               for name in profiles}
    catalog['rt_decay']['changes'] = {'query_state_inheritance':'vc_decay'}
    for axis, (field, values) in AXES.items():
        reference = 'rt_decay' if axis == 'tau' else 'rt_control'
        for i, value in enumerate(values):
            if value == getattr(profiles[reference], field):
                continue
            name = f'rt_{axis}_{i}'
            profiles[name] = replace(profiles[reference], **{field:value})
            catalog[name] = {'stage':'single','axis':axis,'value':value,
                             'reference':reference,'changes':{field:value}}
    for name, changes in {
        'rt_no_memory': {'query_memory_mode':'none','query_memory_slots':0},
        'rt_no_merge': {'occlusion_merge':False},
    }.items():
        profiles[name] = replace(base, **changes)
        catalog[name] = {'stage':'mechanism','reference':'rt_control','changes':changes}
    # Matched controls for global-weight x decay interactions.
    for i, value in enumerate(AXES['global'][1]):
        if value == base.dual_anchor_weight: continue
        name = f'rt_dglobal_{i}'
        profiles[name] = replace(profiles['rt_decay'], dual_anchor_weight=value)
        catalog[name] = {'stage':'bridge','reference':'rt_decay','changes':{'dual_anchor_weight':value}}
    for left, right in PAIRS:
        for a, b in product(AXES[left][1], AXES[right][1]):
            reference = 'rt_decay' if right == 'tau' else 'rt_control'
            fields = {AXES[left][0]:a, AXES[right][0]:b}
            if any(value == getattr(profiles[reference], field) for field,value in fields.items()):
                continue  # this is already a single-axis experiment
            name = f'rc_{left}_{AXES[left][1].index(a)}_{right}_{AXES[right][1].index(b)}'
            profiles[name] = replace(profiles[reference], **fields)
            catalog[name] = {'stage':'combination','axes':[left,right],
                             'values':[a,b],'reference':reference,'changes':fields}
    for name in profiles:
        catalog[name]['config'] = asdict(profiles[name])
    return profiles, catalog
