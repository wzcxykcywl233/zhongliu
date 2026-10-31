"""Lossless CPU snapshots, including AdamW moments and RNG; no tolerance repair."""
import hashlib
import json
import random
import numpy as np
import torch


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone(v) for v in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError('Unsupported snapshot type: ' + str(type(value)))


def fingerprint(value):
    h = hashlib.sha256()
    def add(v):
        h.update(type(v).__name__.encode() + b':')
        if isinstance(v, torch.Tensor):
            x = v.detach().cpu().contiguous()
            h.update(str((x.dtype, tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                add(k); add(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(str(len(v)).encode())
            for x in v: add(x)
        else:
            h.update(json.dumps(v, allow_nan=False).encode())
        h.update(b';')
    add(value)
    return h.hexdigest()


def rng_state():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
                torch=torch.get_rng_state().clone(),
                cuda=[v.cpu().clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def restore_rng(rng):
    random.setstate(rng['python'])
    n = rng['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(rng['torch'])
    if len(rng['cuda']) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError('RNG device count changed')
    if rng['cuda']: torch.cuda.set_rng_state_all(rng['cuda'])


def optimizer_names(module, optimizer):
    if optimizer is None: return []
    names = {id(p): n for n, p in module.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def capture(model, opt, head=None, disc=None):
    return dict(schema=1, student=clone(dict(model.state_dict())), optimizer=clone(opt.state_dict()),
                classifier=clone(dict(head.state_dict())) if head is not None else {},
                classifier_optimizer=clone(disc.state_dict()) if disc is not None else {},
                optimizer_names=optimizer_names(model, opt), classifier_names=optimizer_names(head, disc),
                modes={n: m.training for n, m in model.named_modules()},
                head_modes={n: m.training for n, m in head.named_modules()} if head is not None else {},
                trainable=[n for n, p in model.named_parameters() if p.requires_grad], rng=rng_state())


def restore(saved, model, opt, head=None, disc=None):
    if saved['schema'] != 1 or saved['optimizer_names'] != optimizer_names(model, opt):
        raise ValueError('Optimizer parameter order changed')
    if saved['classifier_names'] != optimizer_names(head, disc):
        raise ValueError('Classifier optimizer parameter order changed')
    if saved['trainable'] != [n for n, p in model.named_parameters() if p.requires_grad]:
        raise ValueError('Trainable parameters changed')
    for module, key in ((model, 'modes'), (head, 'head_modes')):
        modes = {n: m.training for n, m in module.named_modules()} if module is not None else {}
        if modes != saved[key]: raise ValueError('Module train/eval modes changed')
    model.load_state_dict(saved['student'], strict=True)
    opt.load_state_dict(clone(saved['optimizer']))
    opt.zero_grad(set_to_none=True)
    if head is not None:
        head.load_state_dict(saved['classifier'], strict=True)
        disc.load_state_dict(clone(saved['classifier_optimizer']))
        disc.zero_grad(set_to_none=True)
    elif saved['classifier'] or saved['classifier_optimizer']:
        raise ValueError('Unexpected classifier snapshot')
    restore_rng(saved['rng'])
    actual = capture(model, opt, head, disc)
    digest = fingerprint(saved)
    if fingerprint(actual) != digest:
        raise ValueError('Prestate restoration is not bitwise exact')
    return digest


def named_optimizer(optimizer, names):
    """CPU moments keyed by parameter names, not allocator-dependent parameter IDs."""
    if optimizer is None: return {}
    sd = optimizer.state_dict()
    result = {}
    for group, group_names in zip(sd['param_groups'], names):
        if len(group['params']) != len(group_names): raise ValueError('Optimizer binding changed')
        for i, name in zip(group['params'], group_names):
            for key, value in sd['state'].get(i, {}).items():
                if not isinstance(value, torch.Tensor): raise ValueError('Unexpected AdamW state')
                result[name + '/' + key] = clone(value)
    return result
