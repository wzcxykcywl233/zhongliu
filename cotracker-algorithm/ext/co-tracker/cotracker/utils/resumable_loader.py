"""Align Fabric's implicit sampler epoch with the saved training epoch."""


def set_resumable_loader_epoch(loader, epoch):
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError('invalid training epoch')
    # Lightning/Fabric 1.9.4 overwrites sampler.set_epoch() inside __iter__
    # using _num_iter_calls. Setting only the sampler is therefore insufficient.
    if hasattr(loader, '_num_iter_calls'):
        loader._num_iter_calls = epoch
    sampler = getattr(loader, 'sampler', None)
    if hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)
    dataset = getattr(loader, 'dataset', None)
    if hasattr(dataset, 'set_epoch'):
        dataset.set_epoch(epoch)
