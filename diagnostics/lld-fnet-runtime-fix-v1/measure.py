"""Synchronized phase timings and allocator/device snapshots; no training changes."""
from contextlib import contextmanager
import time
import torch


def memory():
    free, total = torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(), device_free=free, device_total=total)


class Phases:
    def __init__(self, callback=None):
        self.seconds = {}; self.memory = {}; self.callback = callback

    @contextmanager
    def part(self, name):
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.seconds[name] = time.perf_counter() - start
            self.memory[name] = memory()
            if self.callback:
                self.callback(self)

    def report(self):
        return dict(phase_seconds=self.seconds, phase_memory=self.memory,
                    measured_seconds=sum(self.seconds.values()))
