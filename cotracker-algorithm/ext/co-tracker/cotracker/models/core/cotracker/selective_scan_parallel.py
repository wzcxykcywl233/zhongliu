"""Affine prefix scan with a first-order analytic backward (no CUDA extension).

Same recurrence h[t] = a[t]*h[t-1] + b[t], h[-1] = 0. Parallel association
changes floating-point summation order, NOT the SSM or its state lifetime.
O(T log T) work, O(log T) dependent launches; NOT the official Mamba kernel.
"""
import torch
from torch.autograd.function import once_differentiable


def affine_prefix(a, b):
    if a.shape != b.shape or a.ndim != 4 or a.shape[1] < 1:
        raise ValueError('affine scan expects matching [points,time,channels,state]')
    coefficients, states = a, b
    offset = 1
    while offset < a.shape[1]:
        states = torch.cat((states[:, :offset], states[:, offset:] +
                            coefficients[:, offset:] * states[:, :-offset]), dim=1)
        coefficients = torch.cat((coefficients[:, :offset],
                                  coefficients[:, offset:] * coefficients[:, :-offset]), dim=1)
        offset *= 2
    return states


class AffineScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        states = affine_prefix(a, b)
        ctx.save_for_backward(a, states)
        return states

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_states):
        a, states = ctx.saved_tensors
        # lambda[t] = g[t] + a[t+1]*lambda[t+1], lambda[T] = 0.
        next_a = torch.cat((a[:, 1:], torch.zeros_like(a[:, :1])), dim=1)
        adjoint = affine_prefix(next_a.flip(1), grad_states.flip(1)).flip(1)
        previous = torch.cat((torch.zeros_like(states[:, :1]), states[:, :-1]), dim=1)
        return adjoint * previous if ctx.needs_input_grad[0] else None, adjoint if ctx.needs_input_grad[1] else None


def parallel_scan(a, b):
    return AffineScan.apply(a, b)
