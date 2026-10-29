"""Gradient synchronization shared by Tac-LeWM and Tac-FRS training."""

import torch
import torch.distributed as dist


def average_gradients(parameters):
    """Average dense gradients in one collective on an initialized process group.

    Every rank must provide the same ordered parameters. Missing local gradients
    contribute zeros, so conditional branches cannot change collective ordering.
    """
    parameters = list(parameters)
    gradients = [
        parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
        for parameter in parameters
    ]
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(dist.get_world_size())
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        reduced = flat[offset : offset + size].view_as(parameter)
        if parameter.grad is None:
            parameter.grad = reduced.clone()
        else:
            parameter.grad.copy_(reduced)
        offset += size
