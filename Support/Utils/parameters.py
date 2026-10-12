"""Structured floating-point model states, with no shared tensor storage."""
from collections import OrderedDict
import torch


def clone_state(state):
    return OrderedDict((k,v.detach().cpu().clone()) for k,v in state.items())


def validate_state(state, reference=None):
    if not state:
        raise ValueError('Empty state')
    if reference is not None and list(state) != list(reference):
        raise ValueError('State keys or ordering differ')
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'Unsupported or nonfinite tensor: {key}')
        if reference is not None and (value.shape != reference[key].shape or value.dtype != reference[key].dtype):
            raise ValueError(f'Shape or dtype mismatch: {key}')


def extract_update(local, global_state):
    validate_state(global_state)
    validate_state(local, global_state)
    return OrderedDict((k, local[k].detach().cpu() - global_state[k].detach().cpu()) for k in global_state)


def flatten_update(update):
    validate_state(update)
    return torch.cat([v.detach().cpu().reshape(-1).to(torch.float64) for v in update.values()])


def restore_update(vector, template):
    validate_state(template)
    if vector.ndim != 1 or vector.numel() != sum(v.numel() for v in template.values()) or not torch.isfinite(vector).all():
        raise ValueError('Invalid flattened update')
    result, offset = OrderedDict(), 0
    for key, value in template.items():
        length = value.numel()
        result[key] = vector[offset:offset+length].detach().cpu().to(value.dtype).reshape(value.shape).clone()
        offset += length
    return result
