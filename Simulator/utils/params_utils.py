import torch
import numpy as np

def set_trainable(trainable_set, all_params):
    """
    Freeze/unfreeze parameters: only the parameters in `trainable_set` 
    will have gradients enabled, others will be frozen.
    """
    trainable_ids = {id(p) for p in trainable_set}
    for p in all_params:
        p.requires_grad_(id(p) in trainable_ids)


def snapshot_params(params):
    """
    Take a snapshot of the current parameter values (detached from the computation graph).
    """
    with torch.no_grad():
        return [p.detach().clone() for p in params]


def load_snapshot(params, snapshot):
    """
    Load a previously saved parameter snapshot back into the model (in-place).
    """
    with torch.no_grad():
        for p, s in zip(params, snapshot):
            p.copy_(s)

