import pandas as pd

import numpy as np
import torch
import random
random.seed(0)

import torch.nn as nn
import torch.nn.functional as F

class SmoothClampedReLU(nn.Module):
    def __init__(self, beta=50):
        super(SmoothClampedReLU, self).__init__()
        self.beta = beta
        
    def forward(self, x):
        # Smooth transition at x=0 (using softplus with high beta)
        activated = F.softplus(x, beta=self.beta)
        # Smooth transition at x=1 (using sigmoid scaled and shifted)
        # As x approaches infinity, this approaches 1
        clamped = activated - F.softplus(activated - 1, beta=self.beta)
        
        return clamped

def preds_to_proba(preds, eps=1e-3, proba_beta=50):
    if preds.shape[1] == 1:
        smooth_clamped = SmoothClampedReLU(beta=proba_beta)
        preds = smooth_clamped(preds)
    else:
        min_preds = preds.min(dim=1, keepdim=True).values
        max_preds = preds.max(dim=1, keepdim=True).values 
        preds = (preds - min_preds) / (max_preds - min_preds) # normalize predictions to [0, 1]
        preds = torch.clamp(preds, eps, 1-eps) # clamp predictions to [eps, 1-eps]
        preds /= preds.sum(dim=1, keepdim=True) # normalize predictions to sum to 1
    return preds
    
def split_indices(N, frac=0.2, max_val_count=1024, random_split=True):
    n_train = N - min(int(frac*N), max_val_count)
    n_train = n_train + n_train%2 # ensure even train samples
    
    if random_split:
        indices = list(range(N))
        random.shuffle(indices)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
    else:
        train_indices = range(n_train)
        val_indices = range(n_train, N)
    return train_indices, val_indices
        
def split_train_states(inputs, train_indices, val_indices):
    train_inputs, val_inputs = {}, {}
    for layer_idx, layer_states in inputs.items():
        train_inputs[layer_idx] = layer_states[train_indices]
        val_inputs[layer_idx] = layer_states[val_indices]
    return train_inputs, val_inputs


def make_json_serializable(obj):
    """
    Recursively convert numpy types and other non-JSON serializable objects to Python native types.
    """
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    else:
        return obj