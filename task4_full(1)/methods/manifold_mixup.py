
import torch
import numpy as np

def manifold_mixup_pair(features, labels, alpha=2.0):
    batch_size = features.size(0)
    perm = torch.randperm(batch_size, device=features.device)

    same_class = labels[perm] == labels
    tries = 0
    while same_class.any() and tries < 10:
        new_perm = torch.randperm(batch_size, device=features.device)
        perm = torch.where(same_class, new_perm, perm)
        same_class = labels[perm] == labels
        tries += 1

    lam = float(np.random.beta(alpha, alpha))
    mixed = lam * features + (1 - lam) * features[perm]
    return mixed, labels, labels[perm], lam
