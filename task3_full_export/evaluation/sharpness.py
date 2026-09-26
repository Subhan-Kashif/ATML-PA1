
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset


def build_fixed_sharpness_batch(source_val_datasets: dict, per_domain_n=32, seed=6304):
    """source_val_datasets: {domain_name: PACSDataset} (validation split).
    Selects exactly per_domain_n examples from EACH source domain, seeded,
    and returns a single DataLoader yielding one fixed batch of
    3 * per_domain_n examples. This exact batch is reused for every model."""
    rng = np.random.RandomState(seed)
    subsets = []
    for dom, ds in source_val_datasets.items():
        n = len(ds)
        take = min(per_domain_n, n)
        idx = rng.choice(n, take, replace=False)
        subsets.append(Subset(ds, idx.tolist()))

    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(subsets)
    loader = DataLoader(combined, batch_size=len(combined), shuffle=False)
    return loader


def compute_sharpness(model, fixed_loader, device, rho=0.05):
    """Delta_sharp = L(theta + eps) - L(theta), where eps is a normalized
    gradient-ascent step of radius rho on the FIXED validation batch.
    Model is placed in eval mode (BN stats untouched either way, since
    they're frozen throughout Task 2/3), a single forward/backward pass
    computes the ascent direction, then we perturb, re-evaluate, and
    restore original weights -- no optimizer step, no lasting change."""
    model.eval()
    imgs, labels, _ = next(iter(fixed_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        if p.grad is not None:
            p.grad = None

    logits, _ = model(imgs)
    loss_theta = F.cross_entropy(logits, labels)
    loss_theta.backward()

    grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in params if p.grad is not None]))
    scale = rho / (grad_norm + 1e-12)

    backup = {}
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            backup[p] = p.data.clone()
            p.add_(p.grad * scale)

    for p in params:
        if p.grad is not None:
            p.grad = None

    with torch.no_grad():
        logits_pert, _ = model(imgs)
        loss_perturbed = F.cross_entropy(logits_pert, labels)

    with torch.no_grad():
        for p in params:
            if p in backup:
                p.data.copy_(backup[p])

    delta_sharp = (loss_perturbed - loss_theta).item()
    return delta_sharp
