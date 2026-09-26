
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMMethod(nn.Module):
    """Wraps backbone + head. Loss is plain ERM cross-entropy; the SAM
    perturbation logic lives in SAMPerturber below, since it needs to act
    on the full parameter vector between two forward/backward passes."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat), feat

    def compute_loss(self, source_batch):
        images, class_labels, _ = source_batch
        logits, _ = self.forward(images)
        cls_loss = F.cross_entropy(logits, class_labels)
        return cls_loss, {"cls_loss": cls_loss.item()}


class SAMPerturber:
    """Implements the two-step SAM update:
      1. first_step(): compute grad at theta, move to theta + eps
         (normalized ascent step of radius rho), zero grads.
      2. caller re-computes loss/backward AT THE PERTURBED POINT.
      3. second_step(): restore original theta, then apply the gradient
         computed at the perturbed point via optimizer.step().
    Usage per training step (see train.py):
        loss1, logs = method.compute_loss(batch)
        loss1.backward()
        sam.first_step()
        loss2, _ = method.compute_loss(batch)
        loss2.backward()
        sam.second_step(optimizer)
    """
    def __init__(self, params, rho=0.05):
        self.params = list(params)
        self.rho = rho
        self._param_backup = {}

    @torch.no_grad()
    def first_step(self):
        grad_norm = torch.norm(
            torch.stack([p.grad.norm(p=2) for p in self.params if p.grad is not None])
        )
        scale = self.rho / (grad_norm + 1e-12)

        self._param_backup = {}
        for p in self.params:
            if p.grad is None:
                continue
            self._param_backup[p] = p.data.clone()
            e_w = p.grad * scale
            p.add_(e_w)  # move to theta + epsilon

        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def second_step(self, optimizer):
        for p in self.params:
            if p in self._param_backup:
                p.data.copy_(self._param_backup[p])  # restore theta
        optimizer.step()  # apply gradient computed AT theta+epsilon, to original theta
        optimizer.zero_grad()
        self._param_backup = {}
