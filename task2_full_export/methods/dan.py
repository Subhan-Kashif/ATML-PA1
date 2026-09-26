
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_sq_dists(x):
    sq = (x ** 2).sum(dim=1, keepdim=True)
    dists = sq + sq.t() - 2 * torch.mm(x, x.t())
    return dists.clamp(min=0)


def rbf_kernel_sum(x, y, bandwidth_multipliers=(0.5, 1.0, 2.0)):
    combined = torch.cat([x, y], dim=0)
    dists = _pairwise_sq_dists(combined)
    off_diag = dists[dists > 0]
    median_dist = torch.median(off_diag) if off_diag.numel() > 0 else torch.tensor(1.0, device=x.device)
    kernels = 0.0
    for mult in bandwidth_multipliers:
        bandwidth = mult * median_dist + 1e-8
        kernels = kernels + torch.exp(-dists / bandwidth)
    return kernels


def mmd_loss(source_feats, target_feats, bandwidth_multipliers=(0.5, 1.0, 2.0)):
    ns = source_feats.size(0)
    K = rbf_kernel_sum(source_feats, target_feats, bandwidth_multipliers)
    K_ss, K_tt, K_st = K[:ns, :ns], K[ns:, ns:], K[:ns, ns:]
    return K_ss.mean() + K_tt.mean() - 2 * K_st.mean()


class DANMethod(nn.Module):
    def __init__(self, backbone, head, lambda_mmd=1.0, bandwidth_multipliers=(0.5, 1.0, 2.0)):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.lambda_mmd = lambda_mmd
        self.bandwidth_multipliers = bandwidth_multipliers

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat), feat

    def compute_loss(self, source_batch, target_batch):
        s_images, s_labels, _ = source_batch
        t_images, _, _ = target_batch
        s_logits, s_feat = self.forward(s_images)
        _, t_feat = self.forward(t_images)
        cls_loss = F.cross_entropy(s_logits, s_labels)
        align_loss = mmd_loss(s_feat, t_feat, self.bandwidth_multipliers)
        total = cls_loss + self.lambda_mmd * align_loss
        return total, {"cls_loss": cls_loss.item(), "mmd_loss": align_loss.item()}
