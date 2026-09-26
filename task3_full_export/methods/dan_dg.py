
import itertools
import torch.nn as nn
import torch.nn.functional as F

from task2.methods.dan import mmd_loss  # exact same MMD/kernel mechanism as Task 2


class DANDGMethod(nn.Module):
    def __init__(self, backbone, head, lambda_dg=1.0, bandwidth_multipliers=(0.5, 1.0, 2.0)):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.lambda_dg = lambda_dg
        self.bandwidth_multipliers = bandwidth_multipliers

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat), feat

    def compute_loss(self, domain_batches):
        """domain_batches: dict {domain_name: (images, class_labels, domain_labels)}
        for the three source domains, one batch each (domain-balanced)."""
        domain_names = list(domain_batches.keys())

        all_logits, all_labels, feats_by_domain = [], [], {}
        for dom in domain_names:
            images, class_labels, _ = domain_batches[dom]
            logits, feat = self.forward(images)
            all_logits.append(logits)
            all_labels.append(class_labels)
            feats_by_domain[dom] = feat

        cls_loss = F.cross_entropy(
            __import__("torch").cat(all_logits), __import__("torch").cat(all_labels)
        )

        pair_losses = []
        for d1, d2 in itertools.combinations(domain_names, 2):
            pair_losses.append(mmd_loss(feats_by_domain[d1], feats_by_domain[d2], self.bandwidth_multipliers))
        mmd_avg = sum(pair_losses) / len(pair_losses)

        total = cls_loss + self.lambda_dg * mmd_avg
        return total, {"cls_loss": cls_loss.item(), "mmd_dg_loss": mmd_avg.item()}
