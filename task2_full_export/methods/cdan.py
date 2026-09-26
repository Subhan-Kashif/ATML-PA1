
import torch
import torch.nn as nn
import torch.nn.functional as F
from task2.models.domain_discriminator import (
    DomainDiscriminator, GradientReversalLayer, grl_alpha_schedule, multilinear_map
)


class CDANMethod(nn.Module):
    def __init__(self, backbone, head, num_classes=7, discriminator_hidden_dim=256, dropout=0.5,
                 domain_loss_weight=1.0, grl_k=10.0, grl_max_alpha=1.0):
        super().__init__()
        self.backbone = backbone
        self.head = head
        in_dim = backbone.feature_dim * num_classes
        self.discriminator = DomainDiscriminator(in_dim, discriminator_hidden_dim, dropout)
        self.grl = GradientReversalLayer(alpha=0.0)
        self.domain_loss_weight = domain_loss_weight
        self.grl_k = grl_k
        self.grl_max_alpha = grl_max_alpha

    def forward(self, x):
        feat = self.backbone(x)
        feat = F.normalize(feat, p=2, dim=1)
        logits = self.head(feat)
        prob = F.softmax(logits, dim=1)
        return logits, feat, prob

    def compute_loss(self, source_batch, target_batch, progress_p: float):
        s_images, s_labels, _ = source_batch
        t_images, _, _ = target_batch
        s_logits, s_feat, s_prob = self.forward(s_images)
        _, t_feat, t_prob = self.forward(t_images)
        cls_loss = F.cross_entropy(s_logits, s_labels)

        alpha = grl_alpha_schedule(progress_p, k=self.grl_k, max_alpha=self.grl_max_alpha)
        self.grl.alpha = alpha

        g_s = multilinear_map(s_feat, s_prob)
        g_t = multilinear_map(t_feat, t_prob)
        domain_input = torch.cat([g_s, g_t], dim=0)
        domain_labels = torch.cat([
            torch.zeros(g_s.size(0), dtype=torch.long, device=g_s.device),
            torch.ones(g_t.size(0), dtype=torch.long, device=g_t.device),
        ])
        domain_logits = self.discriminator(self.grl(domain_input))
        domain_loss = F.cross_entropy(domain_logits, domain_labels)

        total = cls_loss + self.domain_loss_weight * domain_loss
        return total, {"cls_loss": cls_loss.item(), "domain_loss": domain_loss.item(), "grl_alpha": alpha}
