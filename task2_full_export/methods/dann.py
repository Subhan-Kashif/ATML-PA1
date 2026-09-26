
import torch
import torch.nn as nn
import torch.nn.functional as F
from task2.models.domain_discriminator import DomainDiscriminator, GradientReversalLayer, grl_alpha_schedule


class DANNMethod(nn.Module):
    def __init__(self, backbone, head, discriminator_hidden_dim=256, dropout=0.5,
                 domain_loss_weight=1.0, grl_k=10.0, grl_max_alpha=1.0):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.discriminator = DomainDiscriminator(backbone.feature_dim, discriminator_hidden_dim, dropout)
        self.grl = GradientReversalLayer(alpha=0.0)
        self.domain_loss_weight = domain_loss_weight
        self.grl_k = grl_k
        self.grl_max_alpha = grl_max_alpha

    def forward(self, x):
        feat = self.backbone(x)
        feat = F.normalize(feat, p=2, dim=1)
        return self.head(feat), feat

    def compute_loss(self, source_batch, target_batch, progress_p: float):
        s_images, s_labels, _ = source_batch
        t_images, _, _ = target_batch
        s_logits, s_feat = self.forward(s_images)
        _, t_feat = self.forward(t_images)
        cls_loss = F.cross_entropy(s_logits, s_labels)

        alpha = grl_alpha_schedule(progress_p, k=self.grl_k, max_alpha=self.grl_max_alpha)
        self.grl.alpha = alpha

        domain_feat = torch.cat([s_feat, t_feat], dim=0)
        domain_labels = torch.cat([
            torch.zeros(s_feat.size(0), dtype=torch.long, device=s_feat.device),
            torch.ones(t_feat.size(0), dtype=torch.long, device=t_feat.device),
        ])
        domain_logits = self.discriminator(self.grl(domain_feat))
        domain_loss = F.cross_entropy(domain_logits, domain_labels)

        total = cls_loss + self.domain_loss_weight * domain_loss
        return total, {"cls_loss": cls_loss.item(), "domain_loss": domain_loss.item(), "grl_alpha": alpha}
