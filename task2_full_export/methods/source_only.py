
import torch.nn as nn
import torch.nn.functional as F


class SourceOnlyMethod(nn.Module):
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
