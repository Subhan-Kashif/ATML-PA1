
import torch
import torch.nn as nn
import torch.nn.functional as F


class ERMMethod(nn.Module):
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


def load_erm_from_task2_checkpoint(backbone, head, checkpoint_path, device):
    """Loads Task 2's Source-only checkpoint into an ERMMethod wrapper.
    Task 2's SourceOnlyMethod and ERMMethod share an identical state_dict
    shape (backbone + head only, no discriminator), so this loads directly."""
    method = ERMMethod(backbone, head).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    method.load_state_dict(state_dict)
    method.eval()
    return method
