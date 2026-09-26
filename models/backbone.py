
# Reuses Task 2's backbone architecture verbatim -- same ResNet-18,
# same ImageNet1K_V1 pretrained init, same frozen-BatchNorm policy applies
# via shared.pacs_protocol.freeze_batchnorm at train time (unchanged).
from task2.models.backbone import ResNet18Backbone, build_backbone

__all__ = ["ResNet18Backbone", "build_backbone"]
