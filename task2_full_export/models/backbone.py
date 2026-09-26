
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        self.feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x)


def build_backbone(pretrained=True):
    return ResNet18Backbone(pretrained=pretrained)
