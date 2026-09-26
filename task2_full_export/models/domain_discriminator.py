
import math
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


def grl_alpha_schedule(p: float, k: float = 10.0, max_alpha: float = 1.0) -> float:
    return max_alpha * (2.0 / (1.0 + math.exp(-k * p)) - 1.0)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


def multilinear_map(feature: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    b = feature.size(0)
    outer = torch.bmm(feature.unsqueeze(2), prob.unsqueeze(1))
    return outer.view(b, -1)
