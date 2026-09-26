
import numpy as np

class MahalanobisScorer:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.means_ = None
        self.var_ = None

    def fit(self, features, labels):
        classes = np.unique(labels)
        means = np.stack([features[labels == c].mean(axis=0) for c in classes])
        centered = np.concatenate(
            [features[labels == c] - means[i] for i, c in enumerate(classes)], axis=0
        )
        var = centered.var(axis=0) + self.eps
        self.means_ = means
        self.var_ = var
        return self

    def score(self, features):
        diffs = features[:, None, :] - self.means_[None, :, :]
        dist = (diffs ** 2 / self.var_[None, None, :]).sum(axis=2)
        return dist.min(axis=1)
