
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def extract_features(model, loader, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for imgs, _, _ in loader:
            imgs = imgs.to(device)
            f = model.backbone(imgs)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_source_domain_separability(features_by_domain: dict, seed=6304, test_size=0.3, C=1.0):
    """features_by_domain: {'photo': (N,512) array, 'art_painting': ..., 'cartoon': ...}
    Balances counts across the three domains, trains a multinomial
    logistic regression to predict WHICH source domain a feature came
    from. Held-out accuracy is the separability score; chance = 1/3."""
    domain_names = list(features_by_domain.keys())
    n = min(len(features_by_domain[d]) for d in domain_names)

    rng = np.random.RandomState(seed)
    X_parts, y_parts = [], []
    for i, dom in enumerate(domain_names):
        idx = rng.choice(len(features_by_domain[dom]), n, replace=False)
        X_parts.append(features_by_domain[dom][idx])
        y_parts.append(np.full(n, i))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, multi_class="multinomial")
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)
