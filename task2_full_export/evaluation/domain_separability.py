
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

def compute_domain_separability(source_val_feats, target_feats, seed=6304, test_size=0.3, C=1.0):
    n = min(len(source_val_feats), len(target_feats))
    rng = np.random.RandomState(seed)
    s_idx = rng.choice(len(source_val_feats), n, replace=False)
    t_idx = rng.choice(len(target_feats), n, replace=False)
    X = np.concatenate([source_val_feats[s_idx], target_feats[t_idx]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)
