
import numpy as np
from sklearn.metrics import roc_auc_score

def auroc_known_vs_unknown(known_scores, unknown_scores):
    """known/unknown unknownness scores (larger = more unknown)."""
    y_true = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
    y_score = np.concatenate([known_scores, unknown_scores])
    return roc_auc_score(y_true, y_score)
