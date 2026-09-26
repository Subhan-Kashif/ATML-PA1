
import numpy as np

def threshold_at_val_percentile(val_scores, percentile=95):
    return np.percentile(val_scores, percentile)

def acceptance_rate(scores, tau):
    return float((scores <= tau).mean())

def fpr_at_threshold(unknown_scores, tau):
    return float((unknown_scores <= tau).mean())
