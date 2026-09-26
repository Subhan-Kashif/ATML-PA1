
from scipy.special import logsumexp

def energy_score(logits):
    return -logsumexp(logits, axis=1)
