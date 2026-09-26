
import numpy as np

def msp_score(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return 1.0 - probs.max(axis=1)
