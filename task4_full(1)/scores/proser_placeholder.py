
import numpy as np

def proser_placeholder_score(logits, dummy_logits):
    """
    Combine the K known-class logits with the max dummy-classifier logit
    into one (K+1)-way softmax and return the probability mass on the
    placeholder/dummy class as the unknownness score.
    """
    dummy_max = dummy_logits.max(axis=1, keepdims=True)
    combined = np.concatenate([logits, dummy_max], axis=1)
    combined = combined - combined.max(axis=1, keepdims=True)
    exp = np.exp(combined)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, -1]
