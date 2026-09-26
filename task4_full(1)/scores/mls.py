
def mls_score(logits):
    return -logits.max(axis=1)
