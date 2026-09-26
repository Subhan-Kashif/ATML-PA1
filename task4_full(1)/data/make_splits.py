
import numpy as np
from sklearn.model_selection import train_test_split

def stratified_split(labels, val_fraction=0.1, seed=6304):
    labels = np.array(labels)
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, stratify=labels, random_state=seed
    )
    return train_idx, val_idx
