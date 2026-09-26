
import json, os, random
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

SEED = 6304
SPLITS_PATH = "shared/splits/pacs_sketch_seed6304.json"


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_domain_assignment():
    return {"sources": ["photo", "art_painting", "cartoon"], "target": "sketch"}


def make_stratified_split(image_paths, class_labels, seed=SEED, val_frac=0.2):
    tr_p, va_p, tr_l, va_l = train_test_split(
        image_paths, class_labels, test_size=val_frac,
        random_state=seed, stratify=class_labels,
    )
    return tr_p, tr_l, va_p, va_l


def save_splits(splits_dict, path=SPLITS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits_dict, f, indent=2)


def load_splits(path=SPLITS_PATH):
    with open(path) as f:
        return json.load(f)


def freeze_batchnorm(model: nn.Module):
    """Call after model.train(). Keeps BN running stats at pretrained
    values; gamma/beta stay trainable."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    return model
