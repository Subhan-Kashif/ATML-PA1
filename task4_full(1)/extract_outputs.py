
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import yaml

from data.cifar10 import get_cifar10_loaders
from data.cifar100_unknowns import get_cifar100_unknown_loaders
from models.resnet_cifar import ResNetCIFAR


@torch.no_grad()
def extract(model, loader, device, has_dummy=False):
    feats, logits_all, dummy_all, labels_all = [], [], [], []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        if has_dummy:
            feat, logits, dummy_logits = model(x)
            dummy_all.append(dummy_logits.cpu().numpy())
        else:
            feat, logits = model(x)
        feats.append(feat.cpu().numpy())
        logits_all.append(logits.cpu().numpy())
        labels_all.append(y.numpy())

    out = {
        "features": np.concatenate(feats, axis=0),
        "logits": np.concatenate(logits_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
    }
    if has_dummy:
        out["dummy_logits"] = np.concatenate(dummy_all, axis=0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_prefix", required=True, help="e.g. task4/cache/vanilla")
    parser.add_argument("--has_dummy", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_dummy = cfg.get("num_dummy", 0) if args.has_dummy else 0
    model = ResNetCIFAR(num_classes=10, num_dummy=num_dummy).to(device)
    ckpt = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model_state"])

    train_loader, val_loader, test_loader, train_eval_loader = get_cifar10_loaders(
        cfg["data_dir"], batch_size=cfg["batch_size"], use_randaugment=False,
        seed=cfg["seed"], val_fraction=cfg["val_fraction"], num_workers=cfg["num_workers"],
    )
    near_loader, far_loader, _ = get_cifar100_unknown_loaders(
        cfg["data_dir"], batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
    )

    splits = {"train": train_eval_loader, "val": val_loader, "test": test_loader,
              "near": near_loader, "far": far_loader}

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    for name, loader in splits.items():
        out = extract(model, loader, device, has_dummy=args.has_dummy)
        path = f"{args.out_prefix}_{name}.npz"
        np.savez(path, **out)
        print(f"saved {path}  (n={out['features'].shape[0]})")


if __name__ == "__main__":
    main()
