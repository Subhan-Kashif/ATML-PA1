
import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate_source_domain(model, loader, device):
    """Runs the model (any of ERM/DAN-DG/SAM, since all share the same
    forward() signature) on one source domain's loader. Source-domain-only
    -- never called with a target/Sketch loader anywhere in this file."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            logits, _ = model(imgs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = torch.argmax(logits, dim=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def evaluate_all_sources(model, source_val_loaders, device):
    """source_val_loaders: dict {domain_name: DataLoader}, all three
    SOURCE domains only. Returns per-domain metrics + mean + worst."""
    per_domain = {dom: evaluate_source_domain(model, loader, device)
                  for dom, loader in source_val_loaders.items()}
    accs = [v["accuracy"] for v in per_domain.values()]
    f1s = [v["macro_f1"] for v in per_domain.values()]
    result = dict(per_domain)
    result["mean"] = {"accuracy": sum(accs) / len(accs), "macro_f1": sum(f1s) / len(f1s)}
    result["worst"] = {"accuracy": min(accs), "macro_f1": min(f1s)}
    return result


def checkpoint_selection_score(all_sources_result):
    """The single scalar used for early stopping / best-checkpoint
    selection, per spec: mean macro-F1 across the three source validation
    domains. (Worst-domain is reported for analysis but NOT used to select
    checkpoints, per Task 2's precedent and this task's own instructions.)"""
    return all_sources_result["mean"]["macro_f1"]
