
import torch
from sklearn.metrics import accuracy_score, f1_score


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return accuracy_score(labels.cpu(), preds.cpu())


def compute_macro_f1(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return f1_score(labels.cpu(), preds.cpu(), average="macro")


def evaluate_on_loader(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            feat = model.backbone(imgs)
            logits = model.head(feat)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return {
        "accuracy": compute_accuracy(logits, labels),
        "macro_f1": compute_macro_f1(logits, labels),
        "logits": logits,
        "labels": labels,
    }


def aggregate_source_val_metrics(per_domain_metrics: dict):
    accs = [v["accuracy"] for v in per_domain_metrics.values()]
    f1s = [v["macro_f1"] for v in per_domain_metrics.values()]
    result = dict(per_domain_metrics)
    result["mean"] = {"accuracy": sum(accs) / len(accs), "macro_f1": sum(f1s) / len(f1s)}
    return result


def target_accuracy_delta(method_metrics, source_only_metrics):
    return method_metrics["accuracy"] - source_only_metrics["accuracy"]
