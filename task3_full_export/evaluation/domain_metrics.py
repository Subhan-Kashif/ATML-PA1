
import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate_on_loader(model, loader, device):
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
        "logits": logits,
        "labels": labels,
    }


def aggregate_source_metrics(per_domain_metrics: dict):
    """Returns per-domain + mean + worst (Task 3 needs both, unlike Task 2
    which only needed mean)."""
    accs = [v["accuracy"] for v in per_domain_metrics.values()]
    f1s = [v["macro_f1"] for v in per_domain_metrics.values()]
    result = dict(per_domain_metrics)
    result["mean"] = {"accuracy": sum(accs) / len(accs), "macro_f1": sum(f1s) / len(f1s)}
    result["worst"] = {"accuracy": min(accs), "macro_f1": min(f1s)}
    return result


def sketch_accuracy_delta(method_metrics, erm_metrics):
    return method_metrics["accuracy"] - erm_metrics["accuracy"]


def per_class_accuracy(logits, labels, num_classes=7):
    preds = torch.argmax(logits, dim=1)
    result = {}
    for c in range(num_classes):
        mask = labels == c
        result[c] = (preds[mask] == labels[mask]).float().mean().item() if mask.sum() > 0 else None
    return result


def compare_to_erm(method_per_class, erm_per_class):
    deltas = {
        c: method_per_class[c] - erm_per_class[c]
        for c in method_per_class
        if method_per_class[c] is not None and erm_per_class[c] is not None
    }
    return dict(sorted(deltas.items(), key=lambda kv: kv[1]))


def top_confusions(logits, labels, class_names, k=5):
    from sklearn.metrics import confusion_matrix
    preds = torch.argmax(logits, dim=1).numpy()
    labels_np = labels.numpy()
    cm = confusion_matrix(labels_np, preds, labels=list(range(len(class_names))))
    entries = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                entries.append((class_names[i], class_names[j], int(cm[i, j])))
    entries.sort(key=lambda x: -x[2])
    return entries[:k]
