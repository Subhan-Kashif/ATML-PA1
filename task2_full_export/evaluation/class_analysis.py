
import torch
from sklearn.metrics import confusion_matrix


def per_class_accuracy(logits, labels, num_classes=7):
    preds = torch.argmax(logits, dim=1)
    result = {}
    for c in range(num_classes):
        mask = labels == c
        result[c] = (preds[mask] == labels[mask]).float().mean().item() if mask.sum() > 0 else None
    return result


def compare_to_source_only(method_per_class, source_only_per_class):
    deltas = {
        c: method_per_class[c] - source_only_per_class[c]
        for c in method_per_class
        if method_per_class[c] is not None and source_only_per_class[c] is not None
    }
    return dict(sorted(deltas.items(), key=lambda kv: kv[1]))


def top_confusions(logits, labels, class_names, k=5):
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
