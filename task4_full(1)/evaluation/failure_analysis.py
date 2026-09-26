
import numpy as np

def find_incorrect_accepts(scores, tau, logits, labels, class_names, min_count=3):
    accepted_idx = np.where(scores <= tau)[0]
    preds = logits.argmax(axis=1)

    records = []
    for idx in accepted_idx:
        records.append({
            "unknown_class": class_names[labels[idx]],
            "predicted_known_class": int(preds[idx]),
            "score": float(scores[idx]),
            "threshold": float(tau),
        })

    if len(records) < min_count:
        print(f"warning: only {len(records)} incorrectly-accepted examples found "
              f"(need at least {min_count})")
    return records
