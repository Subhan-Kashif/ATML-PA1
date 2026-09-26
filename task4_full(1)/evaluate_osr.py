
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scores.msp import msp_score
from scores.mls import mls_score
from scores.energy import energy_score
from scores.mahalanobis import MahalanobisScorer
from scores.proser_placeholder import proser_placeholder_score
from evaluation.metrics import auroc_known_vs_unknown
from evaluation.thresholds import threshold_at_val_percentile, acceptance_rate, fpr_at_threshold
from evaluation.failure_analysis import find_incorrect_accepts

CACHE = "task4/cache"


def load(prefix, split):
    return dict(np.load(f"{CACHE}/{prefix}_{split}.npz"))


def csa(logits, labels):
    return float((logits.argmax(axis=1) == labels).mean())


def run_score_comparison_table(prefix="vanilla"):
    train_d, val_d, test_d = load(prefix, "train"), load(prefix, "val"), load(prefix, "test")
    near_d, far_d = load(prefix, "near"), load(prefix, "far")

    maha = MahalanobisScorer().fit(train_d["features"], train_d["labels"])
    score_fns = {
        "MSP": lambda d: msp_score(d["logits"]),
        "MLS": lambda d: mls_score(d["logits"]),
        "Energy": lambda d: energy_score(d["logits"]),
        "Mahalanobis": lambda d: maha.score(d["features"]),
    }

    rows = []
    for name, fn in score_fns.items():
        val_s, test_s = fn(val_d), fn(test_d)
        near_s, far_s = fn(near_d), fn(far_d)
        all_s = np.concatenate([near_s, far_s])
        tau = threshold_at_val_percentile(val_s, 95)
        rows.append({
            "score": name,
            "AUROC_near": auroc_known_vs_unknown(test_s, near_s),
            "AUROC_far": auroc_known_vs_unknown(test_s, far_s),
            "AUROC_all": auroc_known_vs_unknown(test_s, all_s),
            "test_acceptance_rate": acceptance_rate(test_s, tau),
            "near_FPR@95TPR": fpr_at_threshold(near_s, tau),
            "far_FPR@95TPR": fpr_at_threshold(far_s, tau),
            "all_FPR@95TPR": fpr_at_threshold(all_s, tau),
        })
    return pd.DataFrame(rows)


def run_model_comparison_table():
    rows = []
    for prefix in ["vanilla", "gcsc"]:
        val_d, test_d = load(prefix, "val"), load(prefix, "test")
        near_d, far_d = load(prefix, "near"), load(prefix, "far")
        val_s, test_s = mls_score(val_d["logits"]), mls_score(test_d["logits"])
        near_s, far_s = mls_score(near_d["logits"]), mls_score(far_d["logits"])
        all_s = np.concatenate([near_s, far_s])
        tau = threshold_at_val_percentile(val_s, 95)
        rows.append({
            "model": prefix, "score": "MLS", "CSA": csa(test_d["logits"], test_d["labels"]),
            "AUROC_near": auroc_known_vs_unknown(test_s, near_s),
            "AUROC_far": auroc_known_vs_unknown(test_s, far_s),
            "AUROC_all": auroc_known_vs_unknown(test_s, all_s),
            "acceptance_rate": acceptance_rate(test_s, tau),
            "near_FPR@95TPR": fpr_at_threshold(near_s, tau),
            "far_FPR@95TPR": fpr_at_threshold(far_s, tau),
        })

    val_d, test_d = load("proser", "val"), load("proser", "test")
    near_d, far_d = load("proser", "near"), load("proser", "far")

    val_s, test_s = mls_score(val_d["logits"]), mls_score(test_d["logits"])
    near_s, far_s = mls_score(near_d["logits"]), mls_score(far_d["logits"])
    all_s = np.concatenate([near_s, far_s])
    tau = threshold_at_val_percentile(val_s, 95)
    rows.append({
        "model": "proser", "score": "MLS", "CSA": csa(test_d["logits"], test_d["labels"]),
        "AUROC_near": auroc_known_vs_unknown(test_s, near_s),
        "AUROC_far": auroc_known_vs_unknown(test_s, far_s),
        "AUROC_all": auroc_known_vs_unknown(test_s, all_s),
        "acceptance_rate": acceptance_rate(test_s, tau),
        "near_FPR@95TPR": fpr_at_threshold(near_s, tau),
        "far_FPR@95TPR": fpr_at_threshold(far_s, tau),
    })

    val_ph = proser_placeholder_score(val_d["logits"], val_d["dummy_logits"])
    test_ph = proser_placeholder_score(test_d["logits"], test_d["dummy_logits"])
    near_ph = proser_placeholder_score(near_d["logits"], near_d["dummy_logits"])
    far_ph = proser_placeholder_score(far_d["logits"], far_d["dummy_logits"])
    all_ph = np.concatenate([near_ph, far_ph])
    tau_ph = threshold_at_val_percentile(val_ph, 95)
    rows.append({
        "model": "proser", "score": "placeholder", "CSA": csa(test_d["logits"], test_d["labels"]),
        "AUROC_near": auroc_known_vs_unknown(test_ph, near_ph),
        "AUROC_far": auroc_known_vs_unknown(test_ph, far_ph),
        "AUROC_all": auroc_known_vs_unknown(test_ph, all_ph),
        "acceptance_rate": acceptance_rate(test_ph, tau_ph),
        "near_FPR@95TPR": fpr_at_threshold(near_ph, tau_ph),
        "far_FPR@95TPR": fpr_at_threshold(far_ph, tau_ph),
    })
    return pd.DataFrame(rows)


def plot_score_distributions(prefix="vanilla", out_path="task4/results/score_distributions.png"):
    from sklearn.metrics import roc_curve

    train_d, test_d = load(prefix, "train"), load(prefix, "test")
    near_d, far_d = load(prefix, "near"), load(prefix, "far")
    maha = MahalanobisScorer().fit(train_d["features"], train_d["labels"])

    score_fns = {
        "MSP": lambda d: msp_score(d["logits"]),
        "MLS": lambda d: mls_score(d["logits"]),
        "Mahalanobis": lambda d: maha.score(d["features"]),
    }

    fig, axes = plt.subplots(1, len(score_fns), figsize=(5 * len(score_fns), 4))
    for ax, (name, fn) in zip(axes, score_fns.items()):
        test_s = fn(test_d)
        unknown_s = np.concatenate([fn(near_d), fn(far_d)])
        y_true = np.concatenate([np.zeros_like(test_s), np.ones_like(unknown_s)])
        y_score = np.concatenate([test_s, unknown_s])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title(name)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def run_failure_analysis(prefix="vanilla"):
    from torchvision.datasets import CIFAR100

    val_d, near_d, far_d = load(prefix, "val"), load(prefix, "near"), load(prefix, "far")
    val_s = mls_score(val_d["logits"])
    tau = threshold_at_val_percentile(val_s, 95)

    cifar100_meta = CIFAR100(root="/kaggle/working/data", train=False, download=True)
    class_names = cifar100_meta.classes

    near_s, far_s = mls_score(near_d["logits"]), mls_score(far_d["logits"])
    near_fail = find_incorrect_accepts(near_s, tau, near_d["logits"], near_d["labels"], class_names)
    far_fail = find_incorrect_accepts(far_s, tau, far_d["logits"], far_d["labels"], class_names)
    return near_fail, far_fail


if __name__ == "__main__":
    os.makedirs("task4/results", exist_ok=True)

    table1 = run_score_comparison_table("vanilla")
    print(table1)
    table1.to_csv("task4/results/table1_score_comparison.csv", index=False)

    table2 = run_model_comparison_table()
    print(table2)
    table2.to_csv("task4/results/table2_model_comparison.csv", index=False)

    plot_score_distributions("vanilla")

    near_fail, far_fail = run_failure_analysis("vanilla")
    print("near failures (first 3):", near_fail[:3])
    print("far failures (first 3):", far_fail[:3])
