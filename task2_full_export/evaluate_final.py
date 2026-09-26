
import argparse, os, json, csv, copy
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from shared.pacs import PACSDataset, build_transforms, scan_pacs_domain, CLASSES
from shared.pacs_protocol import make_stratified_split, SEED
from task2.models.backbone import build_backbone
from task2.models.classifier_head import ClassifierHead
from task2.train import load_config, build_method, train_method
from task2.evaluation.metrics import evaluate_on_loader, aggregate_source_val_metrics, target_accuracy_delta
from task2.evaluation.domain_separability import extract_features, compute_domain_separability
from task2.evaluation.class_analysis import per_class_accuracy, compare_to_source_only, top_confusions

METHODS = ["source_only", "dan", "dann", "cdan"]
DAN_SWEEP = [0.1, 1.0, 10.0]
DANN_SWEEP = [0.25, 0.5, 1.0]


def build_source_val_and_target_loaders(cfg, data_root):
    eval_tf = build_transforms("eval")
    sources, target = cfg["data"]["sources"], cfg["data"]["target"]
    source_val_loaders = {}
    for dom_idx, dom in enumerate(sources):
        paths, labels = scan_pacs_domain(data_root, dom)
        _, _, va_p, va_l = make_stratified_split(paths, labels, seed=cfg["seed"], val_frac=cfg["data"]["val_frac"])
        source_val_loaders[dom] = DataLoader(PACSDataset(va_p, va_l, dom_idx, eval_tf),
                                              batch_size=64, shuffle=False, num_workers=2)
    target_paths, target_labels = scan_pacs_domain(data_root, target)
    target_loader = DataLoader(PACSDataset(target_paths, target_labels, len(sources), eval_tf),
                                batch_size=64, shuffle=False, num_workers=2)
    return source_val_loaders, target_loader


def evaluate_checkpoint(method_name, cfg, ckpt_path, source_val_loaders, target_loader, device):
    backbone = build_backbone(pretrained=True).to(device)
    head = ClassifierHead(in_dim=backbone.feature_dim, num_classes=cfg["model"]["num_classes"]).to(device)
    method = build_method(method_name, backbone, head, cfg).to(device)
    method.load_state_dict(torch.load(ckpt_path, map_location=device))
    method.eval()

    per_domain = {dom: evaluate_on_loader(method, loader, device) for dom, loader in source_val_loaders.items()}
    agg = aggregate_source_val_metrics(per_domain)
    target_metrics = evaluate_on_loader(method, target_loader, device)

    source_feats = np.concatenate(
        [extract_features(method, loader, device) for loader in source_val_loaders.values()], axis=0)
    target_feats = extract_features(method, target_loader, device)
    sep_score = compute_domain_separability(source_feats, target_feats, seed=SEED)

    return method, agg, target_metrics, sep_score


# ------------------------------------------------------------
# Required Evidence -- loss / alignment curves per method
# ------------------------------------------------------------
def plot_training_curves(method, results_dir):
    log_path = os.path.join(results_dir, method, "train_log.json")
    if not os.path.exists(log_path):
        print(f"skip curves for {method}: no train_log.json found")
        return
    with open(log_path) as f:
        log = json.load(f)
    if not log:
        print(f"skip curves for {method}: empty train_log.json")
        return

    loss_keys = [k for k in log[0].keys() if k not in ("epoch", "step") and "loss" in k]
    fig, axes = plt.subplots(1, len(loss_keys), figsize=(5 * len(loss_keys), 3))
    if len(loss_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, loss_keys):
        by_epoch = defaultdict(list)
        for l in log:
            if key in l:
                by_epoch[l["epoch"]].append(l[key])
        epochs = sorted(by_epoch.keys())
        means = [sum(by_epoch[e]) / len(by_epoch[e]) for e in epochs]
        ax.plot(epochs, means, marker="o")
        ax.set_title(f"{method} mean {key} per epoch")
        ax.set_xlabel("epoch")

    plt.tight_layout()
    save_path = os.path.join(results_dir, method, f"{method}_loss_curves.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"saved {save_path}")


# ------------------------------------------------------------
# Step 5 -- main comparison across the four locked checkpoints
# ------------------------------------------------------------
def run_main_comparison(data_root, results_dir, configs_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg0 = load_config(os.path.join(configs_dir, "source_only.yaml"))
    source_val_loaders, target_loader = build_source_val_and_target_loaders(cfg0, data_root)
    sources = cfg0["data"]["sources"]

    all_results, all_target_eval = {}, {}
    for method_name in METHODS:
        cfg = load_config(os.path.join(configs_dir, f"{method_name}.yaml"))
        ckpt_path = os.path.join(results_dir, method_name, "best_checkpoint.pt")
        _, agg, target_metrics, sep_score = evaluate_checkpoint(
            method_name, cfg, ckpt_path, source_val_loaders, target_loader, device)
        all_results[method_name] = {"per_domain": agg, "target": target_metrics, "separability": sep_score}
        all_target_eval[method_name] = target_metrics
        print(f"{method_name}: mean_source_acc={agg['mean']['accuracy']:.4f} "
              f"mean_source_f1={agg['mean']['macro_f1']:.4f} "
              f"target_acc={target_metrics['accuracy']:.4f} target_f1={target_metrics['macro_f1']:.4f} "
              f"sep={sep_score:.4f}")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "comparison_table.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        header = ["method"] + [f"{d}_acc" for d in sources] + [f"{d}_f1" for d in sources] + \
                 ["mean_source_acc", "mean_source_f1", "target_acc", "target_f1", "target_acc_delta", "separability"]
        writer.writerow(header)
        for method_name in METHODS:
            r = all_results[method_name]
            row = [method_name]
            row += [r["per_domain"][d]["accuracy"] for d in sources]
            row += [r["per_domain"][d]["macro_f1"] for d in sources]
            row += [r["per_domain"]["mean"]["accuracy"], r["per_domain"]["mean"]["macro_f1"]]
            row += [r["target"]["accuracy"], r["target"]["macro_f1"]]
            row += [target_accuracy_delta(r["target"], all_results["source_only"]["target"]), r["separability"]]
            writer.writerow(row)
    print(f"Wrote {results_dir}/comparison_table.csv")

    # --- per-class analysis, including Source-only itself (needed for RQ1) ---
    so_per_class = per_class_accuracy(all_target_eval["source_only"]["logits"],
                                       all_target_eval["source_only"]["labels"], num_classes=len(CLASSES))
    so_confusions = top_confusions(all_target_eval["source_only"]["logits"],
                                    all_target_eval["source_only"]["labels"], CLASSES, k=5)
    class_report = {
        "source_only": {
            "per_class_acc": {CLASSES[c]: v for c, v in so_per_class.items()},
            "top_confusions": so_confusions,
        }
    }
    for method_name in METHODS:
        if method_name == "source_only":
            continue
        pc = per_class_accuracy(all_target_eval[method_name]["logits"],
                                 all_target_eval[method_name]["labels"], num_classes=len(CLASSES))
        deltas = compare_to_source_only(pc, so_per_class)
        confusions = top_confusions(all_target_eval[method_name]["logits"],
                                     all_target_eval[method_name]["labels"], CLASSES, k=5)
        class_report[method_name] = {
            "per_class_acc": {CLASSES[c]: v for c, v in pc.items()},
            "delta_vs_source_only": {CLASSES[c]: v for c, v in deltas.items()},
            "top_confusions": confusions,
        }
    with open(os.path.join(results_dir, "class_analysis.json"), "w") as f:
        json.dump(class_report, f, indent=2)
    print(f"Wrote {results_dir}/class_analysis.json")

    # --- Required Evidence: loss / alignment curves, saved as PNGs ---
    for method_name in METHODS:
        plot_training_curves(method_name, results_dir)


# ------------------------------------------------------------
# Step 6 -- controlled design study (DAN lambda_mmd OR DANN grl_max_alpha)
# ------------------------------------------------------------
def run_ablation(which, data_root, results_dir, configs_dir):
    assert which in ("dan", "dann")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = load_config(os.path.join(configs_dir, f"{which}.yaml"))
    source_val_loaders, target_loader = build_source_val_and_target_loaders(base_cfg, data_root)

    ablation_dir = os.path.join(results_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)
    sweep_values = DAN_SWEEP if which == "dan" else DANN_SWEEP
    rows = []

    for val in sweep_values:
        cfg = copy.deepcopy(base_cfg)
        run_name = f"{which}_sweep_{val}"
        if which == "dan":
            cfg["dan"]["lambda_mmd"] = val
        else:
            cfg["dann"]["grl_schedule"]["max_alpha"] = val
        cfg["method"] = which

        print(f"\n=== Training {run_name} ===")
        train_method(cfg, data_root, output_dir=ablation_dir)

        ckpt_path = os.path.join(ablation_dir, which, "best_checkpoint.pt")
        _, agg, target_metrics, sep_score = evaluate_checkpoint(
            which, cfg, ckpt_path, source_val_loaders, target_loader, device)

        row = {
            "setting_value": val,
            "mean_source_acc": agg["mean"]["accuracy"],
            "mean_source_f1": agg["mean"]["macro_f1"],
            "target_acc": target_metrics["accuracy"],
            "target_f1": target_metrics["macro_f1"],
            "separability": sep_score,
        }
        rows.append(row)
        print(f"{run_name}: {row}")

        os.rename(ckpt_path, os.path.join(ablation_dir, which, f"checkpoint_{val}.pt"))
        # keep the per-run train_log too, renamed so later sweep values don't overwrite it
        log_src = os.path.join(ablation_dir, which, "train_log.json")
        if os.path.exists(log_src):
            os.rename(log_src, os.path.join(ablation_dir, which, f"train_log_{val}.json"))

    out_csv = os.path.join(ablation_dir, f"{which}_ablation_table.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")

    # --- compact plot for the ablation, required by "Required Evidence" ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xvals = [r["setting_value"] for r in rows]
    axes[0].plot(xvals, [r["mean_source_acc"] for r in rows], marker="o")
    axes[0].set_title("Mean source accuracy"); axes[0].set_xlabel(f"{which} setting")
    axes[1].plot(xvals, [r["separability"] for r in rows], marker="o")
    axes[1].set_title("Domain separability"); axes[1].set_xlabel(f"{which} setting")
    axes[2].plot(xvals, [r["target_acc"] for r in rows], marker="o")
    axes[2].set_title("Target accuracy"); axes[2].set_xlabel(f"{which} setting")
    plt.tight_layout()
    plot_path = os.path.join(ablation_dir, f"{which}_ablation_plot.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Wrote {plot_path}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="task2/results")
    parser.add_argument("--configs_dir", type=str, default="task2/configs")
    parser.add_argument("--run_ablation", choices=["none", "dan", "dann"], default="none",
                         help="Step 6: run the controlled design study for this method after the main comparison.")
    args = parser.parse_args()

    run_main_comparison(args.data_root, args.results_dir, args.configs_dir)

    if args.run_ablation != "none":
        run_ablation(args.run_ablation, args.data_root, args.results_dir, args.configs_dir)


if __name__ == "__main__":
    main()
