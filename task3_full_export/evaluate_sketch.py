
import argparse, os, json, copy
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from shared.pacs import PACSDataset, build_transforms, scan_pacs_domain, CLASSES
from shared.pacs_protocol import make_stratified_split, SEED, set_all_seeds
from task3.models.backbone import build_backbone
from task3.models.classifier_head import ClassifierHead
from task3.methods.erm import ERMMethod, load_erm_from_task2_checkpoint
from task3.methods.dan_dg import DANDGMethod
from task3.methods.sam import SAMMethod
from task3.train import load_config, train_method
from task3.selection.source_validation import evaluate_all_sources
from task3.evaluation.domain_metrics import (
    evaluate_on_loader, aggregate_source_metrics, sketch_accuracy_delta,
    per_class_accuracy, compare_to_erm, top_confusions,
)
from task3.evaluation.source_domain_separability import extract_features, compute_source_domain_separability
from task3.evaluation.sharpness import build_fixed_sharpness_batch, compute_sharpness

METHODS = ["erm", "dan_dg", "sam"]
DAN_DG_SWEEP = [0.1, 1.0, 10.0]
SAM_SWEEP = [0.01, 0.05, 0.1]


def build_loaders(cfg, data_root):
    eval_tf = build_transforms("eval")
    sources, target = cfg["data"]["sources"], cfg["data"]["target"]

    source_val_loaders, source_val_datasets = {}, {}
    for dom_idx, dom in enumerate(sources):
        paths, labels = scan_pacs_domain(data_root, dom)
        _, _, va_p, va_l = make_stratified_split(paths, labels, seed=cfg["seed"], val_frac=cfg["data"]["val_frac"])
        ds = PACSDataset(va_p, va_l, dom_idx, eval_tf)
        source_val_datasets[dom] = ds
        source_val_loaders[dom] = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    # Sketch -- loaded ONLY here, in this file
    target_paths, target_labels = scan_pacs_domain(data_root, target)
    sketch_loader = DataLoader(PACSDataset(target_paths, target_labels, len(sources), eval_tf),
                                batch_size=64, shuffle=False, num_workers=2)
    return source_val_loaders, source_val_datasets, sketch_loader


def build_model_for_method(method_name, cfg, device):
    backbone = build_backbone(pretrained=True).to(device)
    head = ClassifierHead(in_dim=backbone.feature_dim, num_classes=cfg["model"]["num_classes"]).to(device)
    if method_name == "erm":
        return ERMMethod(backbone, head).to(device)
    elif method_name == "dan_dg":
        d = cfg["dan_dg"]
        return DANDGMethod(backbone, head, lambda_dg=d["lambda_dg"],
                            bandwidth_multipliers=tuple(d["bandwidth_multipliers"])).to(device)
    elif method_name == "sam":
        return SAMMethod(backbone, head).to(device)
    raise ValueError(method_name)


def load_checkpoint(method_name, cfg, results_dir, device):
    model = build_model_for_method(method_name, cfg, device)
    if method_name == "erm":
        ckpt_path = cfg["checkpoint"]["path"]  # points at the Task 2 checkpoint (see erm.yaml)
    else:
        ckpt_path = os.path.join(results_dir, method_name, "best_checkpoint.pt")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_checkpoint(method_name, cfg, results_dir, source_val_loaders, sketch_loader, source_val_datasets, device):
    model = load_checkpoint(method_name, cfg, results_dir, device)

    all_sources_result = evaluate_all_sources(model, source_val_loaders, device)
    sketch_metrics = evaluate_on_loader(model, sketch_loader, device)

    features_by_domain = {dom: extract_features(model, loader, device)
                           for dom, loader in source_val_loaders.items()}
    sep_score = compute_source_domain_separability(features_by_domain, seed=SEED)

    fixed_batch_loader = build_fixed_sharpness_batch(source_val_datasets, per_domain_n=32, seed=SEED)
    sharp_score = compute_sharpness(model, fixed_batch_loader, device, rho=0.05)

    return model, all_sources_result, sketch_metrics, sep_score, sharp_score


def plot_training_curves(method, results_dir):
    log_path = os.path.join(results_dir, method, "train_log.json")
    if not os.path.exists(log_path):
        print(f"skip curves for {method}: no train_log.json (expected for erm -- not retrained)")
        return
    with open(log_path) as f:
        log = json.load(f)
    if not log:
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


def run_main_comparison(data_root, results_dir, configs_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg0 = load_config(os.path.join(configs_dir, "erm.yaml"))
    set_all_seeds(cfg0["seed"])
    source_val_loaders, source_val_datasets, sketch_loader = build_loaders(cfg0, data_root)
    sources = cfg0["data"]["sources"]

    all_results, all_sketch_eval = {}, {}
    for method_name in METHODS:
        cfg = load_config(os.path.join(configs_dir, f"{method_name}.yaml"))
        _, all_sources_result, sketch_metrics, sep_score, sharp_score = evaluate_checkpoint(
            method_name, cfg, results_dir, source_val_loaders, sketch_loader, source_val_datasets, device)
        all_results[method_name] = {
            "per_domain": all_sources_result, "sketch": sketch_metrics,
            "separability": sep_score, "sharpness": sharp_score,
        }
        all_sketch_eval[method_name] = sketch_metrics
        print(f"{method_name}: mean_acc={all_sources_result['mean']['accuracy']:.4f} "
              f"worst_acc={all_sources_result['worst']['accuracy']:.4f} "
              f"sketch_acc={sketch_metrics['accuracy']:.4f} sketch_f1={sketch_metrics['macro_f1']:.4f} "
              f"sep={sep_score:.4f} sharp={sharp_score:.4f}")

    rows = []
    for method_name in METHODS:
        r = all_results[method_name]
        row = {"method": method_name}
        for d in sources:
            row[f"{d}_acc"] = r["per_domain"][d]["accuracy"]
            row[f"{d}_f1"] = r["per_domain"][d]["macro_f1"]
        row["mean_acc"] = r["per_domain"]["mean"]["accuracy"]
        row["mean_f1"] = r["per_domain"]["mean"]["macro_f1"]
        row["worst_acc"] = r["per_domain"]["worst"]["accuracy"]
        row["worst_f1"] = r["per_domain"]["worst"]["macro_f1"]
        row["sketch_acc"] = r["sketch"]["accuracy"]
        row["sketch_f1"] = r["sketch"]["macro_f1"]
        row["sketch_acc_delta"] = sketch_accuracy_delta(r["sketch"], all_results["erm"]["sketch"])
        row["separability"] = r["separability"]
        row["sharpness"] = r["sharpness"]
        rows.append(row)

    comparison_df = pd.DataFrame(rows).set_index("method")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n=== Task 3 Step 4: Comparison Table ===")
    print(comparison_df.to_string())

    erm_per_class = per_class_accuracy(all_sketch_eval["erm"]["logits"], all_sketch_eval["erm"]["labels"], num_classes=len(CLASSES))
    erm_confusions = top_confusions(all_sketch_eval["erm"]["logits"], all_sketch_eval["erm"]["labels"], CLASSES, k=5)
    class_report = {"erm": {"per_class_acc": {CLASSES[c]: v for c, v in erm_per_class.items()}, "top_confusions": erm_confusions}}
    for method_name in METHODS:
        if method_name == "erm":
            continue
        pc = per_class_accuracy(all_sketch_eval[method_name]["logits"], all_sketch_eval[method_name]["labels"], num_classes=len(CLASSES))
        deltas = compare_to_erm(pc, erm_per_class)
        confusions = top_confusions(all_sketch_eval[method_name]["logits"], all_sketch_eval[method_name]["labels"], CLASSES, k=5)
        class_report[method_name] = {
            "per_class_acc": {CLASSES[c]: v for c, v in pc.items()},
            "delta_vs_erm": {CLASSES[c]: v for c, v in deltas.items()},
            "top_confusions": confusions,
        }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "class_analysis.json"), "w") as f:
        json.dump(class_report, f, indent=2)
    print(f"Wrote {results_dir}/class_analysis.json")

    for method_name in METHODS:
        plot_training_curves(method_name, results_dir)

    return comparison_df


def run_ablation(which, data_root, results_dir, configs_dir):
    assert which in ("dan_dg", "sam")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = load_config(os.path.join(configs_dir, f"{which}.yaml"))
    set_all_seeds(base_cfg["seed"])
    source_val_loaders, source_val_datasets, sketch_loader = build_loaders(base_cfg, data_root)

    ablation_dir = os.path.join(results_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)
    sweep_values = DAN_DG_SWEEP if which == "dan_dg" else SAM_SWEEP
    rows = []

    for val in sweep_values:
        cfg = copy.deepcopy(base_cfg)
        run_name = f"{which}_sweep_{val}"
        if which == "dan_dg":
            cfg["dan_dg"]["lambda_dg"] = val
        else:
            cfg["sam"]["rho"] = val
        cfg["method"] = which

        print(f"\n=== Training {run_name} ===")
        train_method(cfg, data_root, output_dir=ablation_dir)

        ckpt_path = os.path.join(ablation_dir, which, "best_checkpoint.pt")
        model = build_model_for_method(which, cfg, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        all_sources_result = evaluate_all_sources(model, source_val_loaders, device)
        sketch_metrics = evaluate_on_loader(model, sketch_loader, device)
        features_by_domain = {dom: extract_features(model, loader, device) for dom, loader in source_val_loaders.items()}
        sep_score = compute_source_domain_separability(features_by_domain, seed=SEED)
        fixed_batch_loader = build_fixed_sharpness_batch(source_val_datasets, per_domain_n=32, seed=SEED)
        sharp_score = compute_sharpness(model, fixed_batch_loader, device, rho=0.05)

        row = {
            "setting_value": val,
            "mean_acc": all_sources_result["mean"]["accuracy"],
            "mean_f1": all_sources_result["mean"]["macro_f1"],
            "worst_f1": all_sources_result["worst"]["macro_f1"],
            "sketch_acc": sketch_metrics["accuracy"],
            "sketch_f1": sketch_metrics["macro_f1"],
            "separability": sep_score,
            "sharpness": sharp_score,
        }
        rows.append(row)
        print(f"{run_name}: {row}")
        os.rename(ckpt_path, os.path.join(ablation_dir, which, f"checkpoint_{val}.pt"))

    ablation_df = pd.DataFrame(rows).set_index("setting_value")
    print(f"\n=== Task 3 Step 5: {which.upper()} Ablation Table ===")
    print(ablation_df.to_string())

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    xvals = [r["setting_value"] for r in rows]
    axes[0].plot(xvals, [r["mean_f1"] for r in rows], marker="o"); axes[0].set_title("Mean source F1")
    axes[1].plot(xvals, [r["sketch_f1"] for r in rows], marker="o"); axes[1].set_title("Sketch F1")
    axes[2].plot(xvals, [r["separability"] for r in rows], marker="o"); axes[2].set_title("Source separability")
    axes[3].plot(xvals, [r["sharpness"] for r in rows], marker="o"); axes[3].set_title("Sharpness (Δ)")
    for ax in axes:
        ax.set_xlabel(f"{which} setting")
    plt.tight_layout()
    plot_path = os.path.join(ablation_dir, f"{which}_ablation_plot.png")
    plt.savefig(plot_path, dpi=120)
    plt.show()
    print(f"Wrote {plot_path}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="task3/results")
    parser.add_argument("--configs_dir", type=str, default="task3/configs")
    parser.add_argument("--run_ablation", choices=["none", "dan_dg", "sam"], default="none")
    args = parser.parse_args()

    run_main_comparison(args.data_root, args.results_dir, args.configs_dir)
    if args.run_ablation != "none":
        run_ablation(args.run_ablation, args.data_root, args.results_dir, args.configs_dir)


if __name__ == "__main__":
    main()
