
import argparse, os, json, yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from shared.pacs import PACSDataset, build_transforms, make_domain_balanced_loader, scan_pacs_domain
from shared.pacs_protocol import set_all_seeds, make_stratified_split, freeze_batchnorm
from task2.models.backbone import build_backbone
from task2.models.classifier_head import ClassifierHead
from task2.methods.source_only import SourceOnlyMethod
from task2.methods.dan import DANMethod
from task2.methods.dann import DANNMethod
from task2.methods.cdan import CDANMethod
from task2.evaluation.metrics import evaluate_on_loader, aggregate_source_val_metrics


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("defaults"):
        base_path = os.path.join(os.path.dirname(path), cfg["defaults"])
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        return {**base_cfg, **{k: v for k, v in cfg.items() if k != "defaults"}}
    return cfg


def build_method(method_name, backbone, head, cfg):
    if method_name == "source_only":
        return SourceOnlyMethod(backbone, head)
    elif method_name == "dan":
        d = cfg["dan"]
        return DANMethod(backbone, head, lambda_mmd=d["lambda_mmd"],
                          bandwidth_multipliers=tuple(d["bandwidth_multipliers"]))
    elif method_name == "dann":
        d = cfg["dann"]
        return DANNMethod(backbone, head, discriminator_hidden_dim=d["discriminator_hidden_dim"],
                           dropout=d["dropout"], domain_loss_weight=d["domain_loss_weight"],
                           grl_k=d["grl_schedule"]["k"], grl_max_alpha=d["grl_schedule"]["max_alpha"])
    elif method_name == "cdan":
        d = cfg["cdan"]
        return CDANMethod(backbone, head, num_classes=cfg["model"]["num_classes"],
                           discriminator_hidden_dim=d["discriminator_hidden_dim"], dropout=d["dropout"],
                           domain_loss_weight=d["domain_loss_weight"],
                           grl_k=d["grl_schedule"]["k"], grl_max_alpha=d["grl_schedule"]["max_alpha"])
    raise ValueError(method_name)


def train_method(cfg, data_root, output_dir="task2/results", num_workers=2):
    set_all_seeds(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sources = cfg["data"]["sources"]
    target = cfg["data"]["target"]
    train_tf, eval_tf = build_transforms("train"), build_transforms("eval")

    source_train_loaders, source_val_loaders = [], {}
    for dom_idx, dom in enumerate(sources):
        paths, labels = scan_pacs_domain(data_root, dom)
        tr_p, tr_l, va_p, va_l = make_stratified_split(paths, labels, seed=cfg["seed"],
                                                         val_frac=cfg["data"]["val_frac"])
        source_train_loaders.append(DataLoader(
            PACSDataset(tr_p, tr_l, dom_idx, train_tf),
            batch_size=cfg["training"]["per_source_batch_size"], shuffle=True,
            drop_last=True, num_workers=num_workers))
        source_val_loaders[dom] = DataLoader(
            PACSDataset(va_p, va_l, dom_idx, eval_tf), batch_size=64, shuffle=False, num_workers=num_workers)

    target_paths, target_labels = scan_pacs_domain(data_root, target)
    target_loader = DataLoader(
        PACSDataset(target_paths, target_labels, len(sources), train_tf),
        batch_size=cfg["training"]["target_batch_size"], shuffle=True, drop_last=True, num_workers=num_workers)

    combined_loader = make_domain_balanced_loader(source_train_loaders, target_loader)

    backbone = build_backbone(pretrained=True).to(device)
    head = ClassifierHead(in_dim=backbone.feature_dim, num_classes=cfg["model"]["num_classes"]).to(device)
    method = build_method(cfg["method"], backbone, head, cfg).to(device)

    optimizer = torch.optim.AdamW(method.parameters(), lr=cfg["training"]["lr"],
                                   weight_decay=cfg["training"]["weight_decay"])

    max_epochs, patience = cfg["training"]["max_epochs"], cfg["training"]["patience"]
    total_steps = len(combined_loader) * max_epochs
    global_step, best_score, epochs_no_improve = 0, -1.0, 0

    out_dir = os.path.join(output_dir, cfg["method"])
    os.makedirs(out_dir, exist_ok=True)
    log_lines = []

    for epoch in tqdm(range(max_epochs)):
        method.train()
        freeze_batchnorm(method)
        for source_batch, target_batch in combined_loader:
            s_imgs, s_cls, s_dom = [t.to(device) for t in source_batch]
            t_imgs, t_cls, t_dom = [t.to(device) for t in target_batch]
            progress_p = global_step / max(total_steps, 1)

            if cfg["method"] == "source_only":
                loss, logs = method.compute_loss((s_imgs, s_cls, s_dom))
            elif cfg["method"] == "dan":
                loss, logs = method.compute_loss((s_imgs, s_cls, s_dom), (t_imgs, t_cls, t_dom))
            else:
                loss, logs = method.compute_loss((s_imgs, s_cls, s_dom), (t_imgs, t_cls, t_dom), progress_p)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(method.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            log_lines.append({"epoch": epoch, "step": global_step, **logs})

        per_domain = {dom: evaluate_on_loader(method, loader, device) for dom, loader in source_val_loaders.items()}
        # evaluate_on_loader triggers model.eval(); switch back for next epoch's training loop
        agg = aggregate_source_val_metrics(per_domain)
        mean_f1 = agg["mean"]["macro_f1"]
        print(f"[{cfg['method']}] epoch {epoch}: mean source val macro-F1 = {mean_f1:.4f}")

        if mean_f1 > best_score:
            best_score, epochs_no_improve = mean_f1, 0
            torch.save(method.state_dict(), os.path.join(out_dir, "best_checkpoint.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log_lines, f)
    print(f"Done. Best mean source val macro-F1: {best_score:.4f} -> {out_dir}/best_checkpoint.pt")
    return best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="task2/results")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_method(cfg, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
