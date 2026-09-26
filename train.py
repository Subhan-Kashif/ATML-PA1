
import argparse, os, json, yaml
import torch
from torch.utils.data import DataLoader

from shared.pacs import PACSDataset, build_transforms, scan_pacs_domain
from shared.pacs_protocol import set_all_seeds, make_stratified_split, freeze_batchnorm
from task3.models.backbone import build_backbone
from task3.models.classifier_head import ClassifierHead
from task3.methods.dan_dg import DANDGMethod
from task3.methods.sam import SAMMethod, SAMPerturber
from task3.selection.source_validation import evaluate_all_sources, checkpoint_selection_score
from tqdm import tqdm

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_dan_dg_batches(source_iters, source_loaders, device):
    """Pulls one batch per source domain, cycling any loader that
    exhausts. Returns dict {domain_name: (imgs, labels, dom_labels)}."""
    batches = {}
    for dom, it in list(source_iters.items()):
        try:
            imgs, labels, doms = next(it)
        except StopIteration:
            source_iters[dom] = iter(source_loaders[dom])
            imgs, labels, doms = next(source_iters[dom])
        batches[dom] = (imgs.to(device), labels.to(device), doms.to(device))
    return batches


def train_method(cfg, data_root, output_dir="task3/results", num_workers=2):
    set_all_seeds(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sources = cfg["data"]["sources"]
    train_tf, eval_tf = build_transforms("train"), build_transforms("eval")

    source_train_loaders, source_val_loaders = {}, {}
    per_source_bs = cfg["training"]["per_source_batch_size"]
    for dom in sources:
        paths, labels = scan_pacs_domain(data_root, dom)
        tr_p, tr_l, va_p, va_l = make_stratified_split(paths, labels, seed=cfg["seed"],
                                                         val_frac=cfg["data"]["val_frac"])
        dom_idx = sources.index(dom)
        source_train_loaders[dom] = DataLoader(
            PACSDataset(tr_p, tr_l, dom_idx, train_tf),
            batch_size=per_source_bs, shuffle=True, drop_last=True, num_workers=num_workers)
        source_val_loaders[dom] = DataLoader(
            PACSDataset(va_p, va_l, dom_idx, eval_tf), batch_size=64, shuffle=False, num_workers=num_workers)

    backbone = build_backbone(pretrained=True).to(device)
    head = ClassifierHead(in_dim=backbone.feature_dim, num_classes=cfg["model"]["num_classes"]).to(device)

    method_name = cfg["method"]
    if method_name == "dan_dg":
        d = cfg["dan_dg"]
        method = DANDGMethod(backbone, head, lambda_dg=d["lambda_dg"],
                              bandwidth_multipliers=tuple(d["bandwidth_multipliers"])).to(device)
    elif method_name == "sam":
        method = SAMMethod(backbone, head).to(device)
        sam_perturber = SAMPerturber(method.parameters(), rho=cfg["sam"]["rho"])
    else:
        raise ValueError(f"train.py only trains dan_dg/sam, got {method_name}")

    optimizer = torch.optim.AdamW(method.parameters(), lr=cfg["training"]["lr"],
                                   weight_decay=cfg["training"]["weight_decay"])

    max_epochs, patience = cfg["training"]["max_epochs"], cfg["training"]["patience"]
    steps_per_epoch = max(len(l) for l in source_train_loaders.values())
    global_step, best_score, epochs_no_improve = 0, -1.0, 0

    out_dir = os.path.join(output_dir, method_name)
    os.makedirs(out_dir, exist_ok=True)
    log_lines, epoch_val_log = [], []

    for epoch in tqdm(range(max_epochs)):
        method.train()
        freeze_batchnorm(method)
        source_iters = {dom: iter(loader) for dom, loader in source_train_loaders.items()}

        for _ in range(steps_per_epoch):
            batches = build_dan_dg_batches(source_iters, source_train_loaders, device)

            if method_name == "dan_dg":
                loss, logs = method.compute_loss(batches)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:  # sam
                pooled_imgs = torch.cat([batches[d][0] for d in sources])
                pooled_labels = torch.cat([batches[d][1] for d in sources])
                pooled_doms = torch.cat([batches[d][2] for d in sources])
                pooled_batch = (pooled_imgs, pooled_labels, pooled_doms)

                loss1, logs = method.compute_loss(pooled_batch)
                loss1.backward()
                sam_perturber.first_step()

                loss2, _ = method.compute_loss(pooled_batch)
                loss2.backward()
                sam_perturber.second_step(optimizer)

            global_step += 1
            log_lines.append({"epoch": epoch, "step": global_step, **logs})

        all_sources_result = evaluate_all_sources(method, source_val_loaders, device)
        mean_f1 = checkpoint_selection_score(all_sources_result)
        worst_f1 = all_sources_result["worst"]["macro_f1"]
        print(f"[{method_name}] epoch {epoch}: mean_f1={mean_f1:.4f} worst_f1={worst_f1:.4f}")
        epoch_val_log.append({"epoch": epoch, "mean_source_val_f1": mean_f1, "worst_source_val_f1": worst_f1})

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
    with open(os.path.join(out_dir, "val_log.json"), "w") as f:
        json.dump(epoch_val_log, f)
    print(f"Done. Best mean source val macro-F1: {best_score:.4f} -> {out_dir}/best_checkpoint.pt")
    return best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="task3/results")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_method(cfg, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
