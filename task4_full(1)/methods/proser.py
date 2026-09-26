
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.cifar10 import get_cifar10_loaders
from models.resnet_cifar import ResNetCIFAR
from methods.manifold_mixup import manifold_mixup_pair


def classifier_placeholder_loss(logits, dummy_logits, labels, num_classes, beta=1.0):
    B = logits.size(0)
    device = logits.device

    # loss2 (ordinary term): true label must still win, with the dummy
    # logits included as extra distractor classes -- matches the reference
    # implementation's concatenation of dummy logits into the ordinary CE.
    combined_ordinary = torch.cat([logits, dummy_logits], dim=1)
    loss2 = F.cross_entropy(combined_ordinary, labels)

    # loss3 (exclusion term): mask out the true-class logit, then the
    # strongest dummy logit must beat every remaining known class. This is
    # the actual classifier-placeholder mechanism (dummy = a learned
    # boundary just outside the true class's decision region).
    masked_logits = logits.clone()
    masked_logits[torch.arange(B, device=device), labels] = -1e9
    dummy_max, _ = dummy_logits.max(dim=1, keepdim=True)
    combined_exclude = torch.cat([masked_logits, dummy_max], dim=1)
    target_dummy_slot = torch.full((B,), num_classes, dtype=torch.long, device=device)
    loss3 = F.cross_entropy(combined_exclude, target_dummy_slot)

    return loss2 + beta * loss3


def data_placeholder_loss(mixed_logits, mixed_dummy_logits, num_classes):
    # Manifold-mixed features (from two different known classes) should be
    # classified into the dummy slot among all K+1 outputs.
    B = mixed_logits.size(0)
    device = mixed_logits.device
    dummy_max, _ = mixed_dummy_logits.max(dim=1, keepdim=True)
    combined = torch.cat([mixed_logits, dummy_max], dim=1)
    target = torch.full((B,), num_classes, dtype=torch.long, device=device)
    return F.cross_entropy(combined, target)


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])

    train_loader, val_loader, _, _ = get_cifar10_loaders(
        cfg["data_dir"], batch_size=cfg["batch_size"], use_randaugment=False,
        seed=cfg["seed"], val_fraction=cfg["val_fraction"], num_workers=cfg["num_workers"],
    )

    num_classes = 10
    num_dummy = cfg["num_dummy"]
    model = ResNetCIFAR(num_classes=num_classes, num_dummy=num_dummy).to(device)

    ckpt = torch.load(cfg["init_checkpoint"], map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded vanilla checkpoint -- missing: {missing}, unexpected: {unexpected}")

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                                 momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    beta = cfg.get("beta", 1.0)
    gamma = cfg.get("gamma", 0.1)
    mixup_alpha = cfg.get("mixup_alpha", 2.0)

    best_val_acc = 0.0
    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)
            half = B // 2
            # First half: classifier placeholder (ordinary examples).
            x1, y1 = x[:half], y[:half]
            # Second half: data placeholder (manifold mixup).
            x2, y2 = x[half:], y[half:]

            optimizer.zero_grad()
            total_loss = 0.0

            if x1.size(0) > 0:
                _, logits1, dummy_logits1 = model(x1)
                cls_loss = classifier_placeholder_loss(
                    logits1, dummy_logits1, y1, num_classes, beta=beta
                )
                total_loss = total_loss + cls_loss

            if x2.size(0) > 1:
                h2 = model.forward_stem_to_layer2(x2)
                mixed_h, _, _, _ = manifold_mixup_pair(h2, y2, alpha=mixup_alpha)
                _, mixed_logits, mixed_dummy_logits = model.forward_from_layer2(mixed_h)
                data_loss = data_placeholder_loss(mixed_logits, mixed_dummy_logits, num_classes)
                total_loss = total_loss + gamma * data_loss

            total_loss.backward()
            optimizer.step()

        scheduler.step()
        val_acc = evaluate_known_only(model, val_loader, device)
        print(f"[proser] epoch {epoch+1}/{cfg['epochs']} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "val_acc": val_acc},
                       cfg["checkpoint_path"])

    print(f"[proser] best val acc: {best_val_acc:.4f}")
    return best_val_acc


@torch.no_grad()
def evaluate_known_only(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, logits, _ = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total
