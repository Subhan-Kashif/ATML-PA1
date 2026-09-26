
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.cifar10 import get_cifar10_loaders
from models.resnet_cifar import ResNetCIFAR

def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])

    train_loader, val_loader, _, _ = get_cifar10_loaders(
        cfg["data_dir"], batch_size=cfg["batch_size"],
        use_randaugment=cfg.get("use_randaugment", False),
        seed=cfg["seed"], val_fraction=cfg["val_fraction"],
        num_workers=cfg["num_workers"],
    )

    model = ResNetCIFAR(num_classes=10, num_dummy=0).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                                 momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val_acc = 0.0
    for epoch in range(cfg["epochs"]):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        print(f"[{cfg['method']}] epoch {epoch+1}/{cfg['epochs']} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "val_acc": val_acc},
                       cfg["checkpoint_path"])

    print(f"[{cfg['method']}] best val acc: {best_val_acc:.4f}")
    return best_val_acc

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total
