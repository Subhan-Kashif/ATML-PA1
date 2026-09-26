
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from data.cifar10 import CIFAR10_MEAN, CIFAR10_STD

NEAR_CLASSES = ["bus", "pickup_truck", "motorcycle", "tractor",
                "wolf", "fox", "leopard", "camel"]
FAR_CLASSES = ["bottle", "bowl", "chair", "clock",
               "keyboard", "mushroom", "sunflower", "wardrobe"]

def _eval_transform():
    return transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

def get_cifar100_unknown_loaders(data_dir, batch_size=128, num_workers=2):
    test_set = datasets.CIFAR100(root=data_dir, train=False, download=True,
                                  transform=_eval_transform())
    class_to_idx = {name: i for i, name in enumerate(test_set.classes)}

    near_ids = {class_to_idx[c] for c in NEAR_CLASSES}
    far_ids = {class_to_idx[c] for c in FAR_CLASSES}

    near_indices = [i for i, t in enumerate(test_set.targets) if t in near_ids]
    far_indices = [i for i, t in enumerate(test_set.targets) if t in far_ids]

    near_set = Subset(test_set, near_indices)
    far_set = Subset(test_set, far_indices)
    all_set = Subset(test_set, near_indices + far_indices)

    near_loader = DataLoader(near_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    far_loader = DataLoader(far_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_loader = DataLoader(all_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    assert len(near_indices) == 800, f"expected 800 near images, got {len(near_indices)}"
    assert len(far_indices) == 800, f"expected 800 far images, got {len(far_indices)}"

    return near_loader, far_loader, all_loader
