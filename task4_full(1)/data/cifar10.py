
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from data.make_splits import stratified_split

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def _train_transform(use_randaugment=False):
    t = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if use_randaugment:
        t.append(transforms.RandAugment(num_ops=2, magnitude=9))
    t += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(t)

def _eval_transform():
    return transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

def get_cifar10_loaders(data_dir, batch_size=128, use_randaugment=False,
                         seed=6304, val_fraction=0.1, num_workers=2):
    base_train_aug = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                       transform=_train_transform(use_randaugment))
    base_train_eval = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                        transform=_eval_transform())
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True,
                                 transform=_eval_transform())

    labels = np.array(base_train_aug.targets)
    train_idx, val_idx = stratified_split(labels, val_fraction=val_fraction, seed=seed)

    train_set = Subset(base_train_aug, train_idx)
    val_set = Subset(base_train_eval, val_idx)
    train_eval_set = Subset(base_train_eval, train_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    train_eval_loader = DataLoader(train_eval_set, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_eval_loader
