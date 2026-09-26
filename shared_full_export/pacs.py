
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard PACS class set (alphabetical)
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


def build_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif split == "eval":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    raise ValueError(f"unknown split: {split}")


def scan_pacs_domain(root, domain):
    """Expects root/<domain>/<class_name>/*.jpg|png. Returns (paths, labels)."""
    domain_dir = os.path.join(root, domain)
    paths, labels = [], []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = os.path.join(domain_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)
    if not paths:
        raise RuntimeError(f"No images found under {domain_dir} — check your data_root / folder layout.")
    return paths, labels


class PACSDataset(Dataset):
    def __init__(self, image_paths, class_labels, domain_label, transform):
        self.image_paths = image_paths
        self.class_labels = class_labels
        self.domain_label = domain_label
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.class_labels[idx], self.domain_label


class DomainBalancedIterator:
    def __init__(self, source_loaders, target_loader, steps_per_epoch=None):
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.steps_per_epoch = steps_per_epoch or max(len(l) for l in source_loaders)

    def __iter__(self):
        source_iters = [iter(l) for l in self.source_loaders]
        target_iter = iter(self.target_loader)
        for _ in range(self.steps_per_epoch):
            imgs_list, cls_list, dom_list = [], [], []
            for i, it in enumerate(source_iters):
                try:
                    imgs, cls, dom = next(it)
                except StopIteration:
                    source_iters[i] = iter(self.source_loaders[i])
                    imgs, cls, dom = next(source_iters[i])
                imgs_list.append(imgs); cls_list.append(cls); dom_list.append(dom)
            source_batch = (torch.cat(imgs_list), torch.cat(cls_list), torch.cat(dom_list))
            try:
                t_imgs, t_cls, t_dom = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_loader)
                t_imgs, t_cls, t_dom = next(target_iter)
            yield source_batch, (t_imgs, t_cls, t_dom)

    def __len__(self):
        return self.steps_per_epoch


def make_domain_balanced_loader(source_loaders, target_loader, steps_per_epoch=None):
    return DomainBalancedIterator(source_loaders, target_loader, steps_per_epoch)
