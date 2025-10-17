# dp_utils_diffusion.py
# Utility for loading CIFAR-10 + Diffusion-generated CIFAR data (e.g., EDM-1M)
# Works with animal_classes (2–7) and vehicle_classes (0,1,8,9)

import os
from typing import List
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision
import torchvision.transforms as T

# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# --------------------------- helper: base dataset ---------------------------
def _build_cifar10(root: str, train: bool, batch_size: int, num_workers=4):
    """Return standard CIFAR-10 DataLoader."""
    tfm = T.Compose([
        T.RandomCrop(32, padding=4) if train else T.Lambda(lambda x: x),
        T.RandomHorizontalFlip() if train else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())
    return ds, loader


# --------------------------- helper: subset & remap ---------------------------
def _filter_indices(ds: torchvision.datasets.CIFAR10, keep: List[int]):
    """Return indices and label remapping dict for given subset."""
    idx = [i for i, (_, y) in enumerate(ds) if y in keep]
    remap = {old: new for new, old in enumerate(keep)}
    return idx, remap


class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices, remap):
        self.base = base
        self.indices = indices
        self.remap = remap
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, self.remap[int(y)]


# --------------------------- main: build combined dataset ---------------------------
def build_diffusion_augmented_loader(
    real_root: str,
    diff_root: str,
    keep_labels: List[int],
    batch_size: int,
    train: bool,
    num_workers=4
):
    """
    Build DataLoader combining real CIFAR-10 and diffusion-generated .npz data (EDM-1M).
    keep_labels: list of labels to keep (e.g., [0,1,8,9] for vehicles).
    diff_root: directory containing 1M.npz file.
    """
    import numpy as np
    from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, Subset

    # Transform
    tfm = T.Compose([
        T.RandomCrop(32, padding=4) if train else T.Lambda(lambda x: x),
        T.RandomHorizontalFlip() if train else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # 1️⃣ Load real CIFAR
    real_ds = torchvision.datasets.CIFAR10(root=real_root, train=train, download=True, transform=tfm)
    real_idx, real_remap = _filter_indices(real_ds, keep_labels)
    real_sub = RemappedSubset(real_ds, real_idx, real_remap)

    # 2️⃣ Load diffusion data (.npz)
    npz_path = os.path.join(diff_root, "1M.npz")
    print(f"[Loading diffusion data from {npz_path}]")
    data = np.load(npz_path)
    imgs = data["images"] / 255.0  # scale to [0,1]
    labels = data["labels"].astype(int)

    # Convert to tensor
    imgs = torch.tensor(imgs).permute(0, 3, 1, 2).float()
    labels = torch.tensor(labels).long()

    # Filter only the classes we want
    mask = torch.isin(labels, torch.tensor(keep_labels))
    imgs, labels = imgs[mask], labels[mask]
    diff_subset = TensorDataset(imgs, labels)

    # 3️⃣ Combine both datasets
    combined = ConcatDataset([real_sub, diff_subset])

    # 4️⃣ Build DataLoader
    loader = DataLoader(combined, batch_size=batch_size, shuffle=train,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    print(f"[Diffusion Loader] Real: {len(real_sub)} | Diffusion: {len(diff_subset)} | Total: {len(combined)}")
    return loader
