import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torchvision
import torchvision.transforms as T
from typing import List, Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# =====================================================
# 1️⃣ Diffusion Dataset
# =====================================================
class DiffusionAugmentedDataset(Dataset):
    def __init__(self, npz_path: str, transform=None):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"[Error] Diffusion data file not found: {npz_path}")
        data = np.load(npz_path)

        # Auto-detect field names
        keys = list(data.keys())
        if "images" in keys and "labels" in keys:
            imgs, labels = data["images"], data["labels"]
        elif "image" in keys and "label" in keys:
            imgs, labels = data["image"], data["label"]
        elif "arr_0" in keys and "arr_1" in keys:
            imgs, labels = data["arr_0"], data["arr_1"]
        else:
            raise KeyError(f"Unexpected keys in diffusion npz: {keys}")

        self.images = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        label = self.labels[idx]

        # Convert numpy to tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if isinstance(label, (np.int64, int)):
            label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# =====================================================
# 2️⃣ CIFAR-10 Loader with label filtering
# =====================================================
def _build_cifar10(data_dir: str, train: bool, num_workers=4, batch_size=128):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform_train if train else transform_test
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
    return ds, loader


def _filter_indices(ds: torchvision.datasets.CIFAR10, keep: List[int]):
    idx = [i for i, (_, y) in enumerate(ds) if y in keep]
    remap = {old: new for new, old in enumerate(keep)}
    return idx, remap


class RemappedSubset(Dataset):
    def __init__(self, base, indices, remap):
        self.base = base
        self.indices = indices
        self.remap = remap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, self.remap[int(y)]


# =====================================================
# 3️⃣ Build Diffusion-Augmented Loader
# =====================================================
def build_diffusion_augmented_loader(
    data_dir: str,
    diff_dir: str,
    keep_labels: List[int],
    batch_size: int,
    train: bool,
    num_workers: int = 4,
):
    """Combine real CIFAR subset and diffusion-generated images"""

    # --- Real CIFAR Subset ---
    ds_real, _ = _build_cifar10(data_dir, train=train, num_workers=num_workers, batch_size=batch_size)
    indices, remap = _filter_indices(ds_real, keep_labels)
    sub_real = RemappedSubset(ds_real, indices, remap)

    # --- Diffusion Data ---
    npz_path = os.path.join(diff_dir, "1M.npz")
    print(f"[Loading diffusion data from {npz_path}]")
    diff_ds = DiffusionAugmentedDataset(npz_path)

    # --- Balance ratio ---
    real_len = len(sub_real)
    diff_len = len(diff_ds)
    if train:
        sample_ratio = min(20, diff_len // real_len)  # e.g., 600k : 30k
        diff_indices = np.random.choice(diff_len, real_len * sample_ratio, replace=False)
        diff_ds = Subset(diff_ds, diff_indices)
    else:
        diff_indices = np.random.choice(diff_len, min(6000, diff_len), replace=False)
        diff_ds = Subset(diff_ds, diff_indices)

    merged = ConcatDataset([sub_real, diff_ds])
    print(f"[Diffusion Loader] Real: {len(sub_real)} | Diffusion: {len(diff_ds)} | Total: {len(merged)}")

    # --- Safe collate_fn (to prevent 'int' error) ---
    def safe_collate(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack([torch.as_tensor(i, dtype=torch.float32) for i in imgs])
        labels = torch.tensor(labels, dtype=torch.long)
        return imgs, labels

    loader = DataLoader(merged, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True, collate_fn=safe_collate)
    return loader
