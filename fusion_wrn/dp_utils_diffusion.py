import os
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torchvision
import torchvision.transforms as T
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# =====================================================
# CIFAR10 subset + remap
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
        root=data_dir, train=train, download=True,
        transform=transform_train if train else transform_test
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train,
                        num_workers=num_workers, pin_memory=True)
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
# Diffusion dataset with filter + remap
# =====================================================
class DiffusionNPZ(Dataset):
    def __init__(self, npz_path: str, keep_labels: List[int], remap: dict, train: bool):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"[Error] Diffusion data file not found: {npz_path}")

        data = np.load(npz_path)
        keys = list(data.keys())
        if "images" in keys and "labels" in keys:
            imgs, labels = data["images"], data["labels"]
        elif "image" in keys and "label" in keys:
            imgs, labels = data["image"], data["label"]
        elif "arr_0" in keys and "arr_1" in keys:
            imgs, labels = data["arr_0"], data["arr_1"]
        else:
            raise KeyError(f"Unexpected keys in diffusion npz: {keys}")

        # format check
        if imgs.ndim != 4 or imgs.shape[-1] != 3:
            raise ValueError(f"Expected images as (N,32,32,3), got {imgs.shape}")

        if imgs.dtype != np.uint8:
            imgs = np.clip(imgs, 0, 1) if imgs.dtype.kind == 'f' else imgs
            imgs = (imgs * 255.0).round().astype(np.uint8)

        labels = labels.astype(np.int64)

        # filter + remap
        keep_set = set(keep_labels)
        mask = np.isin(labels, list(keep_set))
        imgs = imgs[mask]
        labels = labels[mask]
        if imgs.shape[0] == 0:
            raise ValueError("After filtering, diffusion dataset is empty.")

        labels = np.vectorize(remap.get)(labels)

        self.images = imgs
        self.labels = labels
        self.transform = (
            T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ])
            if train else
            T.Compose([
                T.ToTensor(),
                T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ])
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        lab = int(self.labels[idx])

        # ✅ ensure image is PIL before transforms
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        img = self.transform(img)
        return img, torch.tensor(lab, dtype=torch.long)


# =====================================================
# Combine CIFAR10 + Diffusion (fraction controlled)
# =====================================================
def build_diffusion_augmented_loader(
    data_dir: str,
    diff_dir: str,
    keep_labels: List[int],
    batch_size: int,
    train: bool,
    num_workers: int = 4,
    diff_fraction: float = 0.7,  # ← 控制扩散数据比例
):
    # --- real CIFAR subset ---
    ds_real, _ = _build_cifar10(data_dir, train=train,
                                num_workers=num_workers, batch_size=batch_size)
    indices, remap = _filter_indices(ds_real, keep_labels)
    sub_real = RemappedSubset(ds_real, indices, remap)

    # --- diffusion subset ---
    npz_path = os.path.join(diff_dir, "1M.npz")
    print(f"[Loading diffusion data from {npz_path}]")
    ds_diff = DiffusionNPZ(npz_path, keep_labels=keep_labels, remap=remap, train=train)

    real_len = len(sub_real)
    diff_len = len(ds_diff)

    if train:
        # 控制比例：diffusion:real ≈ diff_fraction:(1 - diff_fraction)
        take = int(real_len * diff_fraction / (1 - diff_fraction))
        take = min(take, diff_len)
        idx = np.random.choice(diff_len, take, replace=False)
        ds_diff = Subset(ds_diff, idx)
    else:
        # 测试集只取较少 diffusion 样本
        take = min(6000, diff_len)
        idx = np.random.choice(diff_len, take, replace=False)
        ds_diff = Subset(ds_diff, idx)

    merged = ConcatDataset([sub_real, ds_diff])
    print(f"[Diffusion Loader] Real: {len(sub_real)} | Diffusion: {len(ds_diff)} | Total: {len(merged)}")

    def safe_collate(batch):
        xs, ys = zip(*batch)
        xs = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in xs])
        ys = torch.tensor(ys, dtype=torch.long)
        return xs, ys

    loader = DataLoader(
        merged, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=True, collate_fn=safe_collate
    )
    return loader
