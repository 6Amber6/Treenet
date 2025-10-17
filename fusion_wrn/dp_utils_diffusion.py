import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torchvision
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# =========================
# Helper: CIFAR loaders
# =========================
def _build_cifar10(data_dir: str, train: bool, num_workers=4, batch_size=128):
    tfm_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    tfm_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=True,
        transform=tfm_train if train else tfm_test
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train,
                        num_workers=num_workers, pin_memory=True)
    return ds, loader

def _filter_indices(ds: torchvision.datasets.CIFAR10, keep: List[int]):
    idx = [i for i, (_, y) in enumerate(ds) if y in keep]
    remap = {old: new for new, old in enumerate(keep)}  # e.g. {0:0,2:1,...}
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

# =========================
# Diffusion npz dataset
# =========================
class DiffusionNPZ(Dataset):
    """
    Load diffusion-generated CIFAR-10 from a .npz file and
    (1) filter to keep_labels, (2) remap labels to [0..K-1].
    """
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

        # ensure NHWC uint8 -> NCHW float32
        if imgs.ndim != 4 or imgs.shape[-1] != 3:
            raise ValueError(f"Expected images as (N,32,32,3), got {imgs.shape}")
        if imgs.dtype != np.uint8:
            imgs = np.clip(imgs, 0, 1) if imgs.dtype.kind == 'f' else imgs
            imgs = (imgs * 255.0).round().astype(np.uint8)

        labels = labels.astype(np.int64)

        # ---- filter to keep_labels & remap ----
        keep_set = set(keep_labels)
        mask = np.isin(labels, list(keep_set))
        imgs = imgs[mask]
        labels = labels[mask]
        if imgs.shape[0] == 0:
            raise ValueError("After filtering, diffusion dataset is empty. "
                             f"keep_labels={keep_labels}, original labels unique={np.unique(data[list(data.keys())[-1]])}")

        # remap e.g. {2:0,3:1,...} so labels in [0..K-1]
        labels = np.vectorize(remap.get)(labels)

        # store tensors; transforms will run in __getitem__
        self.images = imgs  # still numpy for speed; convert on the fly
        self.labels = labels
        # transforms consistent with CIFAR10
        if train:
            self.transform = T.Compose([
                T.ToTensor(),  # converts HWC uint8 -> CHW float32 [0,1]
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]                 # HWC uint8
        lab = int(self.labels[idx])            # already remapped to [0..K-1]
        # to tensor + augment + normalize
        img = self.transform(img)              # CHW float32 normalized
        lab = torch.tensor(lab, dtype=torch.long)
        return img, lab

# =========================
# Public builder
# =========================
def build_diffusion_augmented_loader(
    data_dir: str,
    diff_dir: str,
    keep_labels: List[int],
    batch_size: int,
    train: bool,
    num_workers: int = 4,
):
    """
    Returns DataLoader of (real CIFAR subset + filtered/remapped diffusion data)
    """
    # real CIFAR subset with remap
    ds_real, _ = _build_cifar10(data_dir, train=train, num_workers=num_workers, batch_size=batch_size)
    indices, remap = _filter_indices(ds_real, keep_labels)
    sub_real = RemappedSubset(ds_real, indices, remap)

    # diffusion subset filtered/remapped the SAME way
    npz_path = os.path.join(diff_dir, "1M.npz")
    print(f"[Loading diffusion data from {npz_path}]")
    ds_diff = DiffusionNPZ(npz_path, keep_labels=keep_labels, remap=remap, train=train)

    # sampling ratio (avoid using too much diffusion on test)
    if train:
        # keep diffusion up to ~ 20x the real subset for balance
        real_len = len(sub_real)
        diff_len = len(ds_diff)
        take = min(diff_len, real_len * 20) if real_len > 0 else diff_len
        if take < diff_len:
            idx = np.random.choice(diff_len, take, replace=False)
            ds_diff = Subset(ds_diff, idx)
    else:
        # eval: cap diffusion to 6k to speed up sanity checks
        diff_len = len(ds_diff)
        take = min(6000, diff_len)
        if take < diff_len:
            idx = np.random.choice(diff_len, take, replace=False)
            ds_diff = Subset(ds_diff, idx)

    merged = ConcatDataset([sub_real, ds_diff])
    print(f"[Diffusion Loader] Real: {len(sub_real)} | Diffusion: {len(ds_diff)} | Total: {len(merged)}")

    # safe collate: ensure labels are long tensors
    def safe_collate(batch):
        xs, ys = zip(*batch)
        xs = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in xs], dim=0)
        ys = torch.tensor(ys, dtype=torch.long)
        return xs, ys

    loader = DataLoader(
        merged, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=True, collate_fn=safe_collate
    )
    return loader
