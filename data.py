import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset

def load_mnist():
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return full_train, test_data

def train_val_split(full_train, val_size=6000, seed=42):
    train_size = len(full_train) - val_size
    train_data, val_data = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    return train_data, val_data

def get_indices_for_digits(dataset, digits):
    mask = np.isin(dataset.targets.numpy(), digits)
    return np.where(mask)[0]

def partition_clients(full_train, client_digit_map):
    """
    client_digit_map: dict like {"client_1": [0,1,2], ...}
    Returns: dict of {"client_1": Subset(...), ...}
    """
    client_datasets = {}
    for name, digits in client_digit_map.items():
        idx = get_indices_for_digits(full_train, digits)
        client_datasets[name] = Subset(full_train, idx)
    return client_datasets

def partition_label_skew(full_train):
    client_digit_map = {
        "client_1": [0, 1, 2],
        "client_2": [3, 4, 5],
        "client_3": [6, 7],
        "client_4": [8, 9],
    }
    return partition_clients(full_train, client_digit_map)

def partition_quantity_skew(full_train, seed=42):
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(full_train))
    rng.shuffle(all_indices)

    sizes = {"client_1": 10000, "client_2": 5000, "client_3": 1000, "client_4": 300}
    client_datasets = {}
    start = 0
    for name, size in sizes.items():
        idx = all_indices[start:start+size]
        client_datasets[name] = Subset(full_train, idx)
        start += size
    return client_datasets

def partition_partial_overlap(full_train, size_each, overlap_fraction=0.5, seed=42):
    """
    2-of-3 overlap, fully IID base: client_1 and client_3 share `overlap_fraction`
    of their data from a common pool; client_2 is independent but drawn from the
    SAME distribution (all digits) as client_1/client_3 — label distribution is
    held constant, only overlap changes.
    """
    rng = np.random.default_rng(seed)
    base_idx = np.arange(len(full_train))
    rng.shuffle(base_idx)

    shared_size = int(size_each * overlap_fraction)
    unique_size = size_each - shared_size

    shared_pool = base_idx[0: shared_size]
    unique_1 = base_idx[shared_size: shared_size + unique_size]
    unique_3 = base_idx[shared_size + unique_size: shared_size + 2*unique_size]
    client_2_idx = base_idx[shared_size + 2*unique_size: shared_size + 2*unique_size + size_each]

    client_1_idx = np.concatenate([shared_pool, unique_1])
    client_3_idx = np.concatenate([shared_pool, unique_3])

    return {
        "client_1": Subset(full_train, client_1_idx),
        "client_2": Subset(full_train, client_2_idx),
        "client_3": Subset(full_train, client_3_idx),
    }


def partition_partial_overlap_3way(full_train, size_each, overlap_fraction=0.5, seed=42):
    """
    3-of-3 overlap, fully IID base — same base_idx construction as the 2-way
    version above, so the only difference between the two experiments is
    HOW MANY clients share the overlap pool, not label distribution.
    """
    rng = np.random.default_rng(seed)
    base_idx = np.arange(len(full_train))
    rng.shuffle(base_idx)

    shared_size = int(size_each * overlap_fraction)
    unique_size = size_each - shared_size

    shared_pool = base_idx[0: shared_size]
    unique_1 = base_idx[shared_size: shared_size + unique_size]
    unique_2 = base_idx[shared_size + unique_size: shared_size + 2*unique_size]
    unique_3 = base_idx[shared_size + 2*unique_size: shared_size + 3*unique_size]

    client_1_idx = np.concatenate([shared_pool, unique_1])
    client_2_idx = np.concatenate([shared_pool, unique_2])
    client_3_idx = np.concatenate([shared_pool, unique_3])

    return {
        "client_1": Subset(full_train, client_1_idx),
        "client_2": Subset(full_train, client_2_idx),
        "client_3": Subset(full_train, client_3_idx),
    }

def partition_noniid_overlap(full_train, overlap_fraction=0.5, seed=42):
    """
    2-of-3 NON-IID overlap: client_1 and client_3 are BOTH restricted to
    digits 0-4 only (a genuinely narrow, non-IID distribution), sharing
    `overlap_fraction` of that restricted pool with each other.
    client_2 holds digits 5-9 — independent, also non-IID, no overlap.
    """
    rng = np.random.default_rng(seed)
    base_idx = get_indices_for_digits(full_train, [0, 1, 2, 3, 4])
    rng.shuffle(base_idx)

    size_each = len(base_idx) // 2
    shared_size = int(size_each * overlap_fraction)
    unique_size = size_each - shared_size

    shared_pool = base_idx[0: shared_size]
    unique_1 = base_idx[shared_size: shared_size + unique_size]
    unique_3 = base_idx[shared_size + unique_size: shared_size + 2*unique_size]

    client_1_idx = np.concatenate([shared_pool, unique_1])
    client_3_idx = np.concatenate([shared_pool, unique_3])
    client_2_idx = get_indices_for_digits(full_train, [5, 6, 7, 8, 9])

    return {
        "client_1": Subset(full_train, client_1_idx),
        "client_2": Subset(full_train, client_2_idx),
        "client_3": Subset(full_train, client_3_idx),
    }


def partition_noniid_overlap_3way(full_train, overlap_fraction=0.5, seed=42):
    """
    3-of-3 NON-IID overlap: ALL THREE clients are restricted to digits 0-4
    only, sharing `overlap_fraction` of that pool across all three.
    """
    rng = np.random.default_rng(seed)
    base_idx = get_indices_for_digits(full_train, [0, 1, 2, 3, 4])
    rng.shuffle(base_idx)

    size_each = len(base_idx) // 3
    shared_size = int(size_each * overlap_fraction)
    unique_size = size_each - shared_size

    shared_pool = base_idx[0: shared_size]
    unique_1 = base_idx[shared_size: shared_size + unique_size]
    unique_2 = base_idx[shared_size + unique_size: shared_size + 2*unique_size]
    unique_3 = base_idx[shared_size + 2*unique_size: shared_size + 3*unique_size]

    client_1_idx = np.concatenate([shared_pool, unique_1])
    client_2_idx = np.concatenate([shared_pool, unique_2])
    client_3_idx = np.concatenate([shared_pool, unique_3])

    return {
        "client_1": Subset(full_train, client_1_idx),
        "client_2": Subset(full_train, client_2_idx),
        "client_3": Subset(full_train, client_3_idx),
    }

def partition_generic_iid(full_train, n_clients, size_each=2000, seed=42):
    """
    Creates n_clients IID clients, each with size_each images,
    randomly sampled from the full dataset (no overlap, no skew) —
    purely for testing how cost scales with N.
    """
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(full_train))
    rng.shuffle(all_idx)

    client_datasets = {}
    start = 0
    for i in range(1, n_clients + 1):
        idx = all_idx[start: start + size_each]
        client_datasets[f"client_{i}"] = Subset(full_train, idx)
        start += size_each
    return client_datasets