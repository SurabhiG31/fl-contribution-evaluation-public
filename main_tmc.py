'''
import torch
import time
from torch.utils.data import DataLoader
from data import load_mnist, partition_label_skew, partition_partial_overlap_3way
from tmc_shapley import compute_tmc_shapley

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

full_train, test_data = load_mnist()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

scenarios = {
    "label_skew": partition_label_skew(full_train),
    "global_overlap_100": partition_partial_overlap_3way(full_train, size_each=6000, overlap_fraction=1.0),
}

for scenario_name, client_datasets in scenarios.items():
    print(f"\n===== TMC-Shapley: {scenario_name} =====")
    start = time.time()
    values = compute_tmc_shapley(
        client_datasets, device, test_loader,
        num_rounds=5, local_epochs=1, num_samples=10, truncation_threshold=0.001
    )
    elapsed = time.time() - start
    print(f"\nTMC-Shapley values ({scenario_name}):")
    for name, val in values.items():
        print(f"  {name}: {val:.4f}")
    print(f"Took {elapsed:.1f} seconds")
    '''

import torch
import time
from torch.utils.data import DataLoader
from data import (
    load_mnist,
    partition_label_skew,
    partition_partial_overlap_3way,
    partition_noniid_overlap,
)
from tmc_shapley import compute_tmc_shapley

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

full_train, test_data = load_mnist()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

scenarios = {
    "label_skew": partition_label_skew(full_train),
    "global_overlap_100": partition_partial_overlap_3way(full_train, size_each=6000, overlap_fraction=1.0),
    "noniid_pairwise_overlap_100": partition_noniid_overlap(full_train, overlap_fraction=1.0),
}

all_tmc_results = {}

for scenario_name, client_datasets in scenarios.items():
    print(f"\n===== TMC-Shapley: {scenario_name} =====")
    start = time.time()
    values = compute_tmc_shapley(
        client_datasets, device, test_loader,
        num_rounds=5, local_epochs=1, num_samples=10, truncation_threshold=0.001
    )
    elapsed = time.time() - start
    all_tmc_results[scenario_name] = values
    print(f"\nTMC-Shapley values ({scenario_name}):")
    for name, val in values.items():
        print(f"  {name}: {val:.4f}")
    print(f"Took {elapsed:.1f} seconds")

print("\n\n===== SUMMARY: TMC-Shapley across scenarios =====")
for scenario_name, values in all_tmc_results.items():
    print(f"\n{scenario_name}:")
    for name, val in values.items():
        print(f"  {name}: {val:.4f}")