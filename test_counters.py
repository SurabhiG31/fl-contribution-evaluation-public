'''
import torch
from torch.utils.data import DataLoader
from data import load_mnist, partition_partial_overlap_3way
from train_utils import reset_counters, get_counters, set_seed
from loo import compute_loo_continuous

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
full_train, test_data = load_mnist()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

client_datasets = partition_partial_overlap_3way(full_train, size_each=6000, overlap_fraction=1.0)

reset_counters()
history = compute_loo_continuous(client_datasets, device, test_loader, max_rounds=5, local_epochs=1)
print(get_counters())
'''

import torch
import csv
from torch.utils.data import DataLoader
from data import (
    load_mnist,
    partition_label_skew,
    partition_quantity_skew,
    partition_partial_overlap,
    partition_partial_overlap_3way,
    partition_noniid_overlap,
    partition_noniid_overlap_3way,
)
from train_utils import reset_counters, get_counters
from loo import compute_loo_continuous
from shapley import compute_exact_shapley, compute_exact_shapley_all_rounds
from tmc_shapley import compute_tmc_shapley, compute_tmc_shapley_exhaustive

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
full_train, test_data = load_mnist()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

SIZE_EACH = 6000
overlap_levels = [0.0, 0.25, 0.50, 0.75, 1.0]

scenarios = {
    "label_skew": partition_label_skew(full_train),
    "quantity_skew": partition_quantity_skew(full_train),
}
for overlap in overlap_levels:
    scenarios[f"pairwise_overlap_{int(overlap*100)}"] = partition_partial_overlap(
        full_train, size_each=SIZE_EACH, overlap_fraction=overlap
    )
    scenarios[f"global_overlap_{int(overlap*100)}"] = partition_partial_overlap_3way(
        full_train, size_each=SIZE_EACH, overlap_fraction=overlap
    )
for overlap in overlap_levels:
    scenarios[f"noniid_pairwise_overlap_{int(overlap*100)}"] = partition_noniid_overlap(
        full_train, overlap_fraction=overlap, seed=42
    )
    scenarios[f"noniid_global_overlap_{int(overlap*100)}"] = partition_noniid_overlap_3way(
        full_train, overlap_fraction=overlap, seed=42
    )

results = []

for scenario_name, client_datasets in scenarios.items():
    n_clients = len(client_datasets)

    # LOO
    reset_counters()
    compute_loo_continuous(client_datasets, device, test_loader, max_rounds=5, local_epochs=1)
    loo_counts = get_counters()
    results.append({"scenario": scenario_name, "method": "LOO", **loo_counts})
    print(f"{scenario_name} - LOO: {loo_counts}")

    # Exact Shapley (only feasible for small N — skip label_skew/quantity_skew if too slow, or include if time allows)
    reset_counters()
    compute_exact_shapley_all_rounds(client_datasets, device, test_loader, max_rounds=5, local_epochs=1)
    shapley_counts = get_counters()
    results.append({"scenario": scenario_name, "method": "Exact Shapley", **shapley_counts})
    print(f"{scenario_name} - Exact Shapley: {shapley_counts}")

    # TMC-Shapley: exhaustive for N=3, random-10 for N=4
    if n_clients == 3:
        reset_counters()
        compute_tmc_shapley_exhaustive(client_datasets, device, test_loader, num_rounds=5, local_epochs=1)
        tmc_counts = get_counters()
        results.append({"scenario": scenario_name, "method": "TMC-Shapley (exhaustive)", **tmc_counts})
        print(f"{scenario_name} - TMC-Shapley (exhaustive): {tmc_counts}")
    else:
        reset_counters()
        compute_tmc_shapley(client_datasets, device, test_loader, num_rounds=5, local_epochs=1, num_samples=10)
        tmc_counts = get_counters()
        results.append({"scenario": scenario_name, "method": "TMC-Shapley (10 samples)", **tmc_counts})
        print(f"{scenario_name} - TMC-Shapley (10 samples): {tmc_counts}")

with open("cost_counts_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["scenario", "method", "trainings", "evaluations"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print("\nDone. Results saved to cost_counts_results.csv")