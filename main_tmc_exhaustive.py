import torch
import time
import csv
from torch.utils.data import DataLoader
from data import (
    load_mnist,
    partition_partial_overlap,
    partition_partial_overlap_3way,
    partition_noniid_overlap,
    partition_noniid_overlap_3way,
)
from tmc_shapley import compute_tmc_shapley_exhaustive

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
full_train, test_data = load_mnist()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

SIZE_EACH = 6000
overlap_levels = [0.0, 0.25, 0.50, 0.75, 1.0]

scenarios = {}
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

all_results = {}
results = []

for scenario_name, client_datasets in scenarios.items():
    print(f"\n===== TMC-Shapley (exhaustive): {scenario_name} =====")
    start = time.time()
    values = compute_tmc_shapley_exhaustive(client_datasets, device, test_loader, num_rounds=5, local_epochs=1)
    elapsed = time.time() - start
    all_results[scenario_name] = values
    results.append({"scenario": scenario_name, "runtime": elapsed})
    print(f"\nValues: {values}")
    print(f"Took {elapsed:.1f}s")

print("\n\n===== SUMMARY =====")
for name, values in all_results.items():
    print(f"{name}: {values}")

with open("tmc_exhaustive_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Scenario", "Runtime (seconds)"])
    for r in results:
        writer.writerow([r["scenario"], r["runtime"]])