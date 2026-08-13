import torch
import time
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
from tmc_shapley import compute_tmc_shapley

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
full_train, test_data = load_mnist()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

SIZE_EACH = 6000
overlap_levels = [0.0, 0.25, 0.50, 0.75, 1.0]

scenarios = {
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

all_results = {}
results = []

for scenario_name, client_datasets in scenarios.items():
    print(f"\n===== TMC-Shapley: {scenario_name} =====")
    start = time.time()
    values = compute_tmc_shapley(
        client_datasets, device, test_loader,
        num_rounds=5, local_epochs=1, num_samples=10, truncation_threshold=0.001
    )
    elapsed = time.time() - start
    all_results[scenario_name] = values
    results.append({"scenario": scenario_name, "runtime": elapsed, "num_clients": len(client_datasets)})
    print(f"\nTMC-Shapley values ({scenario_name}): {values}")
    print(f"Took {elapsed:.1f} seconds")

print("\n\n===== SUMMARY =====")
for scenario_name, values in all_results.items():
    print(f"\n{scenario_name}: {values}")

print("\n===== RUNTIME SUMMARY =====")
for r in results:
    print(f"{r['scenario']} (N={r['num_clients']}): {r['runtime']:.2f}s")

with open("tmc_full_runtime_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Scenario", "Num Clients", "Runtime (seconds)"])
    for r in results:
        writer.writerow([r["scenario"], r["num_clients"], r["runtime"]])