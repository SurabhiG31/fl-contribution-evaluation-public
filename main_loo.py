import torch
import numpy as np
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
    partition_noniid_overlap_3way
)
from loo import compute_loo_continuous


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

all_histories = {}
results = []

for scenario_name, client_datasets in scenarios.items():
    print(f"\n===== Scenario: {scenario_name} =====")
    start = time.time()
    history = compute_loo_continuous(
        client_datasets, device, test_loader, max_rounds=5, local_epochs=1
    )
    elapsed = time.time() - start

    all_histories[scenario_name] = history
    results.append({"scenario": scenario_name, "runtime": elapsed})
    print(f"Scenario '{scenario_name}' took {elapsed:.1f} seconds")

# ---- Runs ONCE, after ALL scenarios are done ----
print("\n\n===== DETAILED RESULTS =====")
for scenario_name, result in all_histories.items():
    print(f"\n{scenario_name}:")
    print(f"  Full accuracy by round: {result['full_accuracy']}")
    for name in result['loo_accuracy']:
        print(f"  Without {name} by round: {result['loo_accuracy'][name]}")

    print(f"  --- Contribution stability ---")
    for name in result['contributions'][1].keys():
        scores = [
            result['contributions'][r][name]
            for r in sorted(result['contributions'].keys())
        ]
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"  {name} contribution across rounds: {[round(s,4) for s in scores]}")
        print(f"     Mean = {mean:.4f} ± {std:.4f}")

print("\n===== RUNTIME SUMMARY =====")
for result in results:
    print(f"{result['scenario']}: {result['runtime']:.2f} seconds")

with open("runtime_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Scenario", "Runtime (seconds)"])
    for result in results:
        writer.writerow([result["scenario"], result["runtime"]])