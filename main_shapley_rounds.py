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
from shapley import compute_exact_shapley_all_rounds

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

all_results = {}
results = []

for scenario_name, client_datasets in scenarios.items():
    print(f"\n===== Exact Shapley (all rounds): {scenario_name} =====")
    start = time.time()
    shapley_by_round = compute_exact_shapley_all_rounds(
        client_datasets, device, test_loader, max_rounds=5, local_epochs=1
    )
    elapsed = time.time() - start

    all_results[scenario_name] = shapley_by_round
    results.append({"scenario": scenario_name, "runtime": elapsed, "num_clients": len(client_datasets)})

    print(f"\nShapley values by round ({scenario_name}):")
    for round_num, values in shapley_by_round.items():
        print(f"  Round {round_num}: {values}")
    print(f"Took {elapsed:.1f} seconds")

print("\n\n===== SUMMARY: Exact Shapley by round, across all scenarios =====")
for scenario_name, shapley_by_round in all_results.items():
    print(f"\n{scenario_name}:")
    client_names = shapley_by_round[1].keys()
    for name in client_names:
        scores = [shapley_by_round[r][name] for r in sorted(shapley_by_round.keys())]
        mean = sum(scores) / len(scores)
        print(f"  {name}: {[round(s,4) for s in scores]}  (mean={mean:.4f})")

print("\n===== RUNTIME SUMMARY =====")
for r in results:
    print(f"{r['scenario']} (N={r['num_clients']}): {r['runtime']:.2f} seconds")

with open("shapley_rounds_runtime_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Scenario", "Num Clients", "Runtime (seconds)"])
    for r in results:
        writer.writerow([r["scenario"], r["num_clients"], r["runtime"]])