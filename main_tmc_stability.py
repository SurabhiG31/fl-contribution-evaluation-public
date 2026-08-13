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
from tmc_shapley import compute_tmc_shapley_all_rounds, compute_tmc_shapley_exhaustive_all_rounds

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
full_train, test_data = load_mnist()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

SIZE_EACH = 6000
overlap_levels = [0.0, 0.25, 0.50, 0.75, 1.0]

# All 3-client scenarios — get BOTH exhaustive and random-sampling stability
three_client_scenarios = {}
for overlap in overlap_levels:
    three_client_scenarios[f"pairwise_overlap_{int(overlap*100)}"] = partition_partial_overlap(
        full_train, size_each=SIZE_EACH, overlap_fraction=overlap
    )
    three_client_scenarios[f"global_overlap_{int(overlap*100)}"] = partition_partial_overlap_3way(
        full_train, size_each=SIZE_EACH, overlap_fraction=overlap
    )
for overlap in overlap_levels:
    three_client_scenarios[f"noniid_pairwise_overlap_{int(overlap*100)}"] = partition_noniid_overlap(
        full_train, overlap_fraction=overlap, seed=42
    )
    three_client_scenarios[f"noniid_global_overlap_{int(overlap*100)}"] = partition_noniid_overlap_3way(
        full_train, overlap_fraction=overlap, seed=42
    )

# 4-client scenarios — random-sampling only (exhaustive infeasible at N=4)
four_client_scenarios = {
    "label_skew": partition_label_skew(full_train),
    "quantity_skew": partition_quantity_skew(full_train),
}

all_results = []

print("\n\n########## EXHAUSTIVE (3-client scenarios) ##########")
for scenario_name, client_datasets in three_client_scenarios.items():
    print(f"\n===== TMC-Shapley stability (EXHAUSTIVE (6 cases)): {scenario_name} =====")
    start = time.time()
    shapley_by_round = compute_tmc_shapley_exhaustive_all_rounds(
        client_datasets, device, test_loader, max_rounds=5, local_epochs=1
    )
    elapsed = time.time() - start
    for round_num, values in shapley_by_round.items():
        print(f"  Round {round_num}: {values}")
        for name, val in values.items():
            all_results.append({"scenario": scenario_name, "method": "exhaustive", "round": round_num, "client": name, "value": val})
    print(f"Took {elapsed:.1f}s")

print("\n\n########## RANDOM SAMPLING, 10 samples (3-client scenarios) ##########")
for scenario_name, client_datasets in three_client_scenarios.items():
    print(f"\n===== TMC-Shapley stability (random sampling): {scenario_name} =====")
    start = time.time()
    shapley_by_round = compute_tmc_shapley_all_rounds(
        client_datasets, device, test_loader, max_rounds=5, local_epochs=1, num_samples=10
    )
    elapsed = time.time() - start
    for round_num, values in shapley_by_round.items():
        print(f"  Round {round_num}: {values}")
        for name, val in values.items():
            all_results.append({"scenario": scenario_name, "method": "random_10", "round": round_num, "client": name, "value": val})
    print(f"Took {elapsed:.1f}s")

print("\n\n########## RANDOM SAMPLING, 10 samples (4-client scenarios) ##########")
for scenario_name, client_datasets in four_client_scenarios.items():
    print(f"\n===== TMC-Shapley stability (random sampling): {scenario_name} =====")
    start = time.time()
    shapley_by_round = compute_tmc_shapley_all_rounds(
        client_datasets, device, test_loader, max_rounds=5, local_epochs=1, num_samples=10
    )
    elapsed = time.time() - start
    for round_num, values in shapley_by_round.items():
        print(f"  Round {round_num}: {values}")
        for name, val in values.items():
            all_results.append({"scenario": scenario_name, "method": "random_10", "round": round_num, "client": name, "value": val})
    print(f"Took {elapsed:.1f}s")

with open("tmc_stability_full_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["scenario", "method", "round", "client", "value"])
    writer.writeheader()
    writer.writerows(all_results)

print("\n\nDone. All results saved to tmc_stability_full_results.csv")