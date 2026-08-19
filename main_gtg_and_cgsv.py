"""
Runs, per scenario:
  1. Exact Shapley (ground truth) -- retrains every subset, every round.
     NOTE: earlier runs of this (main_shapley_rounds.py) only printed these
     values to the terminal and never saved them to a CSV. This script fixes
     that gap -- exact_shapley_by_round_results.csv is the ground-truth file
     everything else (TMC comparison, GTG validation) should read from now.
  2. GTG-Shapley -- ONE real training trajectory, then reconstructs every
     subset's model via weighted delta-sums (no retraining).
  3. CGSV -- reuses GTG-Shapley's captured trajectory at zero extra training
     cost; cosine-similarity signal, NOT a Shapley value.
  4. Validation -- Spearman rank correlation + absolute error + L2 distance,
     GTG-Shapley vs exact Shapley, per round.

Also prints/saves training & evaluation COUNTS (Rajesh's cost-as-counts
convention) for exact Shapley vs GTG-Shapley side by side, so the savings
are visible in the same units as the rest of the report.
"""
import csv
import torch
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
from gtg_shapley import compute_gtg_shapley_all_rounds
from cgsv import compute_cgsv_all_rounds
from compare_utils import compare_to_ground_truth
import train_utils
import scipy

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

full_train, test_data = load_mnist()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

SIZE_EACH = 6000
overlap_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
MAX_ROUNDS = 5

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

exact_rows, gtg_rows, cgsv_rows, validation_rows, cost_rows = [], [], [], [], []

for scenario_name, client_datasets in scenarios.items():
    client_names = list(client_datasets.keys())
    print(f"\n===== {scenario_name} =====")

    train_utils.reset_counters()
    exact_by_round = compute_exact_shapley_all_rounds(
        client_datasets, device, test_loader, max_rounds=MAX_ROUNDS, local_epochs=1
    )
    exact_counters = train_utils.get_counters()
    print(f"  Exact Shapley cost: {exact_counters}")

    train_utils.reset_counters()
    gtg_by_round, round_data = compute_gtg_shapley_all_rounds(
        client_datasets, device, test_loader, max_rounds=MAX_ROUNDS, local_epochs=1
    )
    gtg_counters = train_utils.get_counters()
    print(f"  GTG-Shapley cost:   {gtg_counters}")

    cgsv_by_round = compute_cgsv_all_rounds(round_data)

    comparison = compare_to_ground_truth(exact_by_round, gtg_by_round, client_names)

    for r in range(1, MAX_ROUNDS + 1):
        for c in client_names:
            exact_rows.append({"scenario": scenario_name, "round": r, "client": c, "value": exact_by_round[r][c]})
            gtg_rows.append({"scenario": scenario_name, "round": r, "client": c, "value": gtg_by_round[r][c]})
            cgsv_rows.append({"scenario": scenario_name, "round": r, "client": c, "value": cgsv_by_round[r][c]})
        validation_rows.append({
            "scenario": scenario_name,
            "round": r,
            "spearman": comparison[r]["spearman"],
            "mean_abs_error": comparison[r]["mean_abs_error"],
            "l2_distance": comparison[r]["l2_distance"],
        })

    cost_rows.append({
        "scenario": scenario_name, "method": "Exact Shapley",
        "trainings": exact_counters["trainings"], "evaluations": exact_counters["evaluations"],
    })
    cost_rows.append({
        "scenario": scenario_name, "method": "GTG-Shapley",
        "trainings": gtg_counters["trainings"], "evaluations": gtg_counters["evaluations"],
    })

with open("exact_shapley_by_round_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["scenario", "round", "client", "value"])
    writer.writeheader()
    writer.writerows(exact_rows)

with open("gtg_shapley_by_round_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["scenario", "round", "client", "value"])
    writer.writeheader()
    writer.writerows(gtg_rows)

with open("cgsv_by_round_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["scenario", "round", "client", "value"])
    writer.writeheader()
    writer.writerows(cgsv_rows)

with open("gtg_validation_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["scenario", "round", "spearman", "mean_abs_error", "l2_distance"])
    writer.writeheader()
    writer.writerows(validation_rows)

with open("gtg_cost_comparison.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["scenario", "method", "trainings", "evaluations"])
    writer.writeheader()
    writer.writerows(cost_rows)

print("\nDone. Saved:")
print("  exact_shapley_by_round_results.csv  (ground truth -- also needed for Correction 3)")
print("  gtg_shapley_by_round_results.csv")
print("  cgsv_by_round_results.csv")
print("  gtg_validation_results.csv          (Spearman + abs error + L2, per round)")
print("  gtg_cost_comparison.csv             (trainings/evaluations, exact vs GTG)")
