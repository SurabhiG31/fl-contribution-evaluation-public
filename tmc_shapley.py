import random
from model import SimpleCNN
from federated import run_federated_training
from train_utils import set_seed, evaluate
from federated import run_federated_training_with_checkpoints
import itertools

def compute_tmc_shapley(client_datasets, device, test_loader,
                         num_rounds=5, local_epochs=1,
                         num_samples=20, truncation_threshold=0.001, seed=42):
    """
    Approximates Shapley values via random permutation sampling.

    num_samples: how many random arrival-orders to average over (the main
                 cost/accuracy knob — more samples = closer to true Shapley,
                 more compute).
    truncation_threshold: if adding a client changes accuracy by less than
                 this, treat the REMAINING clients in this permutation as
                 contributing ~0 and skip training them for this round,
                 to save compute.
    """
    client_names = list(client_datasets.keys())
    n = len(client_names)

    running_totals = {name: 0.0 for name in client_names}
    counts = {name: 0 for name in client_names}

    rng = random.Random(seed)

    # Accuracy of an untrained model (empty coalition) — same for every permutation
    set_seed(seed)
    empty_model = SimpleCNN().to(device)
    empty_acc = evaluate(empty_model, test_loader, device)

    for sample_num in range(num_samples):
        permutation = client_names.copy()
        rng.shuffle(permutation)

        print(f"\n  Sample {sample_num+1}/{num_samples}: order = {permutation}")

        current_subset = []
        prev_acc = empty_acc
        truncated = False

        for client in permutation:
            if truncated:
                # remaining clients in this permutation contribute ~0; skip training
                running_totals[client] += 0.0
                counts[client] += 1
                continue

            current_subset.append(client)
            subset_datasets = {name: client_datasets[name] for name in current_subset}

            set_seed(seed + sample_num)  # vary seed per sample, but reproducible
            model = SimpleCNN().to(device)
            trained_model = run_federated_training(
                subset_datasets, model, device, num_rounds=num_rounds, local_epochs=local_epochs
            )
            current_acc = evaluate(trained_model, test_loader, device)

            marginal = current_acc - prev_acc
            running_totals[client] += marginal
            counts[client] += 1

            print(f"    +{client}: accuracy {prev_acc:.4f} -> {current_acc:.4f}  (marginal = {marginal:.4f})")

            if abs(marginal) < truncation_threshold:
                truncated = True
                print(f"    -> truncating remaining clients in this sample (marginal below threshold)")

            prev_acc = current_acc

    shapley_estimates = {
        name: running_totals[name] / counts[name] for name in client_names
    }
    return shapley_estimates

def compute_tmc_shapley_all_rounds(client_datasets, device, test_loader,
                                     max_rounds=5, local_epochs=1,
                                     num_samples=10, truncation_threshold=0.001, seed=42):
    """
    TMC-Shapley with checkpointing: for each sampled permutation, train
    continuously through max_rounds, recording marginal contributions at
    EVERY round checkpoint, not just the final round.
    Returns: {round_num: {"client_1": score, ...}, ...}
    """
    client_names = list(client_datasets.keys())
    rng = random.Random(seed)

    running_totals_by_round = {r: {name: 0.0 for name in client_names} for r in range(1, max_rounds+1)}
    counts_by_round = {r: {name: 0 for name in client_names} for r in range(1, max_rounds+1)}

    set_seed(seed)
    empty_model = SimpleCNN().to(device)
    empty_acc = evaluate(empty_model, test_loader, device)
    empty_acc_by_round = {r: empty_acc for r in range(1, max_rounds+1)}

    for sample_num in range(num_samples):
        permutation = client_names.copy()
        rng.shuffle(permutation)
        print(f"\n  Sample {sample_num+1}/{num_samples}: order = {permutation}")

        current_subset = []
        prev_acc_by_round = empty_acc_by_round
        truncated = False

        for client in permutation:
            if truncated:
                for r in range(1, max_rounds+1):
                    running_totals_by_round[r][client] += 0.0
                    counts_by_round[r][client] += 1
                continue

            current_subset.append(client)
            subset_datasets = {name: client_datasets[name] for name in current_subset}

            set_seed(seed + sample_num)
            model = SimpleCNN().to(device)
            acc_by_round = run_federated_training_with_checkpoints(
                subset_datasets, model, device, max_rounds=max_rounds,
                local_epochs=local_epochs, test_loader=test_loader
            )

            for r in range(1, max_rounds+1):
                marginal = acc_by_round[r] - prev_acc_by_round[r]
                running_totals_by_round[r][client] += marginal
                counts_by_round[r][client] += 1

            final_round_marginal = acc_by_round[max_rounds] - prev_acc_by_round[max_rounds]
            print(f"    +{client}: final-round accuracy {prev_acc_by_round[max_rounds]:.4f} -> {acc_by_round[max_rounds]:.4f}  (marginal = {final_round_marginal:.4f})")

            if abs(final_round_marginal) < truncation_threshold:
                truncated = True
                print(f"    -> truncating remaining clients in this sample")

            prev_acc_by_round = acc_by_round

    shapley_by_round = {}
    for r in range(1, max_rounds+1):
        shapley_by_round[r] = {
            name: running_totals_by_round[r][name] / counts_by_round[r][name]
            for name in client_names
        }
    return shapley_by_round

import itertools

def compute_tmc_shapley_exhaustive(client_datasets, device, test_loader,
                                     num_rounds=5, local_epochs=1,
                                     truncation_threshold=0.001, seed=42):
    """
    Tries EVERY unique permutation exactly once (only feasible for small N).
    For N=3, this means all 6 possible lineups.
    """
    client_names = list(client_datasets.keys())
    all_permutations = list(itertools.permutations(client_names))
    n_permutations = len(all_permutations)
    print(f"Running {n_permutations} unique permutations (N={len(client_names)})")

    running_totals = {name: 0.0 for name in client_names}
    counts = {name: 0 for name in client_names}

    set_seed(seed)
    empty_model = SimpleCNN().to(device)
    empty_acc = evaluate(empty_model, test_loader, device)

    for sample_num, permutation in enumerate(all_permutations):
        print(f"\n  Permutation {sample_num+1}/{n_permutations}: order = {list(permutation)}")
        current_subset = []
        prev_acc = empty_acc
        truncated = False

        for client in permutation:
            if truncated:
                running_totals[client] += 0.0
                counts[client] += 1
                continue

            current_subset.append(client)
            subset_datasets = {name: client_datasets[name] for name in current_subset}

            set_seed(seed + sample_num)
            model = SimpleCNN().to(device)
            trained_model = run_federated_training(
                subset_datasets, model, device, num_rounds=num_rounds, local_epochs=local_epochs
            )
            current_acc = evaluate(trained_model, test_loader, device)

            marginal = current_acc - prev_acc
            running_totals[client] += marginal
            counts[client] += 1
            print(f"    +{client}: {prev_acc:.4f} -> {current_acc:.4f} (marginal = {marginal:.4f})")

            if abs(marginal) < truncation_threshold:
                truncated = True
                print(f"    -> truncating remaining clients")
            prev_acc = current_acc

    return {name: running_totals[name] / counts[name] for name in client_names}

def compute_tmc_shapley_exhaustive_all_rounds(client_datasets, device, test_loader,
                                                max_rounds=5, local_epochs=1, seed=42):
    """
    Exhaustive-permutation TMC-Shapley, checkpointed at every round.
    Only feasible for small N (uses all N! permutations).
    """
    client_names = list(client_datasets.keys())
    all_permutations = list(itertools.permutations(client_names))
    n_permutations = len(all_permutations)

    running_totals_by_round = {r: {name: 0.0 for name in client_names} for r in range(1, max_rounds+1)}
    counts_by_round = {r: {name: 0 for name in client_names} for r in range(1, max_rounds+1)}

    set_seed(seed)
    empty_model = SimpleCNN().to(device)
    empty_acc = evaluate(empty_model, test_loader, device)
    empty_acc_by_round = {r: empty_acc for r in range(1, max_rounds+1)}

    for sample_num, permutation in enumerate(all_permutations):
        print(f"\n  Permutation {sample_num+1}/{n_permutations}: order = {list(permutation)}")
        current_subset = []
        prev_acc_by_round = empty_acc_by_round
        truncated = False

        for client in permutation:
            if truncated:
                for r in range(1, max_rounds+1):
                    running_totals_by_round[r][client] += 0.0
                    counts_by_round[r][client] += 1
                continue

            current_subset.append(client)
            subset_datasets = {name: client_datasets[name] for name in current_subset}

            set_seed(seed + sample_num)
            model = SimpleCNN().to(device)
            acc_by_round = run_federated_training_with_checkpoints(
                subset_datasets, model, device, max_rounds=max_rounds,
                local_epochs=local_epochs, test_loader=test_loader
            )

            for r in range(1, max_rounds+1):
                marginal = acc_by_round[r] - prev_acc_by_round[r]
                running_totals_by_round[r][client] += marginal
                counts_by_round[r][client] += 1

            final_marginal = acc_by_round[max_rounds] - prev_acc_by_round[max_rounds]
            print(f"    +{client}: final-round {prev_acc_by_round[max_rounds]:.4f} -> {acc_by_round[max_rounds]:.4f} (marginal={final_marginal:.4f})")

            if abs(final_marginal) < 0.001:
                truncated = True
                print(f"    -> truncating remaining clients")
            prev_acc_by_round = acc_by_round

    shapley_by_round = {}
    for r in range(1, max_rounds+1):
        shapley_by_round[r] = {
            name: running_totals_by_round[r][name] / counts_by_round[r][name]
            for name in client_names
        }
    return shapley_by_round