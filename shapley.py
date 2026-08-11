import itertools
import math
from model import SimpleCNN
from federated import run_federated_training
from train_utils import set_seed, evaluate


def get_all_subsets(client_names):
    """
    Every possible subset of clients, including the empty set.
    For 3 clients ["A","B","C"], this returns:
    (), ("A",), ("B",), ("C",), ("A","B"), ("A","C"), ("B","C"), ("A","B","C")
    -> 8 subsets total (2^3)
    """
    subsets = []
    for size in range(len(client_names) + 1):
        for combo in itertools.combinations(client_names, size):
            subsets.append(combo)
    return subsets


def get_accuracy_for_subset(subset, client_datasets, device, test_loader,
                             num_rounds=5, local_epochs=1, seed=42):
    """
    Trains a federated model using ONLY the clients named in `subset`.
    Empty subset = untrained model = chance-level accuracy (~10% for MNIST).
    """
    set_seed(seed)
    model = SimpleCNN().to(device)

    if len(subset) == 0:
        return evaluate(model, test_loader, device)

    subset_datasets = {name: client_datasets[name] for name in subset}
    trained_model = run_federated_training(
        subset_datasets, model, device, num_rounds=num_rounds, local_epochs=local_epochs
    )
    return evaluate(trained_model, test_loader, device)


def compute_exact_shapley(client_datasets, device, test_loader, num_rounds=5, local_epochs=1):
    """
    Computes the exact Shapley value for every client.
    """
    client_names = list(client_datasets.keys())
    n = len(client_names)

    # Step 1: train + evaluate every possible subset ONCE, cache the result
    all_subsets = get_all_subsets(client_names)
    subset_accuracy = {}
    for subset in all_subsets:
        acc = get_accuracy_for_subset(subset, client_datasets, device, test_loader, num_rounds, local_epochs)
        subset_accuracy[subset] = acc
        print(f"  Subset {subset if subset else '(empty)'}: accuracy = {acc:.4f}")

    # Step 2: for each client, average their marginal contribution across every team they could join
    shapley_values = {}
    for client in client_names:
        others = [c for c in client_names if c != client]
        total = 0.0

        for size in range(len(others) + 1):
            for combo in itertools.combinations(others, size):
                combo_with_client = tuple(sorted(combo + (client,)))
                combo_without_client = tuple(sorted(combo))

                acc_with = subset_accuracy[combo_with_client]
                acc_without = subset_accuracy[combo_without_client]
                marginal = acc_with - acc_without

                # Shapley's weighting: how many arrival-orders produce this exact team-size scenario
                weight = (math.factorial(size) * math.factorial(n - size - 1)) / math.factorial(n)
                total += weight * marginal

        shapley_values[client] = total

    return shapley_values

from federated import run_federated_training_with_checkpoints

def get_accuracy_for_subset_all_rounds(subset, client_datasets, device, test_loader,
                                        max_rounds=5, local_epochs=1, seed=42):
    """
    Trains a subset continuously through max_rounds, checkpointing accuracy
    after every round. Returns {round_num: accuracy}.
    """
    set_seed(seed)
    model = SimpleCNN().to(device)

    if len(subset) == 0:
        # untrained model — same accuracy at every "round" since nothing trains
        acc = evaluate(model, test_loader, device)
        return {r: acc for r in range(1, max_rounds + 1)}

    subset_datasets = {name: client_datasets[name] for name in subset}
    return run_federated_training_with_checkpoints(
        subset_datasets, model, device, max_rounds=max_rounds,
        local_epochs=local_epochs, test_loader=test_loader
    )


def compute_exact_shapley_all_rounds(client_datasets, device, test_loader,
                                       max_rounds=5, local_epochs=1):
    """
    Computes exact Shapley values at EVERY round (1 through max_rounds),
    training each subset only ONCE (continuously), not restarting per round.

    Returns: {round_num: {"client_1": shapley_value, ...}, ...}
    """
    client_names = list(client_datasets.keys())
    n = len(client_names)

    # Step 1: train every subset ONCE, continuously, get accuracy at every round
    all_subsets = get_all_subsets(client_names)
    subset_accuracy_by_round = {}
    for subset in all_subsets:
        acc_by_round = get_accuracy_for_subset_all_rounds(
            subset, client_datasets, device, test_loader, max_rounds, local_epochs
        )
        subset_accuracy_by_round[subset] = acc_by_round
        print(f"  Subset {subset if subset else '(empty)'}: {acc_by_round}")

    # Step 2: compute Shapley values separately for each round, using that round's accuracies
    shapley_by_round = {}
    for round_num in range(1, max_rounds + 1):
        shapley_values = {}
        for client in client_names:
            others = [c for c in client_names if c != client]
            total = 0.0
            for size in range(len(others) + 1):
                for combo in itertools.combinations(others, size):
                    combo_with = tuple(sorted(combo + (client,)))
                    combo_without = tuple(sorted(combo))

                    acc_with = subset_accuracy_by_round[combo_with][round_num]
                    acc_without = subset_accuracy_by_round[combo_without][round_num]
                    marginal = acc_with - acc_without

                    weight = (math.factorial(size) * math.factorial(n - size - 1)) / math.factorial(n)
                    total += weight * marginal
            shapley_values[client] = total
        shapley_by_round[round_num] = shapley_values

    return shapley_by_round