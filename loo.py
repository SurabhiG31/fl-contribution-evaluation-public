import copy
from model import SimpleCNN
from federated import run_federated_training
from train_utils import set_seed, evaluate
from torch.utils.data import DataLoader
from model import SimpleCNN
from federated import run_federated_training_with_checkpoints

def get_test_accuracy(client_subset_dict, device, test_loader,
                       num_rounds=3, local_epochs=1, seed=42):
    """
    Trains a fresh global model federated-style on whichever clients
    are in client_subset_dict, and returns its test accuracy.
    This is the one function every contribution method calls repeatedly.
    """
    set_seed(seed)  # same starting point every time — fair comparison
    global_model = SimpleCNN().to(device)
    trained_model = run_federated_training(
        client_subset_dict, global_model, device,
        num_rounds=num_rounds, local_epochs=local_epochs
    )
    return evaluate(trained_model, test_loader, device)

def compute_loo(client_datasets, device, test_loader, num_rounds=3, local_epochs=1):
    """
    client_datasets: dict of {"client_1": dataset, ...} — the FULL set of clients
    Returns: dict of {"client_1": contribution_score, ...}
    """
    client_names = list(client_datasets.keys())

    # Step 1: baseline — everyone included
    full_acc = get_test_accuracy(client_datasets, device, test_loader, num_rounds, local_epochs)
    print(f"Baseline (all clients) accuracy: {full_acc:.4f}")

    contributions = {}
    for name in client_names:
        # Step 2: everyone EXCEPT this one client
        subset = {k: v for k, v in client_datasets.items() if k != name}
        acc_without = get_test_accuracy(subset, device, test_loader, num_rounds, local_epochs)

        # Step 3: contribution = how much accuracy dropped without them
        contributions[name] = full_acc - acc_without
        print(f"Without {name}: accuracy = {acc_without:.4f}  ->  contribution = {contributions[name]:.4f}")

    return contributions

def compute_loo_across_rounds(
    client_datasets,
    device,
    test_loader,
    max_rounds=5,
    local_epochs=1,
):
    """
    Compute LOO contribution scores after every communication round count,
    from 1 up to max_rounds. Returns a history of scores per round count,
    so we can check whether scores stabilize as training runs longer.

    Returns
    -------
    history : dict
        {
            1: {"client_1": ..., "client_2": ...},
            2: {"client_1": ..., "client_2": ...},
            ...
        }
    """
    history = {}

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"Communication Round {round_num}")
        print(f"{'='*60}")

        scores = compute_loo(
            client_datasets=client_datasets,
            device=device,
            test_loader=test_loader,
            num_rounds=round_num,
            local_epochs=local_epochs,
        )

        history[round_num] = scores

    return history
'''
def compute_loo_continuous(client_datasets, device, test_loader, max_rounds=5, local_epochs=1, seed=42):
    """
    Trains the 'all clients' trajectory ONCE, and each 'all-but-one-client'
    trajectory ONCE, each checkpointed at every round. Returns:

    history : dict
        { round_num: {"client_1": score, "client_2": score, ...}, ... }
    """
    client_names = list(client_datasets.keys())

    # Full trajectory: everyone included, trained once, checkpointed every round
    set_seed(seed)
    full_model = SimpleCNN().to(device)
    full_acc_by_round = run_federated_training_with_checkpoints(
        client_datasets, full_model, device, max_rounds, local_epochs, test_loader
    )

    # One trajectory per leave-one-out client, each trained once, checkpointed every round
    loo_acc_by_round = {}
    for name in client_names:
        subset = {k: v for k, v in client_datasets.items() if k != name}
        set_seed(seed)  # same starting point as the full trajectory, for fair comparison
        model_without = SimpleCNN().to(device)
        loo_acc_by_round[name] = run_federated_training_with_checkpoints(
            subset, model_without, device, max_rounds, local_epochs, test_loader
        )

    # Derive contribution scores at every round: full_acc - acc_without, per round
    history = {}
    for round_num in range(1, max_rounds + 1):
        history[round_num] = {
            name: full_acc_by_round[round_num] - loo_acc_by_round[name][round_num]
            for name in client_names
        }

    return history
    '''

def compute_loo_continuous(client_datasets, device, test_loader, max_rounds=5, local_epochs=1, seed=42):
    client_names = list(client_datasets.keys())

    set_seed(seed)
    full_model = SimpleCNN().to(device)
    full_acc_by_round = run_federated_training_with_checkpoints(
        client_datasets, full_model, device, max_rounds, local_epochs, test_loader
    )

    loo_acc_by_round = {}
    for name in client_names:
        subset = {k: v for k, v in client_datasets.items() if k != name}
        set_seed(seed)
        model_without = SimpleCNN().to(device)
        loo_acc_by_round[name] = run_federated_training_with_checkpoints(
            subset, model_without, device, max_rounds, local_epochs, test_loader
        )

    contributions_by_round = {}
    for round_num in range(1, max_rounds + 1):
        contributions_by_round[round_num] = {
            name: full_acc_by_round[round_num] - loo_acc_by_round[name][round_num]
            for name in client_names
        }

    return {
        "full_accuracy": full_acc_by_round,        # {1: 0.65, 2: 0.71, ...}
        "loo_accuracy": loo_acc_by_round,           # {"client_1": {1: 0.60, 2: 0.68, ...}, ...}
        "contributions": contributions_by_round,    # {1: {"client_1": 0.05, ...}, ...}
    }