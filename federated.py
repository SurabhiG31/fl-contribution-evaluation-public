import copy
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from train_utils import train_one_epoch, evaluate
import train_utils

def train_local_model(global_model, client_dataset, device, local_epochs=1, batch_size=64, lr=1e-3):
    """One client trains its own copy of the global model on its own data."""
    local_model = copy.deepcopy(global_model)   # each client gets an independent copy
    local_model.to(device)

    loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(local_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        train_one_epoch(local_model, loader, optimizer, criterion, device)

    return local_model

def federated_average(local_models, client_sizes):
    """Combine several clients' weights into one, weighted by how much data each had."""
    total_size = sum(client_sizes)
    new_state_dict = copy.deepcopy(local_models[0].state_dict())

    for key in new_state_dict:
        new_state_dict[key] = sum(
            local_models[i].state_dict()[key].float() * (client_sizes[i] / total_size)
            for i in range(len(local_models))
        )
    return new_state_dict

def run_federated_training(client_datasets_dict, global_model, device,
                            num_rounds=3, local_epochs=1):
    """
    client_datasets_dict: e.g. {"client_1": dataset, "client_3": dataset}
    Only the clients present in this dict participate — this is how we'll
    later simulate "removing" a client for LOO/Shapley: just don't include them here.
    """
    train_utils.training_count += 1          
    client_names = list(client_datasets_dict.keys())

    for round_num in range(num_rounds):
        local_models = []
        client_sizes = []

        for name in client_names:
            local_model = train_local_model(global_model, client_datasets_dict[name], device, local_epochs)
            local_models.append(local_model)
            client_sizes.append(len(client_datasets_dict[name]))

        new_state_dict = federated_average(local_models, client_sizes)
        global_model.load_state_dict(new_state_dict)

    return global_model

def run_federated_training_with_checkpoints(client_datasets_dict, global_model, device,
                                              max_rounds=5, local_epochs=1, test_loader=None):
    """
    Trains continuously for max_rounds, recording test accuracy after
    EVERY round (not just at the end). Returns a dict: {round_num: accuracy}
    """
    train_utils.training_count += 1 
    client_names = list(client_datasets_dict.keys())
    accuracy_by_round = {}

    for round_num in range(1, max_rounds + 1):
        local_models = []
        client_sizes = []

        for name in client_names:
            local_model = train_local_model(global_model, client_datasets_dict[name], device, local_epochs)
            local_models.append(local_model)
            client_sizes.append(len(client_datasets_dict[name]))

        new_state_dict = federated_average(local_models, client_sizes)
        global_model.load_state_dict(new_state_dict)

        # checkpoint: measure accuracy right now, after this round
        acc = evaluate(global_model, test_loader, device)
        accuracy_by_round[round_num] = acc

    return accuracy_by_round