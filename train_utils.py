import torch
import numpy as np
import random

# Counters for tracking compute cost
training_count = 0
evaluation_count = 0

def reset_counters():
    global training_count, evaluation_count
    training_count = 0
    evaluation_count = 0

def get_counters():
    return {"trainings": training_count, "evaluations": evaluation_count}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device):
    global evaluation_count
    evaluation_count += 1 
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total