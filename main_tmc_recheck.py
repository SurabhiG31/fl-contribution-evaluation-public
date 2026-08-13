import torch
import time
from torch.utils.data import DataLoader
from data import load_mnist, partition_partial_overlap_3way
from tmc_shapley import compute_tmc_shapley

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

full_train, test_data = load_mnist()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

client_datasets = partition_partial_overlap_3way(full_train, size_each=6000, overlap_fraction=1.0)

print("\n===== TMC-Shapley (30 samples): global_overlap_100 =====")
start = time.time()
values = compute_tmc_shapley(
    client_datasets, device, test_loader,
    num_rounds=5, local_epochs=1, num_samples=30, truncation_threshold=0.001
)
elapsed = time.time() - start

print(f"\nTMC-Shapley values (30 samples, global_overlap_100):")
for name, val in values.items():
    print(f"  {name}: {val:.4f}")
print(f"Took {elapsed:.1f} seconds")

print(f"\nFor comparison — exact Shapley gave: client_1=0.2899, client_2=0.2899, client_3=0.2899")
print(f"For comparison — TMC-Shapley at 10 samples gave: client_1=0.0878, client_2=0.6947, client_3=0.0892")