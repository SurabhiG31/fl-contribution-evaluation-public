import torch
import time
import csv
from torch.utils.data import DataLoader
from data import load_mnist, partition_generic_iid
from loo import compute_loo
from shapley import compute_exact_shapley
from tmc_shapley import compute_tmc_shapley

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

full_train, test_data = load_mnist()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

results = []

# LOO: cheap enough to test at larger N
loo_n_values = [3, 4, 5, 6, 8, 10]
for n in loo_n_values:
    client_datasets = partition_generic_iid(full_train, n_clients=n, size_each=2000)
    start = time.time()
    compute_loo(client_datasets, device, test_loader, num_rounds=3, local_epochs=1)
    elapsed = time.time() - start
    results.append({"method": "LOO", "n_clients": n, "runtime": elapsed})
    print(f"LOO, N={n}: {elapsed:.1f}s")

# Exact Shapley: only feasible at small N — 2^N grows fast
shapley_n_values = [3, 4, 5]
for n in shapley_n_values:
    client_datasets = partition_generic_iid(full_train, n_clients=n, size_each=2000)
    start = time.time()
    compute_exact_shapley(client_datasets, device, test_loader, num_rounds=3, local_epochs=1)
    elapsed = time.time() - start
    results.append({"method": "Exact Shapley", "n_clients": n, "runtime": elapsed})
    print(f"Exact Shapley, N={n}: {elapsed:.1f}s")

# TMC-Shapley: designed to scale, so test at larger N cheaply
tmc_n_values = [3, 4, 5, 6, 8, 10]
for n in tmc_n_values:
    client_datasets = partition_generic_iid(full_train, n_clients=n, size_each=2000)
    start = time.time()
    compute_tmc_shapley(
        client_datasets, device, test_loader,
        num_rounds=3, local_epochs=1, num_samples=10, truncation_threshold=0.001
    )
    elapsed = time.time() - start
    results.append({"method": "TMC-Shapley", "n_clients": n, "runtime": elapsed})
    print(f"TMC-Shapley, N={n}: {elapsed:.1f}s")    

print("\n===== SCALABILITY RESULTS =====")
for r in results:
    print(f"{r['method']}, N={r['n_clients']}: {r['runtime']:.1f}s")

with open("scalability_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Method", "N Clients", "Runtime (seconds)"])
    for r in results:
        writer.writerow([r["method"], r["n_clients"], r["runtime"]])