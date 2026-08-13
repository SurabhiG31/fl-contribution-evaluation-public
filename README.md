# fl-contribution-evaluation-public

# Federated Learning Contribution Evaluation

Empirical comparison of Leave-One-Out (LOO), exact Shapley value, and
TMC-Shapley for contribution evaluation in federated learning, using a
custom FedAvg simulator built from scratch and tested on MNIST.

## Setup

Install dependencies:

pip install torch torchvision scipy numpy --break-system-packages

## Files

- `model.py` — CNN architecture used as the shared federated model
- `train_utils.py` — training/evaluation helpers, reproducibility seeding, cost counters
- `data.py` — client partitioning: label skew, quantity skew, IID and non-IID overlap
- `federated.py` — FedAvg aggregation and federated training loop
- `loo.py` — Leave-One-Out implementation (single-round and continuous round-checkpointed)
- `shapley.py` — Exact Shapley implementation (single-round and continuous round-checkpointed)
- `tmc_shapley.py` — TMC-Shapley: random-sampling and exhaustive-permutation variants
- `main_*.py` — experiment runner scripts, one per analysis (LOO sweep, Shapley sweep, TMC-Shapley sweep, scalability, cost counting, stability, seed variance)
- `results/` — output CSVs from each experiment (compute cost, round stability, seed variance)

## Reproducing results

Each `main_*.py` script is self-contained. Run directly, e.g.:

python3 main_loo.py
python3 main_shapley.py
python3 main_tmc.py

All random seeds are fixed and specified within each script for reproducibility.
MNIST will auto-download on first run via `load_mnist()` in `data.py`.

## Author

Surabhi Gudla
