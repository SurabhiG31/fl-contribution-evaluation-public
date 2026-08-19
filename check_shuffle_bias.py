import random

client_names = ["client_1", "client_2", "client_3"]

for seed in [0, 42, 7, 100, 2024]:
    rng = random.Random(seed)
    first_position_counts = {name: 0 for name in client_names}
    for _ in range(10):
        permutation = client_names.copy()
        rng.shuffle(permutation)
        first_position_counts[permutation[0]] += 1
    print(f"seed={seed}: {first_position_counts}")