import os
import csv
import re
import numpy as np
from collections import defaultdict

BASE_DIR = os.getcwd()
OUTPUT_CSV = "aggregated_results.csv"

# results[experiment][architecture]["f1"|"acc"] -> list
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# MODIFICA QUI:
# 1. (dual_encoder|lgbt_pretrain): Cattura l'architettura (Gruppo 1)
# 2. .*?: Ignora qualsiasi cosa nel mezzo (non-greedy)
# 3. _(\d+): Cattura l'underscore seguito dai numeri (Gruppo 2, il Seed)
# 4. \.csv$: Assicura che finisca con .csv
csv_pattern = re.compile(r"(dual_encoder|lgbt_pretrain).*?_(\d+)\.csv$")

print(f"Scanning directory: {BASE_DIR}\n")

for experiment in os.listdir(BASE_DIR):
    experiment_path = os.path.join(BASE_DIR, experiment)

    if not os.path.isdir(experiment_path):
        continue

    # Saltiamo eventuali cartelle nascoste o non pertinenti se necessario
    if experiment.startswith("."): 
        continue

    found_files = 0
    for file in os.listdir(experiment_path):
        if not file.endswith(".csv"):
            continue

        match = csv_pattern.search(file)
        if not match:
            # Opzionale: de-commenta per vedere quali file vengono ignorati
            # print(f"Skipping non-matching file: {file}")
            continue

        architecture, seed = match.groups()
        file_path = os.path.join(experiment_path, file)

        try:
            with open(file_path, newline="", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # Gestione file vuoti o malformati
                try:
                    row = next(reader)
                    results[experiment][architecture]["f1"].append(float(row["f1"]))
                    results[experiment][architecture]["acc"].append(float(row["acc"]))
                    found_files += 1
                except StopIteration:
                    print(f"Warning: File empty {file_path}")
                except KeyError:
                    print(f"Warning: Missing columns in {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if found_files > 0:
        print(f"Found {found_files} valid CSVs in experiment: {experiment}")

# ==========================
# AGGREGAZIONE
# ==========================

rows = []

print("\n=== AGGREGATED RESULTS ===\n")

# Se results è vuoto, avvisiamo
if not results:
    print("No results found. Check regex or directory structure.")

for experiment, arch_data in sorted(results.items()):
    print(f"Experiment: {experiment}")

    for architecture, metrics in arch_data.items():
        f1 = np.array(metrics["f1"])
        acc = np.array(metrics["acc"])

        if len(f1) == 0:
            continue

        # ddof=1 per deviazione standard campionaria, se hai 1 solo seed evito NaN
        f1_mean = f1.mean()
        f1_std = f1.std(ddof=1) if len(f1) > 1 else 0.0
        
        acc_mean = acc.mean()
        acc_std = acc.std(ddof=1) if len(acc) > 1 else 0.0

        print(
            f"  {architecture:15s} | "
            f"F1: {f1_mean:.4f} ± {f1_std:.4f} | "
            f"Acc: {acc_mean:.4f} ± {acc_std:.4f} | Seeds: {len(f1)}"
        )

        rows.append({
            "experiment": experiment,
            "architecture": architecture,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "num_seeds": len(f1)
        })

    print()

# ==========================
# CSV OUTPUT
# ==========================

if rows:
    with open(OUTPUT_CSV, "w", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "architecture",
                "f1_mean",
                "f1_std",
                "acc_mean",
                "acc_std",
                "num_seeds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved aggregated results to {os.path.join(BASE_DIR, OUTPUT_CSV)}")
else:
    print("No data to save.")