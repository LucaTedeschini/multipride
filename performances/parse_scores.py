import os
import csv
import re
import numpy as np
from collections import defaultdict

# Controllo libreria scipy
try:
    from scipy.stats import ttest_ind
except ImportError:
    print("Errore: La libreria 'scipy' non è installata. Esegui: pip install scipy")
    exit()

BASE_DIR = os.getcwd()
OUTPUT_CSV = "aggregated_results.csv"

# Nome dell'architettura di riferimento
BASELINE_ARCH_NAME = "baseline" 

# Struttura dati: results[nome_esperimento][nome_architettura]["f1"|"acc"] -> list
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Regex per i file CSV
csv_pattern = re.compile(r"(dual_encoder|lgbt_pretrain|baseline).*?_(\d+)\.csv$")

print(f"Scanning directory: {BASE_DIR}\n")

# ==========================
# 1. LETTURA DATI
# ==========================
for experiment_folder in os.listdir(BASE_DIR):
    experiment_path = os.path.join(BASE_DIR, experiment_folder)

    if not os.path.isdir(experiment_path) or experiment_folder.startswith("."):
        continue

    found_files = 0
    for file in os.listdir(experiment_path):
        if not file.endswith(".csv"):
            continue

        match = csv_pattern.search(file)
        if not match:
            continue

        architecture, seed = match.groups()
        file_path = os.path.join(experiment_path, file)

        try:
            with open(file_path, newline="", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                try:
                    row = next(reader)
                    results[experiment_folder][architecture]["f1"].append(float(row["f1"]))
                    results[experiment_folder][architecture]["acc"].append(float(row["acc"]))
                    found_files += 1
                except StopIteration:
                    pass # File vuoto
                except KeyError:
                    pass # Colonne mancanti
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if found_files > 0:
        print(f"Found {found_files} CSVs in: {experiment_folder}")

# ==========================
# 2. AGGREGAZIONE E CALCOLO P-VALUE (CROSS-FOLDER)
# ==========================

rows = []
print("\n=== AGGREGATED RESULTS WITH CROSS-FOLDER P-VALUES ===\n")

if not results:
    print("No results found.")

for experiment_name, arch_data in sorted(results.items()):
    print(f"Experiment: {experiment_name}")

    # --- LOGICA PER TROVARE LA BASELINE ---
    # Caso 1: La baseline è dentro questa stessa cartella? (es. cartella mista)
    if BASELINE_ARCH_NAME in arch_data:
        baseline_source = "local"
        ref_f1 = arch_data[BASELINE_ARCH_NAME]["f1"]
        ref_acc = arch_data[BASELINE_ARCH_NAME]["acc"]
    
    # Caso 2: La baseline è in un'altra cartella? (es. final_es vs baseline_es)
    else:
        # Cerchiamo di capire il suffisso (es. "es" da "final_es")
        # Assumiamo che il nome sia tipo "qualcosa_es" o "qualcosa_it"
        parts = experiment_name.split('_')
        if len(parts) > 1:
            suffix = parts[-1] # "es", "it", etc.
            # Costruiamo il nome presunto della cartella baseline
            target_baseline_folder = f"baseline_{suffix}"
            
            # Controlliamo se esiste questa cartella nei risultati e se ha la baseline
            if (target_baseline_folder in results and 
                BASELINE_ARCH_NAME in results[target_baseline_folder]):
                
                baseline_source = target_baseline_folder
                ref_f1 = results[target_baseline_folder][BASELINE_ARCH_NAME]["f1"]
                ref_acc = results[target_baseline_folder][BASELINE_ARCH_NAME]["acc"]
            else:
                baseline_source = None
        else:
            baseline_source = None

    # Se abbiamo trovato una fonte per i dati di baseline
    has_baseline_data = (baseline_source is not None and len(ref_f1) > 1)

    for architecture, metrics in arch_data.items():
        f1 = np.array(metrics["f1"])
        acc = np.array(metrics["acc"])

        if len(f1) == 0:
            continue

        # Medie e Deviazioni Standard
        f1_mean = f1.mean()
        f1_std = f1.std(ddof=1) if len(f1) > 1 else 0.0
        acc_mean = acc.mean()
        acc_std = acc.std(ddof=1) if len(acc) > 1 else 0.0

        # --- CALCOLO P-VALUE ---
        p_val_f1 = np.nan
        p_val_acc = np.nan
        p_str = ""

        # Calcoliamo solo se:
        # 1. Non siamo noi stessi la baseline
        # 2. Abbiamo trovato i dati della baseline corrispondente
        # 3. Abbiamo abbastanza seed (>1)
        if architecture != BASELINE_ARCH_NAME and has_baseline_data and len(f1) > 1:
            try:
                # Welch's t-test (equal_var=False)
                _, p_val_f1 = ttest_ind(f1, ref_f1, equal_var=False)
                _, p_val_acc = ttest_ind(acc, ref_acc, equal_var=False)
                
                source_label = "self" if baseline_source == "local" else baseline_source
                p_str = f"| p-val (vs {source_label}): F1={p_val_f1:.4f}, Acc={p_val_acc:.4f}"
            except Exception as e:
                p_str = "| p-val Error"

        elif architecture == BASELINE_ARCH_NAME:
            p_str = "| (REFERENCE)"

        print(
            f"  {architecture:15s} | "
            f"F1: {f1_mean:.4f} ± {f1_std:.4f} | "
            f"Acc: {acc_mean:.4f} ± {acc_std:.4f} | Seeds: {len(f1)} "
            f"{p_str}"
        )

        rows.append({
            "experiment": experiment_name,
            "architecture": architecture,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "p_val_f1": p_val_f1 if not np.isnan(p_val_f1) else "",
            "p_val_acc": p_val_acc if not np.isnan(p_val_acc) else "",
            "baseline_source": baseline_source if (architecture != BASELINE_ARCH_NAME and has_baseline_data) else "",
            "num_seeds": len(f1)
        })

    print()

# ==========================
# 3. SALVATAGGIO CSV
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
                "p_val_f1",
                "p_val_acc",
                "baseline_source",
                "num_seeds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved aggregated results to {os.path.join(BASE_DIR, OUTPUT_CSV)}")
else:
    print("No data to save.")