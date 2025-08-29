import os
import re
import pandas as pd

# Configuración de entrada/salida
INPUT_DIR = "resultados"  
OUTPUT_RAW = "combined_raw.csv"
OUTPUT_SUMMARY = "summary_stats.csv"

# Regex para identificar dimensión y hilos en los nombres de archivo
pattern = re.compile(r"mmClasicaOpenMP-(\d+)-Hilos-(\d+)\.dat")

data = []

# Recorremos los archivos de resultados
for fname in os.listdir(INPUT_DIR):
    match = pattern.match(fname)
    if not match:
        continue

    n = int(match.group(1))       # Tamaño de matriz
    threads = int(match.group(2)) # Número de hilos

    path = os.path.join(INPUT_DIR, fname)
    with open(path, "r") as f:
        lines = f.readlines()

    # Cada línea corresponde a un tiempo de ejecución (ms)
    for i, line in enumerate(lines, start=1):
        try:
            data.append({
                "n": n,
                "threads": threads,
                "run": i,
                "time_ms": float(line.strip())
            })
        except ValueError:
            pass

# Guardamos datos crudos
df = pd.DataFrame(data)
df.to_csv(OUTPUT_RAW, index=False)
print(f"✅ CSV generado: {OUTPUT_RAW} con {len(df)} registros")

# Generamos estadísticas por tamaño e hilos
summary = df.groupby(["n", "threads"]).agg(
    mean_time=("time_ms", "mean"),
    median_time=("time_ms", "median"),
    std_time=("time_ms", "std"),
    min_time=("time_ms", "min"),
    max_time=("time_ms", "max"),
    runs=("time_ms", "count")
).reset_index()

summary.to_csv(OUTPUT_SUMMARY, index=False)
print(f"✅ CSV de resumen generado: {OUTPUT_SUMMARY}")

