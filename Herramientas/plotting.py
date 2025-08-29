import argparse          # Para manejar argumentos desde la línea de comandos
from pathlib import Path # Manejo de rutas de archivos y directorios
import pandas as pd      # Manejo de datos tabulares (lectura CSV, etc.)
import numpy as np       # Operaciones matemáticas y arrays
import matplotlib.pyplot as plt  # Librería para graficar
import zipfile           # Para comprimir resultados en un archivo ZIP
import sys               # Para manejo de errores y salidas del programa

# -------------------------
# Funciones Auxiliares
# -------------------------
def detect_columns(summary, combined):
    """
    Detecta nombres de columnas relevantes en los DataFrames de entrada
    (summary y combined) buscando palabras clave como 'n', 'threads', 'time', etc.
    Devuelve un diccionario con las columnas detectadas y un diagnóstico.
    """
    cols_summary = list(summary.columns)
    cols_combined = list(combined.columns)

    # Función interna para buscar columnas por palabras clave
    def find(dfcols, keywords):
        for k in keywords:
            for c in dfcols:
                if k in c.lower():
                    return c
        return None

    # Buscar columnas relevantes
    n_col = find(cols_summary, ['n', 'dim', 'size'])
    threads_col = find(cols_summary, ['thread', 'threads', 'th'])
    mean_time_col = find(cols_summary, ['mean_time', 'mean time', 'mean_time_s', 'mean'])
    gflops_col = find(cols_summary, ['gflop', 'gflops', 'flops'])

    comb_time = find(cols_combined, ['time', 'duration', 'sec', 'ms'])
    comb_n = find(cols_combined, ['n', 'dim', 'size'])
    comb_threads = find(cols_combined, ['thread', 'threads', 'th'])
    comb_run = find(cols_combined, ['run', 'iter', 'rep', 'trial'])

    diagnostic = {
        'summary_cols': cols_summary,
        'combined_cols': cols_combined,
        'mapped': {
            'n_col': n_col, 'threads_col': threads_col, 'mean_time_col': mean_time_col, 'gflops_col': gflops_col,
            'comb_time': comb_time, 'comb_n': comb_n, 'comb_threads': comb_threads, 'comb_run': comb_run
        }
    }
    return diagnostic

def standardize_names(summary, combined, convert_ms=False):
    """
    Estandariza los nombres de las columnas en ambos DataFrames.
    Convierte tiempos de milisegundos a segundos si se activa convert_ms.
    Valida que existan las columnas necesarias para continuar.
    """
    diag = detect_columns(summary, combined)
    mapped = diag['mapped']

    # Renombrar columnas en summary
    if mapped['n_col']:
        summary = summary.rename(columns={mapped['n_col']: 'n'})
    if mapped['threads_col']:
        summary = summary.rename(columns={mapped['threads_col']: 'threads'})
    if mapped['mean_time_col']:
        summary = summary.rename(columns={mapped['mean_time_col']: 'mean_time'})
    if mapped['gflops_col']:
        summary = summary.rename(columns={mapped['gflops_col']: 'mean_gflops'})

    # Renombrar columnas en combined
    if mapped['comb_n']:
        combined = combined.rename(columns={mapped['comb_n']: 'n'})
    if mapped['comb_threads']:
        combined = combined.rename(columns={mapped['comb_threads']: 'threads'})
    if mapped['comb_run']:
        combined = combined.rename(columns={mapped['comb_run']: 'run'})
    if mapped['comb_time']:
        ct = mapped['comb_time']
        # Si la columna contiene 'ms' o se pidió conversión explícita
        if 'ms' in ct.lower() or convert_ms:
            combined = combined.rename(columns={ct: 'time_ms'})
            if convert_ms:
                combined['time_s'] = combined['time_ms'] / 1000.0
                time_col = 'time_s'
            else:
                # Mantener ambas: ms y s
                combined['time_s'] = combined['time_ms'] / 1000.0
                time_col = 'time_s'
        else:
            combined = combined.rename(columns={ct: 'time'})
            time_col = 'time'  # asumimos segundos
    else:
        time_col = 'time' if 'time' in combined.columns else None

    # Validaciones
    if 'n' not in summary.columns or 'threads' not in summary.columns or 'mean_time' not in summary.columns:
        raise ValueError(f"Faltan columnas en summary. Diagnóstico: {diag}")

    if 'n' not in combined.columns or 'threads' not in combined.columns or time_col is None:
        raise ValueError(f"Faltan columnas en combined. Diagnóstico: {diag}")

    # Convertir tipos de datos
    summary['n'] = summary['n'].astype(int)
    summary['threads'] = summary['threads'].astype(int)
    combined['n'] = combined['n'].astype(int)
    combined['threads'] = combined['threads'].astype(int)

    return summary, combined, time_col, diag

def add_speedup_eff(summary):
    """
    Calcula speedup y eficiencia para cada valor de n.
    speedup = tiempo_con_1_hilo / tiempo_con_threads
    eficiencia = speedup / threads
    """
    s = summary.copy()
    s['speedup'] = np.nan
    s['efficiency'] = np.nan
    for n_val, g in s.groupby('n'):
        if 1 in g['threads'].values:
            base = g.loc[g['threads']==1, 'mean_time'].values[0]
            idx = s['n'] == n_val
            s.loc[idx, 'speedup'] = base / s.loc[idx, 'mean_time'].replace(0, np.nan)
            s.loc[idx, 'efficiency'] = s.loc[idx, 'speedup'] / s.loc[idx, 'threads'].astype(float)
    return s

def savefig(fig, outdir, name, dpi=300):
    """
    Guarda una figura en el directorio especificado con nombre y dpi dados.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return path

# -------------------------
# Funciones de Graficación
# -------------------------
def plot_speedup(summary, outdir):
    """
    Grafica speedup vs número de hilos para cada N.
    """
    fig, ax = plt.subplots(figsize=(10,6))
    for n_val, g in summary.groupby('n'):
        if 1 not in g['threads'].values:
            continue
        ax.plot(g['threads'], g['speedup'], marker='o', label=f"N={n_val}")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup (relativo a 1 hilo)")
    ax.set_title("Speedup vs Threads")
    ax.grid(True)
    ax.legend(fontsize='small', ncol=2)
    return savefig(fig, outdir, "speedup_vs_threads_allN.png")

def plot_gflops(summary, outdir):
    """
    Grafica GFLOPS promedio vs número de hilos.
    Si no hay columna mean_gflops, se omite.
    """
    if 'mean_gflops' not in summary.columns:
        print("mean_gflops no encontrado — se omite la gráfica de GFLOPS.")
        return None
    fig, ax = plt.subplots(figsize=(10,6))
    for n_val, g in summary.groupby('n'):
        ax.plot(g['threads'], g['mean_gflops'], marker='o', label=f"N={n_val}")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Mean GFLOPS")
    ax.set_title("GFLOPS vs Threads")
    ax.grid(True)
    ax.legend(fontsize='small', ncol=2)
    return savefig(fig, outdir, "gflops_vs_threads_allN.png")

def plot_efficiency(summary, outdir):
    """
    Grafica la eficiencia paralela vs número de hilos.
    """
    fig, ax = plt.subplots(figsize=(10,6))
    for n_val, g in summary.groupby('n'):
        if g['efficiency'].notna().any():
            ax.plot(g['threads'], g['efficiency'], marker='o', label=f"N={n_val}")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Efficiency (speedup/threads)")
    ax.set_title("Parallel Efficiency vs Threads")
    ax.set_ylim(0,1.05)
    ax.grid(True)
    ax.legend(fontsize='small', ncol=2)
    return savefig(fig, outdir, "efficiency_vs_threads_allN.png")

def plot_time_with_ci(summary, outdir, n_list=None):
    """
    Grafica tiempo promedio con intervalo de confianza (95%) si existe.
    Genera un gráfico por cada valor de N.
    """
    if n_list is None:
        n_list = sorted(summary['n'].unique())
    saved = []
    for n_val in n_list:
        g = summary[summary['n']==n_val].sort_values('threads')
        if g.empty:
            continue
        if 'ci95_time_lo' in g.columns and 'ci95_time_hi' in g.columns:
            # Gráfico con barras de error
            y = g['mean_time'].values
            low = g['ci95_time_lo'].values
            high = g['ci95_time_hi'].values
            yerr = np.vstack([y - low, high - y])
            fig, ax = plt.subplots(figsize=(8,5))
            ax.errorbar(g['threads'], y, yerr=yerr, fmt='o-', capsize=4)
            ax.set_title(f"Mean time con 95% CI — N={n_val}")
            ax.set_xlabel("Threads"); ax.set_ylabel("Mean time (s)")
            ax.grid(True)
        else:
            # Gráfico simple sin CI
            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(g['threads'], g['mean_time'], 'o-')
            ax.set_title(f"Mean time — N={n_val}")
            ax.set_xlabel("Threads"); ax.set_ylabel("Mean time (s)")
            ax.grid(True)
        saved.append(savefig(fig, outdir, f"time_mean_with_ci_N{n_val}.png"))
    return saved

def plot_boxplots(combined, outdir, time_col, sample_ns=None):
    """
    Genera diagramas de caja (boxplots) del tiempo de ejecución
    para cada número de hilos y tamaño de problema N.
    """
    if sample_ns is None:
        sample_ns = sorted(combined['n'].unique())
        saved = []
    for n_val in sample_ns:
        df_n = combined[combined['n']==n_val]
        if df_n.empty:
            continue
        threads_order = sorted(df_n['threads'].unique())
        data = [df_n[df_n['threads']==t][time_col].values for t in threads_order]
        if all(len(d)==0 for d in data):
            continue
        fig, ax = plt.subplots(figsize=(8,5))
        ax.boxplot(data, labels=[str(t) for t in threads_order], showfliers=True)
        ax.set_xlabel("Threads"); ax.set_ylabel(f"Time (s)")
        ax.set_title(f"Distribución de tiempos por threads — N={n_val}")
        ax.grid(axis='y')
        saved.append(savefig(fig, outdir, f"boxplot_time_N{n_val}.png"))
    return saved

# -------------------------
# Función Principal
# -------------------------
def main():
    """
    Punto de entrada principal del script.
    - Lee argumentos desde la línea de comandos
    - Procesa los CSV de summary y combined
    - Calcula métricas (speedup, eficiencia)
    - Genera gráficas y las guarda
    - Crea un ZIP con todas las gráficas
    """
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True, help='Archivo summary_stats.csv')
    p.add_argument('--combined', required=True, help='Archivo combined_raw.csv')
    p.add_argument('--outdir', default='./plots', help='Directorio de salida para gráficas')
    p.add_argument('--convert-ms', action='store_true', help='Si los tiempos están en ms, convertir a segundos')
    p.add_argument('--plots', default='all', help='Lista de gráficas: speedup,gflops,efficiency,timeci,boxplots,heatmap')
    args = p.parse_args()

    # Leer CSV
    summary = pd.read_csv(args.summary)
    summary = summary.rename(columns={"mean_time": "mean_time", "mean_time_ms":"mean_time"})
    combined = pd.read_csv(args.combined)
    combined = combined.rename(columns={"time_ms":"time_ms"})

    # Estandarizar nombres
    try:
        summary, combined, time_col_comb, diag = standardize_names(summary, combined, convert_ms=args.convert_ms)
    except Exception as e:
        print("Error detectando/estandarizando columnas:", e)
        sys.exit(1)

    # Determinar la columna de tiempo a usar
    time_col = time_col_comb if 'time_s' in combined.columns else (time_col_comb if time_col_comb else 'time')

    # Validación
    if 'mean_time' not in summary.columns:
        raise ValueError("summary debe contener una columna mean_time.")

    # Calcular speedup y eficiencia
    summary = add_speedup_eff(summary)

    # Determinar qué gráficas generar
    todo = [x.strip().lower() for x in args.plots.split(',')] if args.plots != 'all' else ['speedup','gflops','efficiency','timeci','boxplots','heatmap']

    generated = []
    if 'speedup' in todo:
        p = plot_speedup(summary, args.outdir); generated.append(p)
    if 'gflops' in todo:
        p = plot_gflops(summary, args.outdir)
        if p: generated.append(p)
    if 'efficiency' in todo:
        p = plot_efficiency(summary, args.outdir); generated.append(p)
    if 'timeci' in todo:
        all_ns = sorted(summary['n'].unique())
        ps = plot_time_with_ci(summary, args.outdir, n_list=all_ns)
        generated += ps
    if 'boxplots' in todo:
        ps = plot_boxplots(combined, args.outdir, time_col)
        generated += ps

    # Crear archivo ZIP con las gráficas
    outzip = Path(args.outdir) / "plots_bundle.zip"
    with zipfile.ZipFile(outzip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in generated:
            if f is None:
                continue
            z.write(f, arcname=Path(f).name)

    # Mostrar resultados en consola
    print("Gráficas generadas:", [Path(x).name for x in generated])
    print("Archivo ZIP generado:", outzip)

if __name__ == "__main__":
    main()
