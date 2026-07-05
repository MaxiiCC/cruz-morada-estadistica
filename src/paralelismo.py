"""
paralelismo.py - Benchmark secuencial vs paralelo usando multiprocessing.
Estrategia: procesar múltiples archivos CSV simulados en paralelo,
reflejando el caso de uso real donde Cruz Morada recibe reportes
diarios de cada local que deben procesarse simultáneamente.
"""
import time
import logging
import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Pool, cpu_count

from config import COL_LOCAL, COL_MONTO, COL_DESCUENTO, COL_CANAL

logger = logging.getLogger(__name__)

N_ARCHIVOS_SIMULADOS = 32  # Simula 32 reportes diarios de locales


def _procesar_reporte_local(args):
    """
    Simula el procesamiento completo de un reporte diario de un local:
    carga, limpieza, estadísticas completas y tests de hipótesis.
    Se incrementan las iteraciones de Bootstrap a 10.000 para forzar carga de CPU
    y justificar la paralelización.
    """
    monto_arr, descuento_arr, id_local, seed = args
    rng = np.random.default_rng(seed)

    if len(monto_arr) == 0:
        return {"id": id_local, "n": 0}

    # 1. Estadísticas descriptivas completas
    media = float(monto_arr.mean())
    mediana = float(np.median(monto_arr))
    std = float(monto_arr.std())
    skew = float(stats.skew(monto_arr))
    kurt = float(stats.kurtosis(monto_arr))
    percentiles = np.percentile(monto_arr, [5, 25, 50, 75, 95]).tolist()

    # 2. Test Shapiro-Wilk
    muestra_sw = (monto_arr if len(monto_arr) <= 5000
                  else rng.choice(monto_arr, 5000, replace=False))
    _, p_sw = stats.shapiro(muestra_sw) if len(muestra_sw) >= 3 else (0, 1.0)

    # 3. Correlación Spearman
    n_min = min(len(monto_arr), len(descuento_arr))
    rho, p_rho = stats.spearmanr(
        monto_arr[:n_min], descuento_arr[:n_min]
    ) if n_min > 3 else (0.0, 1.0)

    # 4. Bootstrap IC 95% de la media (40.000 iteraciones para forzar carga de CPU)
    # Esto tomará aprox 1.5s secuencial por cada local
    bootstrap = np.array([
        rng.choice(monto_arr, size=min(500, len(monto_arr)), replace=True).mean()
        for _ in range(40000)
    ])
    ic95 = (float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)))

    # 5. Test Mann-Whitney vs distribución de referencia simulada
    ref = rng.normal(loc=media, scale=std, size=min(1000, len(monto_arr)))
    _, p_mw = stats.mannwhitneyu(
        monto_arr[:1000], ref, alternative="two-sided"
    )

    return {
        "id": id_local,
        "n": len(monto_arr),
        "media": round(media, 2),
        "mediana": round(mediana, 2),
        "std": round(std, 2),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "p5_p95": [round(percentiles[0], 2), round(percentiles[4], 2)],
        "p_shapiro": round(float(p_sw), 4),
        "rho_spearman": round(float(rho), 4),
        "ic95_media": [round(ic95[0], 2), round(ic95[1], 2)],
        "p_mann_whitney": round(float(p_mw), 4),
    }


def _preparar_tareas(df, n_tareas):
    monto_arr = df[COL_MONTO].dropna().values.astype(np.float64)
    descuento_arr = df[COL_DESCUENTO].dropna().values.astype(np.float64)
    n = min(len(monto_arr), len(descuento_arr))

    monto_chunks = np.array_split(monto_arr[:n], n_tareas)
    desc_chunks = np.array_split(descuento_arr[:n], n_tareas)

    return [
        (m, d, i, 42 + i)
        for i, (m, d) in enumerate(zip(monto_chunks, desc_chunks))
    ]


def procesar_secuencial(tareas):
    t_ini = time.perf_counter()
    res = [_procesar_reporte_local(t) for t in tareas]
    return res, time.perf_counter() - t_ini


def procesar_paralelo(tareas):
    t_ini = time.perf_counter()
    with Pool(processes=cpu_count()) as pool:
        res = pool.map(_procesar_reporte_local, tareas)
    return res, time.perf_counter() - t_ini


def ejecutar_paralelismo(df):
    n_nucleos = cpu_count()
    tareas = _preparar_tareas(df, N_ARCHIVOS_SIMULADOS)

    logger.info(
        f"Benchmark paralelo: {N_ARCHIVOS_SIMULADOS} tareas, "
        f"{n_nucleos} núcleos disponibles"
    )

    res_seq, t_sec = procesar_secuencial(tareas)
    res_par, t_par = procesar_paralelo(tareas)

    speedup = t_sec / t_par if t_par > 0 else 1.0
    eficiencia = (speedup / n_nucleos) * 100

    logger.info(
        f"Secuencial: {t_sec:.4f}s | "
        f"Paralelo: {t_par:.4f}s | "
        f"Speedup: {speedup:.2f}x"
    )

    return {
        "n_nucleos": n_nucleos,
        # Se renombra a n_particiones para que se registre correctamente en resultados.txt
        "n_particiones": N_ARCHIVOS_SIMULADOS,
        "n_filas_total": len(df),
        "tiempo_secuencial": round(t_sec, 4),
        "tiempo_paralelo": round(t_par, 4),
        "speedup": round(speedup, 4),
        "eficiencia_pct": round(eficiencia, 2),
    }