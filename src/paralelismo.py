"""
paralelismo.py - Benchmark secuencial vs paralelo usando multiprocessing.
Estrategia: procesar múltiples archivos CSV simulados en paralelo,
reflejando el caso de uso real donde Cruz Morada recibe reportes
diarios de cada local que deben procesarse simultáneamente.

NOTA METODOLÓGICA: este benchmark mide el speedup de multiprocessing
sobre una carga de trabajo representativa (estadísticas + bootstrap por
partición de datos), pero es un benchmark aislado: no paraleliza las
etapas del pipeline real (preprocesamiento.py, eda.py, modelado.py corren
secuencialmente sobre un único DataFrame). Se documenta así explícitamente
en el informe para ser honestos sobre el alcance del paralelismo aplicado.
"""
import time
import logging
import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Pool, cpu_count

from config import COL_LOCAL, COL_MONTO, COL_DESCUENTO, COL_CANAL, SEED, UMBRAL_SHAPIRO, n_jobs_reales

logger = logging.getLogger(__name__)

N_ARCHIVOS_SIMULADOS = 32  # Simula 32 reportes diarios de locales
N_BOOTSTRAP = 40000  # Iteraciones de bootstrap por partición (carga de CPU)


def _procesar_reporte_local(args):
    """
    Simula el procesamiento completo de un reporte diario de un local:
    carga, limpieza, estadísticas completas y tests de hipótesis.
    El bootstrap con N_BOOTSTRAP iteraciones fuerza carga de CPU real,
    justificando la paralelización (no es solo I/O-bound).
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

    # 2. Test Shapiro-Wilk (usa el mismo umbral que eda.py/modelado.py)
    muestra_sw = (monto_arr if len(monto_arr) <= UMBRAL_SHAPIRO
                  else rng.choice(monto_arr, UMBRAL_SHAPIRO, replace=False))
    _, p_sw = stats.shapiro(muestra_sw) if len(muestra_sw) >= 3 else (0, 1.0)

    # 3. Correlación Spearman
    n_min = min(len(monto_arr), len(descuento_arr))
    rho, p_rho = stats.spearmanr(
        monto_arr[:n_min], descuento_arr[:n_min]
    ) if n_min > 3 else (0.0, 1.0)

    # 4. Bootstrap IC 95% de la media (fuerza carga de CPU real, no solo I/O)
    bootstrap = np.array([
        rng.choice(monto_arr, size=min(500, len(monto_arr)), replace=True).mean()
        for _ in range(N_BOOTSTRAP)
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

    # Semillas derivadas de la semilla global (config.SEED), no un 42
    # hardcodeado suelto en este módulo. Así todo el pipeline respeta
    # una única fuente de reproducibilidad (CPYD_SEED).
    return [
        (m, d, i, SEED + i)
        for i, (m, d) in enumerate(zip(monto_chunks, desc_chunks))
    ]


def procesar_secuencial(tareas):
    t_ini = time.perf_counter()
    res = [_procesar_reporte_local(t) for t in tareas]
    return res, time.perf_counter() - t_ini


def procesar_paralelo(tareas, n_procesos):
    t_ini = time.perf_counter()
    with Pool(processes=n_procesos) as pool:
        res = pool.map(_procesar_reporte_local, tareas)
    return res, time.perf_counter() - t_ini


def _verificar_resultados_identicos(res_seq, res_par):
    """
    Verifica que el modo secuencial y el paralelo produzcan resultados
    IDÉNTICOS (mismo pipeline, misma semilla, misma partición). Esto es
    clave para el informe: demuestra que paralelizar no introduce
    inconsistencias ni condiciones de carrera, solo acelera el cómputo.
    """
    if len(res_seq) != len(res_par):
        logger.warning(f"Discrepancia de tamaño entre resultados secuenciales ({len(res_seq)}) y paralelos ({len(res_par)}).")
        return False

    por_id_par = {r["id"]: r for r in res_par}
    todos_iguales = True
    for r_seq in res_seq:
        r_par = por_id_par.get(r_seq["id"])
        if r_par != r_seq:
            todos_iguales = False
            logger.warning(f"Resultado distinto entre secuencial y paralelo para el local id={r_seq['id']}.")

    return todos_iguales


def ejecutar_paralelismo(df):
    n_nucleos = cpu_count()
    n_procesos = n_jobs_reales()
    tareas = _preparar_tareas(df, N_ARCHIVOS_SIMULADOS)

    if n_nucleos == 1:
        logger.warning("Solo hay 1 núcleo disponible: no se espera speedup real del procesamiento paralelo.")

    logger.info(
        f"Benchmark paralelo: {N_ARCHIVOS_SIMULADOS} tareas, "
        f"{n_nucleos} núcleos disponibles, usando {n_procesos} proceso(s)"
    )

    res_seq, t_sec = procesar_secuencial(tareas)
    res_par, t_par = procesar_paralelo(tareas, n_procesos)

    resultados_consistentes = _verificar_resultados_identicos(res_seq, res_par)
    if resultados_consistentes:
        logger.info("Verificación de consistencia: los resultados secuencial y paralelo son idénticos.")
    else:
        logger.warning("Verificación de consistencia: se detectaron diferencias entre secuencial y paralelo (revisar log).")

    speedup = t_sec / t_par if t_par > 0 else 1.0
    eficiencia = (speedup / n_procesos) * 100 if n_procesos > 0 else 0.0

    logger.info(
        f"Secuencial: {t_sec:.4f}s | "
        f"Paralelo: {t_par:.4f}s | "
        f"Speedup: {speedup:.2f}x | "
        f"Eficiencia: {eficiencia:.2f}%"
    )

    return {
        "n_nucleos": n_nucleos,
        "n_procesos_usados": n_procesos,
        "n_particiones": N_ARCHIVOS_SIMULADOS,
        "n_filas_total": len(df),
        "tiempo_secuencial": round(t_sec, 4),
        "tiempo_paralelo": round(t_par, 4),
        "speedup": round(speedup, 4),
        "eficiencia_pct": round(eficiencia, 2),
        "resultados_consistentes": resultados_consistentes,
    }