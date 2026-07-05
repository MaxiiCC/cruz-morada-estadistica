"""
main.py - Orquestador principal del análisis estadístico Cruz Morada.
"""
import sys
import os
import time
import logging
import numpy as np
import random
import warnings

# Ocultar todos los warnings de librerías en la terminal
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SEED
random.seed(SEED)
np.random.seed(SEED)

from reporte import configurar_logging, generar_resultados_txt
from carga_datos import cargar_csv, resumen_inicial
from preprocesamiento import preprocesar
from eda import ejecutar_eda
from inferencia import ejecutar_inferencia
from modelado import entrenar_modelo
from series_tiempo import ejecutar_series_tiempo
from paralelismo import ejecutar_paralelismo

def main(ruta_csv):
    configurar_logging()
    logger = logging.getLogger("main")

    print("\n" + "="*70)
    print("  ANÁLISIS ESTADÍSTICO DE VENTAS — CRUZ MORADA")
    print("  Computación Paralela y Distribuida — UTEM 2026")
    print(f"  Semilla: CPYD_SEED={SEED}")
    print("="*70 + "\n")

    t_total_inicio = time.perf_counter()
    res = {}

    print("▶ [1/7] Cargando datos...")
    df = cargar_csv(ruta_csv)
    resumen = resumen_inicial(df)
    print(f"  ✓ Carga exitosa: {len(df):,} filas | Canales: {resumen['canales_unicos']}\n")

    print("▶ [2/7] Preprocesando datos...")
    df, rep_prep = preprocesar(df)
    res["preprocesamiento"] = rep_prep
    print(f"  ✓ Preprocesamiento completo: {len(df):,} filas finales\n")

    print("▶ [3/7] Análisis exploratorio (EDA)...")
    res["eda"] = ejecutar_eda(df)
    print("  ✓ EDA completo. Gráficos guardados en output/graficos/\n")

    print("▶ [4/7] Pruebas de hipótesis (H1–H5)...")
    res["inferencia"] = ejecutar_inferencia(df)
    print("  ✓ Inferencia completa.\n")

    print("▶ [5/7] Regresión OLS con diagnósticos...")
    res["modelado"] = entrenar_modelo(df)
    print("  ✓ Modelado completo.\n")

    print("▶ [6/7] Análisis de series de tiempo...")
    res["series_tiempo"] = ejecutar_series_tiempo(df)
    print("  ✓ Series de tiempo completadas.\n")

    print("▶ [7/7] Procesamiento paralelo y benchmark...")
    res["paralelismo"] = ejecutar_paralelismo(df)
    print("  ✓ Paralelo completo.\n")

    print("▶ Generando resultados.txt...")
    generar_resultados_txt(res)
    print(f"  ✓ resultados.txt generado con éxito.")

    t_total = time.perf_counter() - t_total_inicio
    print("\n" + "="*70)
    print("  PIPELINE COMPLETADO SIN ERRORES")
    print(f"  Tiempo total: {t_total:.1f} segundos")
    print("="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USO: python src/main.py <ruta_al_csv>")
        sys.exit(1)
    main(sys.argv[1])
