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
import traceback

# Ocultar warnings de librerías en la terminal (siguen quedando registrados
# en el log de cada módulo vía logger.warning donde corresponde).
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SEED, RUTA_CSV_DEFAULT
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

    print("\n" + "=" * 70)
    print("  ANÁLISIS ESTADÍSTICO DE VENTAS — CRUZ MORADA")
    print("  Computación Paralela y Distribuida — UTEM 2026")
    print(f"  Semilla: CPYD_SEED={SEED}")
    print("=" * 70 + "\n")

    t_total_inicio = time.perf_counter()
    res = {}

    try:
        print("▶ [1/7] Cargando datos...")
        df = cargar_csv(ruta_csv)
        resumen = resumen_inicial(df)
        tiempo_carga = resumen.get("tiempo_carga_seg")
        pct_descartado = resumen.get("pct_descartado_carga")
        extra = ""
        if tiempo_carga is not None:
            extra = f" | Carga: {tiempo_carga:.2f}s"
        if pct_descartado is not None:
            extra += f" | Descartado en carga: {pct_descartado:.2f}%"
        print(f"  ✓ Carga exitosa: {len(df):,} filas | Canales: {resumen['canales_unicos']}{extra}\n")

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

    except Exception as e:
        # Si algo falla a mitad de camino (ej. en el paso 6 de 7), no se
        # pierden los resultados ya calculados: se registra el error
        # completo en el log y se genera un resultados.txt PARCIAL con
        # lo que sí alcanzó a correr, en vez de perder todo el trabajo
        # de los pasos anteriores.
        logger.error(f"El pipeline falló durante la ejecución: {e}")
        logger.error(traceback.format_exc())
        print(f"\n  ✗ ERROR durante el pipeline: {e}")
        print("  Se generará un resultados.txt PARCIAL con lo que se alcanzó a calcular.\n")
        try:
            generar_resultados_txt(res)
            print("  ✓ resultados.txt (parcial) generado. Revisa output/log.txt para el detalle del error.")
        finally:
            raise

    print("▶ Generando resultados.txt...")
    generar_resultados_txt(res)
    print("  ✓ resultados.txt generado con éxito.")

    t_total = time.perf_counter() - t_total_inicio
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETADO SIN ERRORES")
    print(f"  Tiempo total: {t_total:.1f} segundos")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        ruta_default = str(RUTA_CSV_DEFAULT)
        print(f"No se especificó ruta de CSV, usando por defecto: {ruta_default}")
        print("(Uso explícito: python src/main.py <ruta_al_csv>)\n")
        main(ruta_default)
    else:
        main(sys.argv[1])