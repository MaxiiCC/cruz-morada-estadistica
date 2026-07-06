"""
config.py - Configuración de constantes y rutas del proyecto.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

_RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(_RAIZ / ".env")

# --- Semilla reproducible (CPYD_SEED) ---------------------------------
_seed_env = os.getenv("CPYD_SEED", "42")
try:
    SEED = int(_seed_env)
except ValueError:
    raise ValueError(
        f"CPYD_SEED debe ser un entero, se recibió: {_seed_env!r}. "
        f"Revisa el archivo .env o la variable de entorno."
    )

# --- Rutas del proyecto -------------------------------------------------
RUTA_RAIZ = _RAIZ
RUTA_DATA = _RAIZ / "data"
RUTA_OUTPUT = _RAIZ / "output"
RUTA_GRAFICOS = RUTA_OUTPUT / "graficos"
RUTA_RESULTADOS = RUTA_OUTPUT / "resultados.txt"
RUTA_LOG = RUTA_OUTPUT / "log.txt"

# Ruta por defecto del CSV, usada si no se pasa argumento por línea de comandos
RUTA_CSV_DEFAULT = RUTA_DATA / "ventas_completas.csv"

RUTA_OUTPUT.mkdir(parents=True, exist_ok=True)
RUTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

# --- Parámetros generales ------------------------------------------------
CHUNK_SIZE = 50000

# N_JOBS: núcleos a usar en procesamiento paralelo (-1 = todos los disponibles).
# Se resuelve a un entero positivo real en tiempo de uso con n_jobs_reales().
N_JOBS = -1

# Umbral de tamaño de muestra para usar Shapiro-Wilk (test exacto, lento en
# muestras grandes) vs. Kolmogorov-Smirnov. Se reutiliza en eda.py,
# modelado.py y paralelismo.py para mantener consistencia entre módulos.
UMBRAL_SHAPIRO = 5000

ALPHA = 0.05
TRAIN_SIZE = 0.70


def n_jobs_reales():
    """
    Resuelve N_JOBS a un número entero positivo de procesos.
    Centraliza la lógica para que paralelismo.py (y cualquier otro
    módulo que use multiprocessing) no reimplemente la conversión
    de -1 -> cpu_count() por su cuenta.
    """
    from multiprocessing import cpu_count
    if N_JOBS is None or N_JOBS <= 0:
        return cpu_count()
    return min(N_JOBS, cpu_count())


# --- Nombres de columnas del CSV original ---------------------------------
COL_FECHA = "FECHA"
COL_CANAL = "CANAL"
COL_SKU = "SKU"
COL_PRODUCTO = "PRODUCTO"
COL_UNIDADES = "UNIDADES"
COL_DESCUENTO = "PORCENTAJE DESCUENTO"
COL_MONTO = "MONTO APLICADO"
COL_BOLETA = "BOLETA"
COL_LOCAL = "LOCAL"
COL_CODIGO_CLIENTE = "CODIGO CLIENTE"
COL_RUN_CLIENTE = "RUN CLIENTE"
COL_NOMBRES = "NOMBRES"
COL_APELLIDOS = "APELLIDOS"
COL_FECHA_NACIMIENTO = "FECHA NACIMIENTO"

# OJO: el enunciado especifica esta columna como "GÉNERO" (con tilde).
# Verifica el nombre EXACTO de la columna en tu ventas_completas.csv real
# (encoding latin-1 puede traer la tilde). Si tu CSV la trae con tilde,
# cambia esta constante a "GÉNERO"; si no, déjala así.
COL_GENERO = "GENERO"

# --- Variables derivadas (creadas en preprocesamiento.py) -----------------
COL_MONTO_POR_UNIDAD = "MONTO_POR_UNIDAD"
COL_EDAD = "EDAD"
COL_FRECUENCIA_COMPRA = "FRECUENCIA_COMPRA"

COLS_NUMERICAS = [COL_UNIDADES, COL_DESCUENTO, COL_MONTO, COL_MONTO_POR_UNIDAD, COL_EDAD, COL_FRECUENCIA_COMPRA]
COLS_CATEGORICAS = [COL_CANAL, COL_LOCAL, COL_GENERO]