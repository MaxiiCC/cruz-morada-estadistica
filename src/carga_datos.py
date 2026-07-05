"""
carga_datos.py - Módulo para cargar el CSV usando chunks.
"""
import time
import logging
import pandas as pd
from pathlib import Path
from config import CHUNK_SIZE, COL_FECHA, COL_FECHA_NACIMIENTO

logger = logging.getLogger(__name__)

def cargar_csv(ruta_csv, verbose=True):
    ruta = Path(ruta_csv)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta}")

    t_inicio = time.time()
    dtype_mapa = {
        "SKU": str,
        "UNIDADES": "Int64",
        "PORCENTAJE DESCUENTO": float,
        "MONTO APLICADO": float,
        "BOLETA": "Int64",
        "LOCAL": "Int64",
        "GENERO": "Int64",
    }

    chunks = []
    filas_total = 0

    lector = pd.read_csv(
        ruta,
        sep=";",
        encoding="latin-1",
        chunksize=CHUNK_SIZE,
        dtype=dtype_mapa,
        parse_dates=[COL_FECHA, COL_FECHA_NACIMIENTO],
        dayfirst=False,
        low_memory=False,
    )

    for i, chunk in enumerate(lector):
        chunks.append(chunk)
        filas_total += len(chunk)
        if verbose:
            print(f"  Bloque {i+1} cargado - {filas_total:,} filas leídas", end="\r")

    df = pd.concat(chunks, ignore_index=True)
    tiempo_carga = time.time() - t_inicio
    return df

def resumen_inicial(df):
    return {
        "n_filas": len(df),
        "n_columnas": len(df.columns),
        "duplicados": df.duplicated().sum(),
        "canales_unicos": df["CANAL"].unique().tolist() if "CANAL" in df.columns else [],
        "locales_unicos": sorted(df["LOCAL"].dropna().unique().tolist()) if "LOCAL" in df.columns else [],
    }
