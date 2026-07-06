"""
carga_datos.py - Módulo para cargar el CSV usando chunks.

El chunking aquí no es solo cosmético: cada fragmento se filtra
(descartando filas inválidas de UNIDADES/MONTO APLICADO <= 0) ANTES
de acumularse en memoria, de modo que el pico de RAM sea menor a
"leer todo de una vez + concatenar". El filtro replica exactamente
la condición usada luego en preprocesamiento.limpiar_datos, así que
aplicarlo dos veces es idempotente y no cambia el resultado final,
solo reduce el trabajo y la memoria en el camino.
"""
import time
import logging
import pandas as pd
from pathlib import Path
from config import CHUNK_SIZE, COL_FECHA, COL_FECHA_NACIMIENTO, COL_UNIDADES, COL_MONTO

logger = logging.getLogger(__name__)

# Columnas mínimas requeridas para que el resto del pipeline funcione.
_COLUMNAS_ESPERADAS = [
    COL_FECHA, "CANAL", "SKU", "PRODUCTO", COL_UNIDADES,
    "PORCENTAJE DESCUENTO", COL_MONTO, "BOLETA", "LOCAL",
    "CODIGO CLIENTE", "RUN CLIENTE", "NOMBRES", "APELLIDOS",
    COL_FECHA_NACIMIENTO, "GENERO",
]

_DTYPE_MAPA = {
    "SKU": str,
    "UNIDADES": "Int64",
    "PORCENTAJE DESCUENTO": float,
    "MONTO APLICADO": float,
    "BOLETA": "Int64",
    "LOCAL": "Int64",
    "GENERO": "Int64",
}


def _validar_columnas(columnas_csv):
    """
    Verifica que las columnas esperadas existan en el CSV real.
    Si falta alguna (ej. por un typo de tilde, "GÉNERO" vs "GENERO"),
    lo advierte explícitamente en vez de fallar en silencio más adelante.
    """
    faltantes = [c for c in _COLUMNAS_ESPERADAS if c not in columnas_csv]
    if faltantes:
        logger.warning(
            f"Columnas esperadas no encontradas en el CSV: {faltantes}. "
            f"Columnas reales detectadas: {list(columnas_csv)}"
        )


def _filtrar_chunk_valido(chunk):
    """
    Filtro temprano por chunk: descarta filas con UNIDADES o MONTO
    APLICADO nulos/no positivos. Es el mismo criterio que aplica
    preprocesamiento.limpiar_datos, adelantado aquí para no cargar
    en memoria filas que de todas formas se van a eliminar.
    """
    if COL_UNIDADES in chunk.columns and COL_MONTO in chunk.columns:
        mascara = (chunk[COL_UNIDADES].fillna(0) > 0) & (chunk[COL_MONTO].fillna(0) > 0)
        return chunk.loc[mascara]
    return chunk


def cargar_csv(ruta_csv, verbose=True, filtrar_temprano=True):
    """
    Carga el CSV de ventas usando lectura por fragmentos (chunking).

    Parameters
    ----------
    ruta_csv : str | Path
        Ruta al archivo CSV.
    verbose : bool
        Si True, imprime progreso de carga en consola.
    filtrar_temprano : bool
        Si True (default), aplica el filtro de filas válidas por
        chunk antes de concatenar, reduciendo el pico de memoria.
        Se puede desactivar si se necesita el CSV crudo sin filtrar
        (ej. para auditar cuántas filas se descartan).

    Returns
    -------
    pd.DataFrame
        DataFrame consolidado con todas las filas (filtradas o no).
    """
    ruta = Path(ruta_csv)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta}")

    t_inicio = time.time()

    chunks = []
    filas_leidas = 0
    filas_validas = 0
    n_bloques_con_error = 0

    try:
        lector = pd.read_csv(
            ruta,
            sep=";",
            encoding="latin-1",
            chunksize=CHUNK_SIZE,
            dtype=_DTYPE_MAPA,
            parse_dates=[COL_FECHA, COL_FECHA_NACIMIENTO],
            dayfirst=False,
            low_memory=False,
            on_bad_lines="warn",
        )
    except ValueError as e:
        # Puede ocurrir si el CSV no tiene alguna columna del dtype_mapa
        logger.error(f"Error al preparar el lector de CSV: {e}")
        raise

    for i, chunk in enumerate(lector):
        if i == 0:
            _validar_columnas(chunk.columns)

        filas_leidas += len(chunk)

        try:
            if filtrar_temprano:
                chunk = _filtrar_chunk_valido(chunk)
        except Exception as e:
            n_bloques_con_error += 1
            logger.warning(f"Bloque {i+1}: error al filtrar ({e}), se conserva sin filtrar.")

        filas_validas += len(chunk)
        chunks.append(chunk)

        if verbose:
            print(
                f"  Bloque {i+1} cargado - {filas_leidas:,} filas leídas "
                f"- {filas_validas:,} válidas acumuladas",
                end="\r",
            )

    if verbose:
        print()  # salto de línea final para no pisar el próximo print

    if not chunks:
        raise ValueError(f"El archivo {ruta} no contiene filas legibles.")

    df = pd.concat(chunks, ignore_index=True)
    tiempo_carga = time.time() - t_inicio

    pct_descartado = 0.0
    if filas_leidas > 0:
        pct_descartado = (filas_leidas - filas_validas) / filas_leidas * 100

    logger.info(
        f"Carga completa: {filas_leidas:,} filas leídas, {len(df):,} filas finales "
        f"({pct_descartado:.2f}% descartadas por filtro temprano), "
        f"{n_bloques_con_error} bloque(s) con error de filtrado, "
        f"tiempo={tiempo_carga:.2f}s"
    )

    # Se guarda como atributo del DataFrame para que main.py pueda
    # reportarlo sin cambiar la firma de la función (evita romper
    # el resto del pipeline que espera un único valor de retorno).
    df.attrs["tiempo_carga_seg"] = round(tiempo_carga, 4)
    df.attrs["filas_leidas_csv"] = filas_leidas
    df.attrs["pct_descartado_carga"] = round(pct_descartado, 2)

    return df


def resumen_inicial(df):
    return {
        "n_filas": len(df),
        "n_columnas": len(df.columns),
        "duplicados": int(df.duplicated().sum()),
        "canales_unicos": df["CANAL"].unique().tolist() if "CANAL" in df.columns else [],
        "locales_unicos": sorted(df["LOCAL"].dropna().unique().tolist()) if "LOCAL" in df.columns else [],
        "tiempo_carga_seg": df.attrs.get("tiempo_carga_seg"),
        "filas_leidas_csv": df.attrs.get("filas_leidas_csv"),
        "pct_descartado_carga": df.attrs.get("pct_descartado_carga"),
    }