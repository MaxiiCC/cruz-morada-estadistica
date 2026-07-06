"""
preprocesamiento.py - Limpieza, outliers y nuevas variables.
"""
import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from config import (
    SEED, ALPHA, COL_FECHA, COL_CANAL, COL_UNIDADES, COL_DESCUENTO,
    COL_MONTO, COL_LOCAL, COL_CODIGO_CLIENTE, COL_FECHA_NACIMIENTO,
    COL_MONTO_POR_UNIDAD, COL_EDAD, COL_FRECUENCIA_COMPRA, COLS_NUMERICAS
)

logger = logging.getLogger(__name__)


def limpiar_datos(df):
    df.columns = df.columns.str.strip()
    n_antes = len(df)
    df = df.drop_duplicates()
    n_duplicados = n_antes - len(df)
    if n_duplicados > 0:
        logger.info(f"Se eliminaron {n_duplicados:,} filas duplicadas.")

    if not pd.api.types.is_datetime64_any_dtype(df[COL_FECHA]):
        df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", utc=True)
    if df[COL_FECHA].dt.tz is not None:
        df[COL_FECHA] = df[COL_FECHA].dt.tz_convert("America/Santiago").dt.tz_localize(None)

    if not pd.api.types.is_datetime64_any_dtype(df[COL_FECHA_NACIMIENTO]):
        df[COL_FECHA_NACIMIENTO] = pd.to_datetime(df[COL_FECHA_NACIMIENTO], format="%Y-%m-%d", errors="coerce")

    n_antes_filtro = len(df)
    df = df[(df[COL_UNIDADES].fillna(0) > 0) & (df[COL_MONTO].fillna(0) > 0)].copy()
    n_filtradas = n_antes_filtro - len(df)
    if n_filtradas > 0:
        logger.info(f"Se eliminaron {n_filtradas:,} filas con UNIDADES o MONTO APLICADO <= 0.")

    if COL_CANAL in df.columns:
        df[COL_CANAL] = df[COL_CANAL].str.strip().str.upper()

    return df.reset_index(drop=True)


def _test_mcar_proxy(df, col_con_faltantes, cols_referencia, alpha=ALPHA):
    """
    Proxy del test de MCAR (Missing Completely At Random).

    No existe una implementación directa del test de Little's MCAR en
    scipy/statsmodels, así que se usa un proxy metodológicamente estándar:
    se separa el dataset en dos grupos (con y sin valor faltante en
    col_con_faltantes) y se compara, con Mann-Whitney U, si la distribución
    de otras variables numéricas difiere entre ambos grupos.

    - Si NINGUNA variable de referencia difiere significativamente entre
      los dos grupos -> consistente con MCAR (el faltante no se relaciona
      con nada observable).
    - Si AL MENOS UNA sí difiere significativamente -> evidencia en contra
      de MCAR (probablemente MAR: el faltante depende de otra variable
      observada), y por lo tanto la imputación simple debe justificarse
      con más cuidado en el informe.

    Retorna un dict con el detalle por variable de referencia y una
    conclusión agregada.
    """
    es_faltante = df[col_con_faltantes].isnull()
    n_faltantes = int(es_faltante.sum())

    if n_faltantes == 0:
        return {"n_faltantes": 0, "es_mcar": None, "detalle": {}, "nota": "No hay valores faltantes que testear."}

    if n_faltantes < 5 or (~es_faltante).sum() < 5:
        return {
            "n_faltantes": n_faltantes, "es_mcar": None, "detalle": {},
            "nota": "Muestra insuficiente en alguno de los dos grupos para un test confiable."
        }

    detalle = {}
    alguna_diferencia_significativa = False
    for col_ref in cols_referencia:
        if col_ref not in df.columns or col_ref == col_con_faltantes:
            continue
        grupo_faltante = df.loc[es_faltante, col_ref].dropna()
        grupo_presente = df.loc[~es_faltante, col_ref].dropna()
        if len(grupo_faltante) < 5 or len(grupo_presente) < 5:
            continue
        try:
            stat, p = stats.mannwhitneyu(grupo_faltante, grupo_presente, alternative="two-sided")
        except ValueError:
            continue
        difiere = p < alpha
        alguna_diferencia_significativa = alguna_diferencia_significativa or difiere
        detalle[col_ref] = {"p_value": round(float(p), 4), "difiere_significativamente": bool(difiere)}

    return {
        "n_faltantes": n_faltantes,
        "es_mcar": not alguna_diferencia_significativa,
        "detalle": detalle
    }


def manejar_valores_faltantes(df):
    # --- PORCENTAJE DESCUENTO ---
    # Se cuenta ANTES de imputar (antes este número se reportaba hardcodeado
    # en 0, sin importar cuántos faltantes hubiera realmente).
    cols_referencia_descuento = [c for c in [COL_MONTO, COL_UNIDADES, COL_EDAD] if c in df.columns]
    reporte_mcar_descuento = _test_mcar_proxy(df, COL_DESCUENTO, cols_referencia_descuento)

    n_faltantes_descuento = int(df[COL_DESCUENTO].isnull().sum())
    if n_faltantes_descuento > 0:
        mediana = df[COL_DESCUENTO].median()
        df[COL_DESCUENTO] = df[COL_DESCUENTO].fillna(mediana)
        logger.info(
            f"PORCENTAJE DESCUENTO: {n_faltantes_descuento:,} valores faltantes imputados "
            f"con la mediana ({mediana:.4f}). MCAR proxy: {reporte_mcar_descuento['es_mcar']}."
        )

    # --- FECHA NACIMIENTO ---
    # Aquí se elimina la fila en vez de imputar, porque no hay una forma
    # razonable de "inventar" una fecha de nacimiento sin sesgar EDAD.
    cols_referencia_nacimiento = [c for c in [COL_MONTO, COL_UNIDADES, COL_DESCUENTO] if c in df.columns]
    reporte_mcar_nacimiento = _test_mcar_proxy(df, COL_FECHA_NACIMIENTO, cols_referencia_nacimiento)

    n_faltantes_nacimiento = int(df[COL_FECHA_NACIMIENTO].isnull().sum())
    df = df.dropna(subset=[COL_FECHA_NACIMIENTO])
    if n_faltantes_nacimiento > 0:
        logger.info(
            f"FECHA NACIMIENTO: {n_faltantes_nacimiento:,} filas eliminadas por no tener fecha "
            f"de nacimiento válida. MCAR proxy: {reporte_mcar_nacimiento['es_mcar']}."
        )

    reporte_faltantes = {
        "PORCENTAJE_DESCUENTO": {
            "n_faltantes": n_faltantes_descuento,
            "es_mcar": reporte_mcar_descuento["es_mcar"],
            "detalle_test": reporte_mcar_descuento.get("detalle", {}),
            "metodo": "Imputación por mediana"
        },
        "FECHA_NACIMIENTO": {
            "n_faltantes": n_faltantes_nacimiento,
            "es_mcar": reporte_mcar_nacimiento["es_mcar"],
            "detalle_test": reporte_mcar_nacimiento.get("detalle", {}),
            "metodo": "Eliminación de filas (no se imputa una fecha de nacimiento)"
        }
    }

    return df.reset_index(drop=True), reporte_faltantes


def crear_variables_derivadas(df):
    df[COL_MONTO_POR_UNIDAD] = np.where(df[COL_UNIDADES] > 0, df[COL_MONTO] / df[COL_UNIDADES], np.nan)
    df[COL_EDAD] = ((df[COL_FECHA] - df[COL_FECHA_NACIMIENTO]).dt.days / 365.25).round(1)
    df.loc[(df[COL_EDAD] < 0) | (df[COL_EDAD] > 120), COL_EDAD] = np.nan
    df[COL_FRECUENCIA_COMPRA] = df.groupby(COL_CODIGO_CLIENTE)[COL_FECHA].transform("count")
    return df


def detectar_outliers(df):
    df["ES_OUTLIER"] = False
    detalle_outliers = {}
    for col in [COL_MONTO, COL_DESCUENTO]:
        serie = df[col].dropna()
        Q1 = serie.quantile(0.25)
        Q3 = serie.quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        mascara = (df[col] < lim_inf) | (df[col] > lim_sup)
        df.loc[mascara, "ES_OUTLIER"] = True
        detalle_outliers[col] = {
            "limite_inferior": round(float(lim_inf), 2),
            "limite_superior": round(float(lim_sup), 2),
            "n_outliers": int(mascara.sum())
        }
    df.attrs["detalle_outliers_iqr"] = detalle_outliers
    return df


def estandarizar(df, columnas):
    """
    Estandariza columnas numéricas con StandardScaler.

    Se imputa con la MEDIANA de cada columna antes de escalar (en vez del
    fillna(0) anterior), para ser consistente con la estrategia de
    imputación ya usada en manejar_valores_faltantes. Rellenar con 0 es
    arbitrario y puede distorsionar la escala en variables donde 0 no es
    un valor típico (ej. EDAD, MONTO_POR_UNIDAD).
    """
    scaler = StandardScaler()
    cols_existentes = [c for c in columnas if c in df.columns]

    df_para_escalar = df[cols_existentes].copy()
    medianas = df_para_escalar.median(numeric_only=True)
    df_para_escalar = df_para_escalar.fillna(medianas)

    df[[c + "_NORM" for c in cols_existentes]] = scaler.fit_transform(df_para_escalar)

    parametros = {
        col: {
            "media": float(scaler.mean_[i]),
            "std": float(scaler.scale_[i]),
            "mediana_usada_para_imputar": float(medianas[col])
        }
        for i, col in enumerate(cols_existentes)
    }
    return df, parametros


def preprocesar(df):
    df = limpiar_datos(df)
    df, reporte_faltantes = manejar_valores_faltantes(df)
    df = crear_variables_derivadas(df)
    df = detectar_outliers(df)
    df, params = estandarizar(df, COLS_NUMERICAS)

    return df, {
        "limpieza": {"n_filas_despues_limpieza": len(df), "canales": df[COL_CANAL].unique().tolist()},
        "valores_faltantes": reporte_faltantes,
        "outliers": {
            "MONTO APLICADO": {
                "outliers_total": int(df["ES_OUTLIER"].sum()),
                "pct_outliers": float(df["ES_OUTLIER"].sum() / len(df) * 100)
            },
            "detalle_iqr": df.attrs.get("detalle_outliers_iqr", {})
        },
        "estandarizacion": params,
        "n_filas_final": len(df)
    }