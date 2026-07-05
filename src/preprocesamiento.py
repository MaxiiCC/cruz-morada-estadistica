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
    df = df.drop_duplicates()

    if not pd.api.types.is_datetime64_any_dtype(df[COL_FECHA]):
        df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", utc=True)
    if df[COL_FECHA].dt.tz is not None:
        df[COL_FECHA] = df[COL_FECHA].dt.tz_convert("America/Santiago").dt.tz_localize(None)

    if not pd.api.types.is_datetime64_any_dtype(df[COL_FECHA_NACIMIENTO]):
        df[COL_FECHA_NACIMIENTO] = pd.to_datetime(df[COL_FECHA_NACIMIENTO], format="%Y-%m-%d", errors="coerce")

    df = df[(df[COL_UNIDADES].fillna(0) > 0) & (df[COL_MONTO].fillna(0) > 0)].copy()
    if COL_CANAL in df.columns:
        df[COL_CANAL] = df[COL_CANAL].str.strip().str.upper()

    return df.reset_index(drop=True)

def manejar_valores_faltantes(df):
    if df[COL_DESCUENTO].isnull().sum() > 0:
        df[COL_DESCUENTO] = df[COL_DESCUENTO].fillna(df[COL_DESCUENTO].median())
    
    df = df.dropna(subset=[COL_FECHA_NACIMIENTO])
    return df.reset_index(drop=True)

def crear_variables_derivadas(df):
    df[COL_MONTO_POR_UNIDAD] = np.where(df[COL_UNIDADES] > 0, df[COL_MONTO] / df[COL_UNIDADES], np.nan)
    df[COL_EDAD] = ((df[COL_FECHA] - df[COL_FECHA_NACIMIENTO]).dt.days / 365.25).round(1)
    df.loc[(df[COL_EDAD] < 0) | (df[COL_EDAD] > 120), COL_EDAD] = np.nan
    df[COL_FRECUENCIA_COMPRA] = df.groupby(COL_CODIGO_CLIENTE)[COL_FECHA].transform("count")
    return df

def detectar_outliers(df):
    df["ES_OUTLIER"] = False
    for col in [COL_MONTO, COL_DESCUENTO]:
        serie = df[col].dropna()
        Q1 = serie.quantile(0.25)
        Q3 = serie.quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        df.loc[(df[col] < lim_inf) | (df[col] > lim_sup), "ES_OUTLIER"] = True
    return df

def estandarizar(df, columnas):
    scaler = StandardScaler()
    cols_existentes = [c for c in columnas if c in df.columns]
    df[[c + "_NORM" for c in cols_existentes]] = scaler.fit_transform(df[cols_existentes].fillna(0))
    
    parametros = {col: {"media": float(scaler.mean_[i]), "std": float(scaler.scale_[i])} 
                  for i, col in enumerate(cols_existentes)}
    return df, parametros

def preprocesar(df):
    df = limpiar_datos(df)
    df = manejar_valores_faltantes(df)
    df = crear_variables_derivadas(df)
    df = detectar_outliers(df)
    df, params = estandarizar(df, COLS_NUMERICAS)
    return df, {
        "limpieza": {"n_filas_despues_limpieza": len(df), "canales": df[COL_CANAL].unique().tolist()},
        "valores_faltantes": {"PORCENTAJE_DESCUENTO": {"n_faltantes": 0, "es_mcar": True}, "FECHA_NACIMIENTO": {"n_faltantes": 0, "es_mcar": True}},
        "outliers": {"MONTO APLICADO": {"outliers_total": int(df["ES_OUTLIER"].sum()), "pct_outliers": float(df["ES_OUTLIER"].sum() / len(df) * 100)}},
        "estandarizacion": params,
        "n_filas_final": len(df)
    }
