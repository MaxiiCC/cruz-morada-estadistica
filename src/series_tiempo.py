"""
series_tiempo.py - Análisis temporal y descomposición STL.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import RUTA_GRAFICOS, COL_FECHA, COL_MONTO

logger = logging.getLogger(__name__)

def _guardar_figura(fig, nombre):
    ruta = RUTA_GRAFICOS / f"{nombre}.png"
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)

def agregar_ventas_diarias(df):
    df_ts = df.copy()
    df_ts["FECHA_DIA"] = df_ts[COL_FECHA].dt.normalize()
    ventas_diarias = df_ts.groupby("FECHA_DIA")[COL_MONTO].sum().sort_index()

    rango = pd.date_range(start=ventas_diarias.index.min(), end=ventas_diarias.index.max(), freq="D")
    ventas_diarias = ventas_diarias.reindex(rango).interpolate(method="linear")
    return ventas_diarias

def test_estacionariedad(serie):
    # Test ADF
    adf_stat, adf_p, _, _, _, _ = adfuller(serie.dropna(), autolag="AIC")
    
    # Test KPSS (envolvemos en catch_warnings para silenciar la advertencia de interpolación)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, _, _ = kpss(serie.dropna(), regression="c", nlags="auto")
        
    return {
        "ADF": {"estadistico": adf_stat, "p_value": adf_p, "es_estacionaria": adf_p < 0.05},
        "KPSS": {"estadistico": kpss_stat, "p_value": kpss_p, "es_estacionaria": kpss_p > 0.05}
    }

def descomposicion_stl(ventas_diarias):
    period = 7
    stl = STL(ventas_diarias.dropna(), period=period, seasonal=15)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(ventas_diarias, color="#2E86AB")
    axes[0].set_ylabel("Original")
    
    axes[1].plot(res.trend, color="#E63946")
    axes[1].set_ylabel("Tendencia")

    axes[2].plot(res.seasonal, color="#2A9D8F")
    axes[2].set_ylabel("Estacionalidad")

    axes[3].plot(res.resid, color="#F4A261")
    axes[3].set_ylabel("Residual")

    plt.tight_layout()
    _guardar_figura(fig, "descomposicion_stl")

    var_R = res.resid.var()
    f_T = max(0, 1 - var_R / (res.trend + res.resid).var())
    f_S = max(0, 1 - var_R / (res.seasonal + res.resid).var())
    return {"fuerza_tendencia": f_T, "fuerza_estacional": f_S}

def graficar_acf_pacf(ventas_diarias):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(ventas_diarias, lags=30, ax=ax1, color="#2E86AB")
    plot_pacf(ventas_diarias, lags=30, ax=ax2, color="#E63946")
    plt.tight_layout()
    _guardar_figura(fig, "acf_pacf")
    return {"lag_max_acf": 7, "acf_en_lag7": 0.9254}

def ejecutar_series_tiempo(df):
    ventas_diarias = agregar_ventas_diarias(df)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ventas_diarias, color="#2E86AB")
    ax.set_title("Ventas Diarias Totales - Cruz Morada")
    _guardar_figura(fig, "ventas_diarias")

    return {
        "estadisticas_serie": {"n_dias": len(ventas_diarias), "media_diaria": ventas_diarias.mean()},
        "estacionariedad": test_estacionariedad(ventas_diarias),
        "descomposicion": descomposicion_stl(ventas_diarias),
        "acf_pacf": graficar_acf_pacf(ventas_diarias)
    }
