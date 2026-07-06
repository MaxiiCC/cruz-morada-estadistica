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
from statsmodels.tsa.stattools import adfuller, kpss, acf
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
    ventas_diarias = ventas_diarias.reindex(rango)

    # limit_direction="both" para también rellenar NaN en los bordes
    # (primer/último día) que interpolate() por defecto deja sin cubrir.
    n_faltantes = int(ventas_diarias.isna().sum())
    ventas_diarias = ventas_diarias.interpolate(method="linear", limit_direction="both")
    if n_faltantes > 0:
        logger.info(f"Ventas diarias: se interpolaron {n_faltantes} día(s) sin transacciones registradas.")

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
    n = len(ventas_diarias.dropna())
    if n < 2 * period + 1:
        logger.warning(
            f"Serie demasiado corta ({n} días) para una descomposición STL confiable "
            f"con period={period} (se recomiendan al menos {2*period + 1} observaciones). "
            f"Se intenta de todas formas, pero interpreta el resultado con cautela."
        )

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
    var_trend_resid = (res.trend + res.resid).var()
    var_seasonal_resid = (res.seasonal + res.resid).var()

    f_T = max(0, 1 - var_R / var_trend_resid) if var_trend_resid > 0 else 0.0
    f_S = max(0, 1 - var_R / var_seasonal_resid) if var_seasonal_resid > 0 else 0.0
    return {"fuerza_tendencia": round(float(f_T), 4), "fuerza_estacional": round(float(f_S), 4)}


def graficar_acf_pacf(ventas_diarias, nlags=30):
    """
    Grafica ACF/PACF y retorna los valores REALES calculados (antes esta
    función retornaba {\"lag_max_acf\": 7, \"acf_en_lag7\": 0.9254} hardcodeado
    sin importar los datos de entrada).
    """
    serie = ventas_diarias.dropna()
    # ACF admite hasta n-1 lags, pero PACF exige nlags < 50% del tamaño
    # muestral (restricción propia de statsmodels). Se usa el más
    # restrictivo de los dos para que ambos gráficos sean consistentes.
    nlags_acf = min(nlags, len(serie) - 1)
    nlags_pacf = min(nlags, max(int(len(serie) / 2) - 1, 0))
    nlags_efectivo = min(nlags_acf, nlags_pacf)

    if nlags_efectivo < 1:
        logger.warning("Serie demasiado corta para calcular ACF/PACF de forma confiable.")
        return {"lag_max_acf": None, "acf_en_lag7": None}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(serie, lags=nlags_efectivo, ax=ax1, color="#2E86AB")
    plot_pacf(serie, lags=nlags_efectivo, ax=ax2, color="#E63946")
    plt.tight_layout()
    _guardar_figura(fig, "acf_pacf")

    valores_acf = acf(serie, nlags=nlags_efectivo, fft=True)
    # Se ignora el lag 0 (autocorrelación consigo misma == 1) al buscar el máximo
    lag_max = int(np.argmax(valores_acf[1:]) + 1)
    acf_lag7 = float(valores_acf[7]) if nlags_efectivo >= 7 else None

    return {
        "lag_max_acf": lag_max,
        "acf_en_lag_max": round(float(valores_acf[lag_max]), 4),
        "acf_en_lag7": round(acf_lag7, 4) if acf_lag7 is not None else None
    }


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