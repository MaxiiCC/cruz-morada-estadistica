"""
inferencia.py - Pruebas de hipótesis e inferencia estadística.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr, kruskal
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from config import (
    ALPHA, RUTA_GRAFICOS,
    COL_CANAL, COL_LOCAL, COL_MONTO, COL_UNIDADES, COL_DESCUENTO,
    COL_EDAD, COL_FRECUENCIA_COMPRA
)

logger = logging.getLogger(__name__)


def _guardar_figura(fig, nombre):
    ruta = RUTA_GRAFICOS / f"{nombre}.png"
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)


def h1_app_vs_web(df):
    canales = df[COL_CANAL].unique().tolist()

    if "APP" not in canales or "WEB" not in canales:
        logger.warning(
            f"H1: no se encontraron canales 'APP' y 'WEB' explícitos entre {canales}. "
            f"Se usa un fallback posicional, revisa que el resultado siga siendo interpretable."
        )
    canal_app = "APP" if "APP" in canales else (canales[0] if len(canales) > 0 else None)
    canal_web = "WEB" if "WEB" in canales else (canales[1] if len(canales) > 1 else None)

    if canal_app is None or canal_web is None:
        raise ValueError(f"H1: no hay suficientes canales distintos para comparar. Canales: {canales}")

    grupo_app = df[df[COL_CANAL] == canal_app][COL_MONTO].dropna()
    grupo_web = df[df[COL_CANAL] == canal_web][COL_MONTO].dropna()

    if len(grupo_app) == 0 or len(grupo_web) == 0:
        raise ValueError(f"H1: uno de los grupos ({canal_app}, {canal_web}) no tiene datos de MONTO.")

    U, p = mannwhitneyu(grupo_app, grupo_web, alternative="greater")
    rechaza_H0 = p < ALPHA

    fig, ax = plt.subplots(figsize=(9, 5))
    datos_plot = pd.DataFrame({
        "Monto (CLP)": pd.concat([grupo_app, grupo_web]),
        "Canal": [canal_app] * len(grupo_app) + [canal_web] * len(grupo_web)
    })
    sns.boxplot(data=datos_plot, x="Canal", y="Monto (CLP)", hue="Canal", legend=False, showfliers=False, ax=ax)
    ax.set_title(f"H1 - Monto: {canal_app} vs {canal_web}")
    _guardar_figura(fig, "h1_app_vs_web")

    return {
        "canal_a": canal_app, "canal_b": canal_web,
        "p_value": p, "rechaza_H0": rechaza_H0, "U_statistic": U,
        "tamano_efecto": U / (len(grupo_app) * len(grupo_web))
    }


def h2_descuento_unidades(df):
    df_h2 = df[[COL_DESCUENTO, COL_UNIDADES]].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    muestra_plot = df_h2.sample(min(2000, len(df_h2)), random_state=42)
    ax.scatter(muestra_plot[COL_DESCUENTO], muestra_plot[COL_UNIDADES], alpha=0.3)

    if df_h2[COL_UNIDADES].nunique() == 1:
        logger.warning("H2: UNIDADES es constante en los datos, no se puede ajustar una regresión no degenerada.")
        ax.axhline(df_h2[COL_UNIDADES].iloc[0], color="red")
        ax.set_title("H2 - Descuento vs Unidades (Constante, sin varianza)")
        _guardar_figura(fig, "h2_descuento_unidades")
        return {
            "p_value": 1.0, "rechaza_H0": False, "rho_spearman": 0.0,
            "beta_1": 0.0, "R2": 0.0, "nota": "UNIDADES es constante en los datos, no aplica regresión."
        }

    X = add_constant(df_h2[COL_DESCUENTO])
    y = df_h2[COL_UNIDADES].astype(float)
    modelo = OLS(y, X).fit()
    beta_1 = modelo.params[COL_DESCUENTO]
    p_value_beta = modelo.pvalues[COL_DESCUENTO]
    r2 = modelo.rsquared

    rho, p_rho = spearmanr(df_h2[COL_DESCUENTO], df_h2[COL_UNIDADES])

    x_linea = np.linspace(df_h2[COL_DESCUENTO].min(), df_h2[COL_DESCUENTO].max(), 100)
    y_linea = modelo.params["const"] + beta_1 * x_linea
    ax.plot(x_linea, y_linea, color="red", linewidth=2, label=f"OLS (β₁={beta_1:.3f})")
    ax.legend()
    ax.set_title(f"H2 - Descuento vs Unidades (R²={r2:.4f}, p={p_value_beta:.4f})")
    ax.set_xlabel("Porcentaje de Descuento")
    ax.set_ylabel("Unidades")
    _guardar_figura(fig, "h2_descuento_unidades")

    return {
        "p_value": p_value_beta,
        "rechaza_H0": p_value_beta < ALPHA,
        "rho_spearman": rho,
        "p_value_spearman": p_rho,
        "beta_1": beta_1,
        "R2": r2
    }


def h3_edad_ticket(df):
    df_h3 = df[[COL_EDAD, COL_MONTO]].dropna()
    mayores = df_h3[df_h3[COL_EDAD] >= 60][COL_MONTO]
    menores = df_h3[df_h3[COL_EDAD] < 60][COL_MONTO]

    if len(mayores) == 0 or len(menores) == 0:
        raise ValueError("H3: no hay datos suficientes en alguno de los dos grupos etarios.")

    U, p = mannwhitneyu(mayores, menores, alternative="greater")

    fig, ax = plt.subplots(figsize=(9, 5))
    datos_plot = pd.DataFrame({
        "Monto (CLP)": pd.concat([mayores, menores]),
        "Grupo": ["Senior"] * len(mayores) + ["Joven"] * len(menores)
    })
    sns.boxplot(data=datos_plot, x="Grupo", y="Monto (CLP)", hue="Grupo", legend=False, showfliers=False, ax=ax)
    ax.set_title("H3 - Monto: Senior (60+) vs Joven (<60)")
    _guardar_figura(fig, "h3_edad_ticket")

    return {"p_value": p, "rechaza_H0": p < ALPHA, "U_statistic": U, "r_efecto": U / (len(mayores) * len(menores))}


def h4_local_monto(df):
    grupos_df = list(df.groupby(COL_LOCAL))
    grupos_validos = [grp[COL_MONTO].dropna().values for _, grp in grupos_df if len(grp) >= 5]
    n_excluidos = len(grupos_df) - len(grupos_validos)
    if n_excluidos > 0:
        logger.info(f"H4: se excluyeron {n_excluidos} local(es) con menos de 5 observaciones.")

    if len(grupos_validos) < 2:
        raise ValueError("H4: no hay suficientes locales con datos para comparar (mínimo 2).")

    H, p = kruskal(*grupos_validos)
    return {"p_value": p, "rechaza_H0": p < ALPHA, "H_statistic": H, "n_locales_comparados": len(grupos_validos)}


def h5_frecuencia_descuento(df):
    descuento_prom = df.groupby("CODIGO CLIENTE")[COL_DESCUENTO].mean()
    freq_compra = df.groupby("CODIGO CLIENTE")[COL_FRECUENCIA_COMPRA].first()

    datos = pd.concat([freq_compra, descuento_prom], axis=1).dropna()
    if len(datos) < 3:
        raise ValueError("H5: no hay suficientes clientes con datos completos para correlacionar.")

    rho, p = spearmanr(datos[COL_FRECUENCIA_COMPRA], datos[COL_DESCUENTO])
    return {"p_value": p, "rechaza_H0": p < ALPHA, "rho_spearman": rho, "n_clientes": len(datos)}


def ejecutar_inferencia(df):
    return {
        "H1_app_vs_web": h1_app_vs_web(df),
        "H2_descuento_unidades": h2_descuento_unidades(df),
        "H3_edad_ticket": h3_edad_ticket(df),
        "H4_local_monto": h4_local_monto(df),
        "H5_frecuencia_descuento": h5_frecuencia_descuento(df)
    }