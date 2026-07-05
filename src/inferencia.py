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
    canal_app = "APP" if "APP" in canales else (canales[0] if len(canales) > 0 else None)
    canal_web = "WEB" if "WEB" in canales else (canales[1] if len(canales) > 1 else None)

    grupo_app = df[df[COL_CANAL] == canal_app][COL_MONTO].dropna()
    grupo_web = df[df[COL_CANAL] == canal_web][COL_MONTO].dropna()

    U, p = mannwhitneyu(grupo_app, grupo_web, alternative="greater")
    rechaza_H0 = p < ALPHA

    fig, ax = plt.subplots(figsize=(9, 5))
    datos_plot = pd.DataFrame({
        "Monto (CLP)": pd.concat([grupo_app, grupo_web]),
        "Canal": [canal_app]*len(grupo_app) + [canal_web]*len(grupo_web)
    })
    sns.boxplot(data=datos_plot, x="Canal", y="Monto (CLP)", showfliers=False, ax=ax)
    ax.set_title(f"H1 - Monto: {canal_app} vs {canal_web}")
    _guardar_figura(fig, "h1_app_vs_web")

    return {"p_value": p, "rechaza_H0": rechaza_H0, "U_statistic": U, "tamano_efecto": U / (len(grupo_app) * len(grupo_web))}

def h2_descuento_unidades(df):
    df_h2 = df[[COL_DESCUENTO, COL_UNIDADES]].dropna()
    if df_h2[COL_UNIDADES].nunique() == 1:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(df_h2[COL_DESCUENTO].sample(min(2000, len(df_h2))), 
                   df_h2[COL_UNIDADES].sample(min(2000, len(df_h2))), alpha=0.3)
        ax.axhline(1.0, color="red")
        ax.set_title("H2 - Descuento vs Unidades (Constante)")
        _guardar_figura(fig, "h2_descuento_unidades")
        return {"p_value": 1.0, "rechaza_H0": False, "rho_spearman": 0.0, "beta_1": 0.0, "R2": 0.0}

def h3_edad_ticket(df):
    df_h3 = df[[COL_EDAD, COL_MONTO]].dropna()
    mayores = df_h3[df_h3[COL_EDAD] >= 60][COL_MONTO]
    menores = df_h3[df_h3[COL_EDAD] < 60][COL_MONTO]
    U, p = mannwhitneyu(mayores, menores, alternative="greater")
    
    fig, ax = plt.subplots(figsize=(9, 5))
    datos_plot = pd.DataFrame({"Monto (CLP)": pd.concat([mayores, menores]), "Grupo": ["Senior"]*len(mayores) + ["Joven"]*len(menores)})
    sns.boxplot(data=datos_plot, x="Grupo", y="Monto (CLP)", showfliers=False, ax=ax)
    _guardar_figura(fig, "h3_edad_ticket")
    return {"p_value": p, "rechaza_H0": p < ALPHA, "U_statistic": U, "r_efecto": U / (len(mayores) * len(menores))}

def h4_local_monto(df):
    grupos = [grp[COL_MONTO].values for _, grp in df.groupby(COL_LOCAL) if len(grp) >= 5]
    H, p = kruskal(*grupos)
    return {"p_value": p, "rechaza_H0": p < ALPHA, "H_statistic": H}

def h5_frecuencia_descuento(df):
    descuento_prom = df.groupby("CODIGO CLIENTE")[COL_DESCUENTO].mean()
    freq_compra = df.groupby("CODIGO CLIENTE")[COL_FRECUENCIA_COMPRA].first()
    rho, p = spearmanr(freq_compra, descuento_prom)
    return {"p_value": p, "rechaza_H0": p < ALPHA, "rho_spearman": rho}

def ejecutar_inferencia(df):
    return {
        "H1_app_vs_web": h1_app_vs_web(df),
        "H2_descuento_unidades": h2_descuento_unidades(df),
        "H3_edad_ticket": h3_edad_ticket(df),
        "H4_local_monto": h4_local_monto(df),
        "H5_frecuencia_descuento": h5_frecuencia_descuento(df)
    }
