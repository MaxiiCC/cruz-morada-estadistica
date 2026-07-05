"""
eda.py - Análisis Exploratorio de Datos (Estadísticas y Gráficos).
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, spearmanr, pearsonr

from config import (
    ALPHA, UMBRAL_SHAPIRO, RUTA_GRAFICOS,
    COL_CANAL, COL_LOCAL, COL_MONTO, COL_UNIDADES, COL_DESCUENTO,
    COL_MONTO_POR_UNIDAD, COL_EDAD, COL_FRECUENCIA_COMPRA, COL_GENERO,
    COL_FECHA
)

logger = logging.getLogger(__name__)

def _formatear_numero_chileno(valor, decimales=2):
    if pd.isna(valor):
        return "N/A"
    partes = f"{valor:,.{decimales}f}".split(".")
    entero = partes[0].replace(",", ".")
    decimal = partes[1] if len(partes) > 1 else ""
    return f"{entero},{decimal}" if decimal else entero

def _guardar_figura(fig, nombre):
    ruta = RUTA_GRAFICOS / f"{nombre}.png"
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)

def _test_normalidad(serie, nombre):
    """
    Test de normalidad (SW para muestras pequeñas, KS para muestras grandes).
    """
    serie_limpia = serie.dropna()
    n = len(serie_limpia)

    if n < 3:
        return {"test": "N/A", "estadistico": None, "p_value": None, "es_normal": None, "n": n}

    if n < UMBRAL_SHAPIRO:
        muestra = serie_limpia.sample(min(n, 5000), random_state=42)
        stat, p = stats.shapiro(muestra)
        test_usado = "Shapiro-Wilk"
    else:
        muestra_ks = serie_limpia.sample(min(50000, n), random_state=42)
        media_ks = muestra_ks.mean()
        std_ks = muestra_ks.std()
        stat, p = stats.kstest(muestra_ks.values, stats.norm(loc=media_ks, scale=std_ks).cdf)
        test_usado = "Kolmogorov-Smirnov"

    es_normal = p > ALPHA
    return {
        "variable": nombre, "test": test_usado, "n": n,
        "estadistico": round(float(stat), 4), "p_value": round(float(p), 4),
        "es_normal": es_normal
    }

def estadistica_descriptiva(df):
    cols = [COL_MONTO, COL_UNIDADES, COL_DESCUENTO, COL_MONTO_POR_UNIDAD, COL_EDAD, COL_FRECUENCIA_COMPRA]
    resultados = {}
    for col in cols:
        if col not in df.columns:
            continue
        serie = df[col].dropna()
        resultados[col] = {
            "n": len(serie),
            "media": round(serie.mean(), 2),
            "mediana": round(serie.median(), 2),
            "std": round(serie.std(), 2),
            "asimetria": round(serie.skew(), 4),
            "curtosis": round(serie.kurtosis(), 4),
            "min": round(serie.min(), 2),
            "p5": round(serie.quantile(0.05), 2),
            "p25": round(serie.quantile(0.25), 2),
            "p75": round(serie.quantile(0.75), 2),
            "p95": round(serie.quantile(0.95), 2),
            "max": round(serie.max(), 2),
            "cv": round(serie.std() / (serie.mean() + 1e-10) * 100, 2)
        }
    return resultados

def graficar_histogramas(df):
    cols = [
        (COL_MONTO, "Monto Aplicado (CLP)"),
        (COL_UNIDADES, "Unidades Compradas"),
        (COL_DESCUENTO, "Porcentaje de Descuento (0-1)"),
        (COL_MONTO_POR_UNIDAD, "Monto por Unidad (CLP)"),
        (COL_EDAD, "Edad del Cliente (años)"),
        (COL_FRECUENCIA_COMPRA, "Frecuencia de Compra")
    ]
    resultados_normalidad = {}
    for col, etiqueta in cols:
        if col not in df.columns:
            continue
        serie = df[col].dropna()
        test_res = _test_normalidad(serie, col)
        resultados_normalidad[col] = test_res

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(serie, kde=True, ax=ax, color="#2E86AB", alpha=0.7)
        ax.set_title(f"Distribución de {etiqueta}\n{test_res['test']}: p={test_res['p_value']}")
        ax.axvline(serie.mean(), color="#E63946", linestyle="--", label="Media")
        ax.axvline(serie.median(), color="#2A9D8F", linestyle="-.", label="Mediana")
        ax.legend()
        _guardar_figura(fig, f"hist_{col.lower().replace(' ', '_')}")
    return resultados_normalidad

def graficar_boxplots(df):
    for col_cat, etiqueta_cat, nombre_archivo in [(COL_CANAL, "Canal de Venta", "boxplot_monto_por_canal")]:
        if col_cat not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x=col_cat, y=COL_MONTO, palette="Blues_d", showfliers=False)
        ax.set_title(f"Monto Aplicado por {etiqueta_cat}")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}".replace(",", ".")))
        _guardar_figura(fig, nombre_archivo)

def graficar_correlacion(df):
    cols = [COL_MONTO, COL_UNIDADES, COL_DESCUENTO, COL_MONTO_POR_UNIDAD, COL_EDAD, COL_FRECUENCIA_COMPRA]
    df_num = df[cols].dropna()
    n_muestra = min(50000, len(df_num))
    df_muestra = df_num.sample(n_muestra, random_state=42)

    matriz_r = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    matriz_p = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)

    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if i == j:
                continue
            r, p = pearsonr(df_muestra[col_i], df_muestra[col_j])
            matriz_r.loc[col_i, col_j] = round(r, 3)
            matriz_p.loc[col_i, col_j] = round(p, 4)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matriz_r, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Matriz de Correlación de Pearson")
    _guardar_figura(fig, "matriz_correlacion")
    return {"correlacion": matriz_r.to_dict()}

def chi_cuadrado_canal_local(df):
    tabla = pd.crosstab(df[COL_CANAL], df[COL_LOCAL])
    chi2, p, dof, expected = chi2_contingency(tabla)
    n = tabla.sum().sum()
    V = np.sqrt(chi2 / (n * (min(tabla.shape) - 1)))
    return {"chi2": chi2, "p_value": p, "V_cramer": V, "rechaza_H0": p < ALPHA}

def anova_monto_por_canal(df):
    grupos = [grp[COL_MONTO].dropna().values for _, grp in df.groupby(COL_CANAL) if len(grp) > 1]
    F, p_anova = f_oneway(*grupos)
    H, p_kruskal = stats.kruskal(*grupos)
    return {"F_statistic": F, "p_value_anova": p_anova, "H_kruskal": H, "p_value_kruskal": p_kruskal, "rechaza_H0": p_kruskal < ALPHA}

def ejecutar_eda(df):
    res = {}
    res["descriptiva"] = estadistica_descriptiva(df)
    res["normalidad"] = graficar_histogramas(df)
    graficar_boxplots(df)
    res["correlacion"] = graficar_correlacion(df)
    res["chi_cuadrado"] = chi_cuadrado_canal_local(df)
    res["anova"] = anova_monto_por_canal(df)
    return res
