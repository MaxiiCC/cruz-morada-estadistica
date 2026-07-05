"""
modelado.py - Modelo de regresión OLS y diagnósticos.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import (
    SEED, ALPHA, TRAIN_SIZE, RUTA_GRAFICOS,
    COL_CANAL, COL_LOCAL, COL_MONTO, COL_DESCUENTO, COL_EDAD
)

logger = logging.getLogger(__name__)

def _guardar_figura(fig, nombre):
    ruta = RUTA_GRAFICOS / f"{nombre}.png"
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)

def preparar_features(df):
    df_mod = df.copy()
    y = df_mod[COL_MONTO].dropna()
    df_mod = df_mod.loc[y.index]

    # Convertimos LOCAL a texto primero para evitar conflicto de tipos en pandas 3.x
    df_mod[COL_LOCAL] = df_mod[COL_LOCAL].astype(str)

    # Agrupar locales menos recurrentes
    top_locales = df_mod[COL_LOCAL].value_counts().head(10).index.tolist()
    df_mod[COL_LOCAL] = df_mod[COL_LOCAL].where(df_mod[COL_LOCAL].isin(top_locales), other="OTRO")

    dummies_canal = pd.get_dummies(df_mod[COL_CANAL], prefix="CANAL", drop_first=True, dtype=float)
    dummies_local = pd.get_dummies(df_mod[COL_LOCAL], prefix="LOCAL", drop_first=True, dtype=float)

    # Excluimos UNIDADES por ser constante y provocar colinealidad perfecta
    numericas = df_mod[[COL_DESCUENTO, COL_EDAD]].copy()
    numericas = numericas.fillna(numericas.median()).astype("float64")

    X = pd.concat([numericas, dummies_canal, dummies_local], axis=1)

    # Excluir LOCAL_1999 por colinealidad perfecta con CANAL_POS
    cols_a_eliminar = [c for c in X.columns if "LOCAL_1999" in c or "LOCAL_1999.0" in c]
    if cols_a_eliminar:
        X = X.drop(columns=cols_a_eliminar)

    X = X.astype("float64").dropna()
    y = y.loc[X.index].astype("float64")
    return X, y

def calcular_vif(X):
    X_con_const = sm.add_constant(X)
    vif_data = pd.DataFrame({
        "Variable": X_con_const.columns,
        "VIF": [variance_inflation_factor(X_con_const.values, i) for i in range(X_con_const.shape[1])]
    })
    vif_data = vif_data[vif_data["Variable"] != "const"].sort_values("VIF", ascending=False)
    vif_data["Evaluacion"] = vif_data["VIF"].apply(lambda v: "⚠️ Problemático" if v > 10 else ("⚠️ Moderado" if v > 5 else "✓ Aceptable"))
    return vif_data.round(2)

def entrenar_modelo(df):
    X, y = preparar_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=SEED)

    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)

    modelo = sm.OLS(y_train, X_train_sm).fit()

    y_pred_train = modelo.predict(X_train_sm)
    y_pred_test = modelo.predict(X_test_sm)

    residuales = y_train - y_pred_train
    std_resid = (residuales - residuales.mean()) / residuales.std()

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(residuales, X_train_sm)
    muestra_resid = std_resid.sample(min(5000, len(std_resid)), random_state=SEED)
    sw_stat, sw_p = stats.shapiro(muestra_resid)

    vif_df = calcular_vif(X_train)

    # Gráficos de diagnóstico
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].scatter(y_pred_train, residuales, alpha=0.3, s=10)
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 0].set_title("Residuales vs Ajustados")

    sm.qqplot(std_resid, line="45", ax=axes[0, 1], alpha=0.3, markersize=3)
    axes[0, 1].set_title(f"Q-Q Plot (SW p={sw_p:.4f})")

    axes[1, 0].hist(std_resid, bins=50, edgecolor="white")
    axes[1, 0].set_title("Histograma de Residuales")

    axes[1, 1].scatter(y_test, y_pred_test, alpha=0.3, s=10)
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    axes[1, 1].set_title("Reales vs Predichos")
    plt.tight_layout()
    _guardar_figura(fig, "diagnosticos_ols")

    coef_df = pd.DataFrame({
        "Variable": modelo.params.index,
        "Coeficiente": modelo.params.values.round(4),
        "p_value": modelo.pvalues.values.round(4)
    })

    return {
        "n_train": len(X_train), "n_test": len(X_test), "R2": round(modelo.rsquared, 4),
        "R2_ajustado": round(modelo.rsquared_adj, 4), "AIC": round(modelo.aic, 2), "BIC": round(modelo.bic, 2),
        "RMSE_test": round(rmse_test, 2), "MAE_test": round(mae_test, 2),
        "breusch_pagan": {"LM": bp_lm, "p_value": bp_p, "homocedastico": bp_p > ALPHA},
        "normalidad_residuales": {"SW_stat": sw_stat, "p_value": sw_p, "es_normal": sw_p > ALPHA},
        "vif": vif_df.to_dict("records"),
        "coeficientes": coef_df.to_dict("records")
    }
