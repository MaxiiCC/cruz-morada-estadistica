"""
reporte.py - Generador del archivo resultados.txt y configurador de logs.
"""
import logging
import datetime
from pathlib import Path
from config import RUTA_RESULTADOS, RUTA_LOG, SEED

logger = logging.getLogger(__name__)


def _fmt_num(valor, decimales=2):
    try:
        valor = float(valor)
        if valor != valor:
            return "N/D"
        if decimales == 0:
            return f"{valor:,.0f}".replace(",", ".")
        formateado = f"{valor:,.{decimales}f}"
        partes = formateado.split(".")
        entero = partes[0].replace(",", ".")
        decimal = partes[1] if len(partes) > 1 else ""
        return f"{entero},{decimal}"
    except Exception:
        return str(valor) if valor is not None else "N/D"


def _fmt_pct(valor):
    try:
        return f"{float(valor)*100:.1f}%".replace(".", ",")
    except Exception:
        return str(valor) if valor is not None else "N/D"


def _fmt_p(valor):
    try:
        v = float(valor)
        if v < 0.001:
            return "<0,001"
        return str(round(v, 4)).replace(".", ",")
    except Exception:
        return str(valor) if valor is not None else "N/D"


def _fmt_bool(valor, si="Sí", no="No"):
    if valor is None:
        return "N/D"
    return si if valor else no


def generar_resultados_txt(res):
    lineas = []
    ahora = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    lineas.append("RESULTADOS DEL ANÁLISIS ESTADÍSTICO — CRUZ MORADA")
    lineas.append(f"Generado: {ahora}")
    lineas.append(f"Semilla (CPYD_SEED): {SEED}")
    lineas.append("=" * 70)

    # ==================== 1. PREPROCESAMIENTO ====================
    lineas.append("\n==============================================================")
    lineas.append("1. PREPROCESAMIENTO Y LIMPIEZA")
    lineas.append("==============================================================")
    pre = res.get("preprocesamiento", {})
    lineas.append(f"  Filas finales del dataset: {_fmt_num(pre.get('n_filas_final', 0), 0)}")

    limpieza = pre.get("limpieza", {})
    if limpieza.get("canales"):
        lineas.append(f"  Canales detectados: {', '.join(map(str, limpieza['canales']))}")

    # Valores faltantes (con proxy de test MCAR real, no fabricado)
    faltantes = pre.get("valores_faltantes", {})
    if faltantes:
        lineas.append("\n  VALORES FALTANTES:")
        for col, info in faltantes.items():
            n_falt = info.get("n_faltantes", 0)
            es_mcar = info.get("es_mcar")
            metodo = info.get("metodo", "N/D")
            lineas.append(
                f"    {col}: {_fmt_num(n_falt, 0)} faltantes | "
                f"MCAR (proxy Mann-Whitney): {_fmt_bool(es_mcar, 'Consistente con MCAR', 'Evidencia contra MCAR (posible MAR)')} | "
                f"Método: {metodo}"
            )
            detalle = info.get("detalle_test", {})
            for col_ref, resultado in detalle.items():
                lineas.append(
                    f"      vs. {col_ref}: p={_fmt_p(resultado.get('p_value'))} -> "
                    f"{'Difiere' if resultado.get('difiere_significativamente') else 'No difiere'} significativamente"
                )

    # Outliers (con límites IQR reales, no solo el conteo)
    outliers = pre.get("outliers", {})
    if outliers:
        lineas.append("\n  OUTLIERS (método IQR, 1.5×RIC):")
        info_monto = outliers.get("MONTO APLICADO", {})
        if info_monto:
            lineas.append(
                f"    Total outliers (MONTO APLICADO): {_fmt_num(info_monto.get('outliers_total', 0), 0)} "
                f"({_fmt_pct(info_monto.get('pct_outliers', 0) / 100)})"
            )
        detalle_iqr = outliers.get("detalle_iqr", {})
        for col, lim in detalle_iqr.items():
            lineas.append(
                f"    {col}: límites=[{_fmt_num(lim.get('limite_inferior'))}, "
                f"{_fmt_num(lim.get('limite_superior'))}] -> {_fmt_num(lim.get('n_outliers', 0), 0)} outliers"
            )

    # ==================== 2. EDA ====================
    lineas.append("\n==============================================================")
    lineas.append("2. ANÁLISIS EXPLORATORIO (EDA)")
    lineas.append("==============================================================")
    eda = res.get("eda", {})
    desc = eda.get("descriptiva", {})
    for col, stats_col in desc.items():
        lineas.append(
            f"  [{col}] Media={_fmt_num(stats_col.get('media', 0))} | "
            f"Mediana={_fmt_num(stats_col.get('mediana', 0))} | "
            f"Asimetría={_fmt_num(stats_col.get('asimetria', 0), 4)} | "
            f"Curtosis={_fmt_num(stats_col.get('curtosis', 0), 4)}"
        )

    # Normalidad
    norm = eda.get("normalidad", {})
    lineas.append("\n  TESTS DE NORMALIDAD:")
    for col, v in norm.items():
        lineas.append(
            f"    {col} ({v.get('test', 'N/D')}): p={_fmt_p(v.get('p_value', ''))} -> "
            f"{'Normal' if v.get('es_normal') else 'No normal'}"
        )

    # Chi-cuadrado
    chi2 = eda.get("chi_cuadrado", {})
    if "chi2" in chi2:
        lineas.append(f"\n  CHI-CUADRADO (CANAL vs LOCAL): p={_fmt_p(chi2.get('p_value'))} | Cramér's V={_fmt_num(chi2.get('V_cramer'))}")

    # ANOVA / Kruskal-Wallis
    anova = eda.get("anova", {})
    if "F_statistic" in anova:
        lineas.append(
            f"  ANOVA (MONTO por CANAL): F={_fmt_num(anova.get('F_statistic'))}, p={_fmt_p(anova.get('p_value_anova'))}"
        )
        lineas.append(
            f"  Kruskal-Wallis (MONTO por CANAL): p={_fmt_p(anova.get('p_value_kruskal'))} -> "
            f"{'Significativo' if anova.get('rechaza_H0') else 'No significativo'}"
        )

    # Correlación: se reporta el método usado por par (Pearson/Spearman según normalidad)
    correlacion = eda.get("correlacion", {})
    metodo_usado = correlacion.get("metodo_usado", {})
    if metodo_usado:
        metodos_distintos = {m for fila in metodo_usado.values() for m in fila.values() if m != "-"}
        if metodos_distintos:
            lineas.append(f"\n  MATRIZ DE CORRELACIÓN: método(s) usado(s) según normalidad -> {', '.join(sorted(metodos_distintos))}")
            lineas.append("  (ver output/graficos/matriz_correlacion.png para coeficientes y significancia con *)")

    # ==================== 3. INFERENCIA ====================
    lineas.append("\n==============================================================")
    lineas.append("3. INFERENCIA ESTADÍSTICA")
    lineas.append("==============================================================")
    inf = res.get("inferencia", {})
    for h_name, h_val in inf.items():
        if not h_val:
            lineas.append(f"  [{h_name}] Sin resultado (revisar log).")
            continue
        lineas.append(
            f"  [{h_name}] p={_fmt_p(h_val.get('p_value'))} -> "
            f"{'Rechaza H0 ✓' if h_val.get('rechaza_H0') else 'No rechaza H0 ✗'}"
        )
        # Detalle extra para H2 (regresión), si está disponible
        if "beta_1" in h_val:
            lineas.append(
                f"      β₁={_fmt_num(h_val.get('beta_1'), 4)} | R²={_fmt_num(h_val.get('R2'), 4)} | "
                f"ρ Spearman={_fmt_num(h_val.get('rho_spearman'), 4)}"
            )
        if "nota" in h_val:
            lineas.append(f"      Nota: {h_val['nota']}")

    # ==================== 4. MODELADO OLS ====================
    lineas.append("\n==============================================================")
    lineas.append("4. MODELADO — REGRESIÓN OLS")
    lineas.append("==============================================================")
    mod = res.get("modelado", {})
    if mod:
        lineas.append(f"  Observaciones: train={_fmt_num(mod.get('n_train', 0), 0)} | test={_fmt_num(mod.get('n_test', 0), 0)}")
        lineas.append(f"  R²: {_fmt_num(mod.get('R2', 0), 4)} | R² ajustado: {_fmt_num(mod.get('R2_ajustado', 0), 4)}")
        lineas.append(f"  AIC: {_fmt_num(mod.get('AIC', 0))} | BIC: {_fmt_num(mod.get('BIC', 0))}")
        lineas.append(f"  RMSE (test): {_fmt_num(mod.get('RMSE_test', 0))} CLP")
        lineas.append(f"  MAE  (test): {_fmt_num(mod.get('MAE_test', 0))} CLP")

        lineas.append("\n  DIAGNÓSTICO DE SUPUESTOS:")
        bp = mod.get("breusch_pagan", {})
        if bp:
            lineas.append(
                f"    Homocedasticidad (Breusch-Pagan): p={_fmt_p(bp.get('p_value'))} -> "
                f"{'Homocedástico' if bp.get('homocedastico') else 'Heterocedástico'}"
            )
        norm_resid = mod.get("normalidad_residuales", {})
        if norm_resid:
            lineas.append(
                f"    Normalidad de residuales (Shapiro-Wilk): p={_fmt_p(norm_resid.get('p_value'))} -> "
                f"{'Normal' if norm_resid.get('es_normal') else 'No normal'}"
            )
        lin_reset = mod.get("linealidad_reset", {})
        if lin_reset and lin_reset.get("p_value") is not None:
            lineas.append(
                f"    Linealidad (Ramsey RESET): p={_fmt_p(lin_reset.get('p_value'))} -> "
                f"{'Bien especificado (lineal)' if lin_reset.get('es_lineal') else 'Posible mala especificación'}"
            )

        # Colinealidad eliminada dinámicamente (reemplaza el hardcode de LOCAL_1999)
        colin = mod.get("colinealidad_eliminada", [])
        if colin:
            lineas.append("\n  VARIABLES EXCLUIDAS POR COLINEALIDAD PERFECTA:")
            for c in colin:
                lineas.append(f"    {c['eliminada']} (r={c['r']} con {c['correlacionada_con']})")

        vifs = mod.get("vif", [])
        lineas.append("\n  VIF (Top 5 Multicolinealidad):")
        for v in vifs[:5]:
            lineas.append(f"    {v['Variable']}: VIF={_fmt_num(v['VIF'])} ({v['Evaluacion']})")

    # ==================== 5. SERIES DE TIEMPO ====================
    lineas.append("\n==============================================================")
    lineas.append("5. SERIES DE TIEMPO")
    lineas.append("==============================================================")
    ts = res.get("series_tiempo", {})
    est = ts.get("estadisticas_serie", {})
    if est:
        lineas.append(f"  N° Días Analizados: {_fmt_num(est.get('n_dias', 0), 0)}")
        lineas.append(f"  Media Diaria: {_fmt_num(est.get('media_diaria', 0))} CLP")

    estac = ts.get("estacionariedad", {})
    if estac:
        adf = estac.get("ADF", {})
        kpss_r = estac.get("KPSS", {})
        lineas.append(
            f"  ADF: p={_fmt_p(adf.get('p_value'))} -> "
            f"{'Estacionaria' if adf.get('es_estacionaria') else 'No estacionaria'}"
        )
        lineas.append(
            f"  KPSS: p={_fmt_p(kpss_r.get('p_value'))} -> "
            f"{'Estacionaria' if kpss_r.get('es_estacionaria') else 'No estacionaria'}"
        )

    descomp = ts.get("descomposicion", {})
    if descomp:
        lineas.append(f"  Fuerza Tendencia: {_fmt_num(descomp.get('fuerza_tendencia'))}")
        lineas.append(f"  Fuerza Estacional: {_fmt_num(descomp.get('fuerza_estacional'))}")

    acf_pacf = ts.get("acf_pacf", {})
    if acf_pacf and acf_pacf.get("lag_max_acf") is not None:
        lineas.append(
            f"  Lag con mayor autocorrelación: {acf_pacf.get('lag_max_acf')} "
            f"(ACF={_fmt_num(acf_pacf.get('acf_en_lag_max'), 4)})"
        )
        if acf_pacf.get("acf_en_lag7") is not None:
            lineas.append(f"  ACF en lag 7 (estacionalidad semanal): {_fmt_num(acf_pacf.get('acf_en_lag7'), 4)}")

    # ==================== 6. PARALELISMO ====================
    lineas.append("\n==============================================================")
    lineas.append("6. PROCESAMIENTO PARALELO")
    lineas.append("==============================================================")
    par = res.get("paralelismo", {})
    if par:
        lineas.append(f"  Núcleos usados: {par.get('n_nucleos')}")
        if "n_particiones" in par:
            lineas.append(f"  N° particiones/tareas: {par.get('n_particiones')}")
        lineas.append(f"  Tiempo secuencial: {_fmt_num(par.get('tiempo_secuencial'), 4)}s")
        lineas.append(f"  Tiempo paralelo: {_fmt_num(par.get('tiempo_paralelo'), 4)}s")
        lineas.append(f"  Speedup: {_fmt_num(par.get('speedup'))}x")
        if "eficiencia_pct" in par:
            lineas.append(f"  Eficiencia: {_fmt_num(par.get('eficiencia_pct'))}%")
        if "resultados_consistentes" in par:
            lineas.append(
                f"  Consistencia sec/par: "
                f"{_fmt_bool(par.get('resultados_consistentes'), 'Idénticos ✓', 'Difieren ✗')}"
            )

    lineas.append("\n" + "=" * 70)
    lineas.append("FIN DEL REPORTE")
    lineas.append("=" * 70)

    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))


def configurar_logging():
    logger_raiz = logging.getLogger()
    logger_raiz.setLevel(logging.INFO)

    RUTA_LOG.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(RUTA_LOG, encoding="utf-8", mode="w")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger_raiz.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    logger_raiz.addHandler(ch)
    return logger_raiz