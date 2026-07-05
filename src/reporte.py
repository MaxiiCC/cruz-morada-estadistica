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
    except:
        return str(valor)

def _fmt_pct(valor):
    try:
        return f"{float(valor)*100:.1f}%".replace(".", ",")
    except:
        return str(valor)

def _fmt_p(valor):
    try:
        v = float(valor)
        if v < 0.001:
            return "<0,001"
        return str(round(v, 4)).replace(".", ",")
    except:
        return str(valor)

def generar_resultados_txt(res):
    lineas = []
    ahora = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    lineas.append("RESULTADOS DEL ANÁLISIS ESTADÍSTICO — CRUZ MORADA")
    lineas.append(f"Generado: {ahora}")
    lineas.append(f"Semilla (CPYD_SEED): {SEED}")
    lineas.append("="*70)

    # 1. Preprocesamiento
    lineas.append("\n==============================================================")
    lineas.append("1. PREPROCESAMIENTO Y LIMPIEZA")
    lineas.append("==============================================================")
    pre = res.get("preprocesamiento", {})
    lineas.append(f"  Filas finales del dataset: {_fmt_num(pre.get('n_filas_final', 0), 0)}")
    
    # 2. EDA
    lineas.append("\n==============================================================")
    lineas.append("2. ANÁLISIS EXPLORATORIO (EDA)")
    lineas.append("==============================================================")
    eda = res.get("eda", {})
    desc = eda.get("descriptiva", {})
    for col, stats_col in desc.items():
        lineas.append(f"  [{col}] Media={_fmt_num(stats_col.get('media',0))} | Mediana={_fmt_num(stats_col.get('mediana',0))}")
    
    # Normalidad
    norm = eda.get("normalidad", {})
    lineas.append("\n  TESTS DE NORMALIDAD:")
    for col, v in norm.items():
        lineas.append(f"    {col}: p={_fmt_p(v.get('p_value',''))} -> {'Normal' if v.get('es_normal') else 'No normal'}")

    # Chi-cuadrado
    chi2 = eda.get("chi_cuadrado", {})
    if "chi2" in chi2:
        lineas.append(f"\n  CHI-CUADRADO (CANAL vs LOCAL): p={_fmt_p(chi2.get('p_value'))} | Cramér's V={_fmt_num(chi2.get('V_cramer'))}")

    # ANOVA
    anova = eda.get("anova", {})
    if "F_statistic" in anova:
        lineas.append(f"  Kruskal-Wallis (MONTO por CANAL): p={_fmt_p(anova.get('p_value_kruskal'))} -> {'Significativo' if anova.get('rechaza_H0') else 'No'}")

    # 3. Inferencia
    lineas.append("\n==============================================================")
    lineas.append("3. INFERENCIA ESTADÍSTICA")
    lineas.append("==============================================================")
    inf = res.get("inferencia", {})
    for h_name, h_val in inf.items():
        lineas.append(f"  [{h_name}] p={_fmt_p(h_val.get('p_value'))} -> {'Rechaza H0 ✓' if h_val.get('rechaza_H0') else 'No rechaza H0 ✗'}")

    # 4. Modelado OLS
    lineas.append("\n==============================================================")
    lineas.append("4. MODELADO — REGRESIÓN OLS")
    lineas.append("==============================================================")
    mod = res.get("modelado", {})
    if mod:
        lineas.append(f"  R² ajustado: {_fmt_num(mod.get('R2_ajustado',0), 4)}")
        lineas.append(f"  RMSE (test): {_fmt_num(mod.get('RMSE_test',0))} CLP")
        lineas.append(f"  MAE  (test): {_fmt_num(mod.get('MAE_test',0))} CLP")
        
        # VIF
        vifs = mod.get("vif", [])
        lineas.append("\n  VIF (Top 5 Multicolinealidad):")
        for v in vifs[:5]:
            lineas.append(f"    {v['Variable']}: VIF={_fmt_num(v['VIF'])} ({v['Evaluacion']})")

    # 5. Series de Tiempo
    lineas.append("\n==============================================================")
    lineas.append("5. SERIES DE TIEMPO")
    lineas.append("==============================================================")
    ts = res.get("series_tiempo", {})
    est = ts.get("estadisticas_serie", {})
    if est:
        lineas.append(f"  N° Días Analizados: {_fmt_num(est.get('n_dias', 0), 0)}")
        lineas.append(f"  Media Diaria: {_fmt_num(est.get('media_diaria', 0))} CLP")
    
    descomp = ts.get("descomposicion", {})
    if descomp:
        lineas.append(f"  Fuerza Tendencia: {_fmt_num(descomp.get('fuerza_tendencia'))}")
        lineas.append(f"  Fuerza Estacional: {_fmt_num(descomp.get('fuerza_estacional'))}")

    # 6. Paralelismo
    lineas.append("\n==============================================================")
    lineas.append("6. PROCESAMIENTO PARALELO")
    lineas.append("==============================================================")
    par = res.get("paralelismo", {})
    if par:
        lineas.append(f"  Núcleos usados: {par.get('n_nucleos')}")
        lineas.append(f"  Tiempo secuencial: {_fmt_num(par.get('tiempo_secuencial'), 4)}s")
        lineas.append(f"  Tiempo paralelo: {_fmt_num(par.get('tiempo_paralelo'), 4)}s")
        lineas.append(f"  Speedup: {_fmt_num(par.get('speedup'))}x")

    lineas.append("\n" + "="*70)
    lineas.append("FIN DEL REPORTE")
    lineas.append("="*70)

    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))

def configurar_logging():
    logger_raiz = logging.getLogger()
    logger_raiz.setLevel(logging.INFO)
    
    fh = logging.FileHandler(RUTA_LOG, encoding="utf-8", mode="w")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger_raiz.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    logger_raiz.addHandler(ch)
    return logger_raiz
