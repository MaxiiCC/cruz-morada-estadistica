# Cruz Morada — Análisis Estadístico de Ventas

Trabajo práctico para la asignatura **Computación Paralela y Distribuida** (UTEM, 2026).

El objetivo del proyecto es procesar y analizar el dataset de ventas de la cadena de farmacias Cruz Morada (~3.2 millones de registros), aplicando técnicas de estadística descriptiva, inferencial y modelado predictivo, junto con un benchmark de procesamiento paralelo para evaluar el speedup obtenido.

## Estructura del proyecto

```
cruz-morada-estadistica/
├── src/
│   ├── main.py                 # Orquestador principal del pipeline
│   ├── config.py               # Constantes, rutas y semilla (CPYD_SEED)
│   ├── carga_datos.py          # Lectura del CSV por chunks
│   ├── preprocesamiento.py     # Limpieza, outliers IQR, variables derivadas
│   ├── eda.py                  # Estadística descriptiva, normalidad, correlación
│   ├── inferencia.py           # Pruebas de hipótesis H1-H5
│   ├── modelado.py             # Regresión OLS + diagnósticos
│   ├── series_tiempo.py        # Descomposición STL, ACF/PACF, estacionariedad
│   ├── paralelismo.py          # Benchmark secuencial vs paralelo
│   └── reporte.py              # Generación de resultados.txt y logs
├── data/                       # Colocar aquí ventas_completas.csv
├── output/                     # Se genera automáticamente al ejecutar
│   ├── resultados.txt
│   ├── log.txt
│   └── graficos/               # Histogramas, boxplots, STL, etc.
├── requirements.txt
├── .env.example                # Plantilla de configuración (copiar a .env)
└── .gitignore
```

## Requisitos

- Python 3.10 o superior
- pip

## Instalación y ejecución

1. Clonar el repositorio:
```bash
git clone https://github.com/MaxiiCC/cruz-morada-estadistica.git
cd cruz-morada-estadistica
```

2. Crear el entorno virtual e instalar dependencias:
```bash
python -m venv venv

# En Windows (PowerShell):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1

# En Linux/Mac:
# source venv/bin/activate

pip install -r requirements.txt
```

3. Configurar la semilla:
```bash
# Copiar la plantilla de variables de entorno
cp .env.example .env        # Linux/Mac
Copy-Item .env.example .env # Windows PowerShell
```

4. Colocar el dataset `ventas_completas.csv` dentro de la carpeta `data/`.

5. Ejecutar:
```bash
python src/main.py
```

El pipeline tarda aproximadamente 4-5 minutos dependiendo del hardware. Al finalizar se generan los archivos de salida en `output/`.

## Librerías principales

| Librería | Uso |
|---|---|
| Pandas / NumPy | Manejo de datos y operaciones numéricas |
| SciPy | Tests estadísticos (Mann-Whitney, KS, Chi², Kruskal-Wallis) |
| Statsmodels | Regresión OLS, diagnósticos (Breusch-Pagan, RESET, VIF), ACF/PACF |
| Scikit-learn | Train/test split, métricas (RMSE, MAE), StandardScaler |
| Matplotlib / Seaborn | Visualizaciones |
| multiprocessing | Benchmark de procesamiento paralelo |

## Hipótesis evaluadas

- **H1**: El ticket promedio en APP es mayor que en WEB (Mann-Whitney)
- **H2**: El porcentaje de descuento afecta las unidades vendidas (Regresión / Spearman)
- **H3**: Los clientes mayores de 60 años gastan más que los menores de 60 (Mann-Whitney)
- **H4**: Existen diferencias significativas en el monto entre locales (Kruskal-Wallis)
- **H5**: La frecuencia de compra se correlaciona con el descuento promedio obtenido (Spearman)

## Semilla

La semilla se lee desde la variable de entorno `CPYD_SEED` (por defecto 42). Esto garantiza la reproducibilidad de todos los resultados que involucran aleatoriedad (muestreo, train/test split, bootstrap).

## Integrantes

| Nombre |
|---|---|
| Martin Cerda | 
| Maximiliano Campos | 
