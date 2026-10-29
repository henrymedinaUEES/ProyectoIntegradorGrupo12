# Modelos

## `best_model.pkl`

**Modelo de producción: LightGBM (variante A2, categóricas nativas).**
Diccionario guardado con `joblib` (formato: `{"modelo", "params",
"pr_auc_val", "columnas", "variante"}`). Elegido en
`notebooks/04_optimizacion.ipynb` por sobre la variante A1 (empate técnico en
PR-AUC, decisión por simplicidad de pipeline) y por sobre la regresión
logística y los dos baselines de negocio (nivel 0: regla de recencia; nivel
1: RFM) — ver `docs/optimizacion.md` para la comparación completa.

- PR-AUC validación: 0.6975 · PR-AUC test (una sola pasada, T3.6): 0.6875
- Hiperparámetros: ver `src/model.py` (`MEJORES_HIPERPARAMETROS`)
- Se reproduce corriendo `python src/train.py` desde la raíz del proyecto
- Se evalúa corriendo `python src/evaluate.py`, o en `notebooks/05_evaluacion.ipynb`

## `model_v1.pkl`

Versión anterior: **regresión logística** (pipeline B, L2, C=0.01). PR-AUC
test = 0.6701 — supera a los baselines pero queda por debajo de LightGBM en
las seis métricas de precision@k/lift. Se conserva como referencia de
interpretabilidad (coeficientes con signo interpretable en las variables
dominantes, ver `results/metrics/_coeficientes_logistica.csv`), no como
candidato a producción.

## `model_v0_lgbm_a1.pkl`

LightGBM variante A1 (categóricas con one-hot, 18 columnas en vez de 14).
PR-AUC de validación prácticamente idéntico a A2 (diferencia en la 4ª cifra
decimal) — se conserva por trazabilidad de la decisión A1 vs. A2, documentada
en `notebooks/04_optimizacion.ipynb`.

## `preprocesamiento.joblib`

`ColumnTransformer` + metadatos del pipeline B (regresión logística):
agrupación de categorías raras, winsorización, log con signo y escalado,
ajustados solo con train. Necesario para transformar datos **nuevos** con
`model_v1.pkl` — los modelos LightGBM (A1/A2) no lo necesitan porque ya
reciben las categóricas nativas o one-hot precomputadas en `data/processed/`.
Para deserializarlo hace falta importar antes las clases de
`src/preprocesamiento.py` (`AgrupadorRaras`, `LimpiaInfinitos`,
`Winsorizador`, `LogConSigno`).
