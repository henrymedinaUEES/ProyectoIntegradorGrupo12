# Análisis exploratorio de datos

Resumen ejecutivo de `notebooks/01_exploracion.ipynb` (secciones 1-4 del EDA
original), más el análisis adicional de correlaciones, outliers y datos
faltantes corrido el 17/10/2026 sobre `data/processed/ds_entrenable.parquet`
(resultados completos en `results/metrics/eda_correlaciones_outliers.json` y
`results/metrics/eda_matriz_correlacion.csv`). El detalle celda por celda, con
todos los gráficos originales, vive en ese notebook; aquí se documentan los
hallazgos con número y las seis visualizaciones de referencia.

## Tamaño y cobertura

- Dataset crudo (`data/raw/ds_desercion_real.parquet`): 2.417.698 filas
  (clientes × 5 cortes).
- Tras excluir censurados (`df_e`, en `data/processed/ds_entrenable.parquet`):
  2.035.200 filas entrenables.
- Cobertura: ~26% de los clientes con actividad (4+ visitas/año). El 69,7%
  restante queda fuera por tener 1-3 compras — ver `consideraciones_eticas.md`.
- Tasa de churn global: 38,95%.

## Visualizaciones del EDA (`results/figures/`)

| Figura | Qué muestra |
|---|---|
| `eda_01_distribucion_recency.png` | Distribución de `recency_dias` por clase — los desertores se concentran en recencias altas, pero hay solape importante, lo que explica que el PR-AUC no sea perfecto con esta sola variable. |
| `eda_02_churn_por_segmento.png` | Tasa de churn por segmento de frecuencia: más alta en el segmento de baja frecuencia, más baja en el de alta frecuencia (consistente con que clientes frecuentes tienen más evidencia reciente de actividad). |
| `eda_03_churn_por_corte.png` | Tasa de churn por corte (C1-C5) — evidencia visual de la inestabilidad temporal documentada abajo. |
| `eda_04_boxplot_monto.png` | `monto_total_12m` por clase, recortado al percentil 99 — los desertores tienden a gastar menos, con una cola larga de outliers en ambas clases. |
| `eda_05_heatmap_correlacion.png` | Correlación de Pearson entre las 12 features numéricas — ningún par supera 0,9 (no hay colinealidad extrema que justifique eliminar features). |
| `eda_06_umbral_por_segmento.png` | Distribución de `umbral_propio_dias` por segmento — confirma visualmente el factor ~7x entre el ritmo del segmento alto y el bajo (ver `CONTEXTO_PROYECTO.md`, sección 4). |

## Correlaciones con la etiqueta

Correlación de Pearson de cada feature numérica con `churn` (todas las
magnitudes son moderadas, ninguna cercana a 1, lo que descarta que el modelo
dependa de una sola variable trivial):

| Feature | Correlación con `churn` |
|---|---|
| `recency_dias` | **+0,369** (la más fuerte, consistente con SHAP) |
| `n_dias_activo_12m` | -0,201 |
| `n_visitas_12m` | -0,185 |
| `n_categorias_distintas_12m` | -0,172 |
| `n_bodegas_distintas_12m` | -0,152 |
| `monto_total_12m` | -0,135 |
| `antiguedad_meses` | -0,067 |
| `umbral_propio_dias` | -0,045 |
| `n_metodos_pago_distintos_12m` | -0,041 |
| `pct_gasto_cronico_12m` | -0,025 |
| `ticket_promedio` | -0,014 |
| `tendencia_gasto` | -0,001 |

Ninguna correlación por sí sola explica el fenómeno (la máxima es 0,369), lo
que justifica usar un modelo no lineal (LightGBM) capaz de capturar
interacciones entre variables en vez de solo efectos aditivos — consistente
con el hallazgo de `optimizacion.md` de que `subsample` y `reg_lambda` (que
controlan cómo el modelo combina evidencia de múltiples árboles) son más
influyentes que `num_leaves` por sí solo.

## Outliers (regla IQR: fuera de [Q1-1.5·IQR, Q3+1.5·IQR])

| Feature | % outliers |
|---|---|
| `pct_gasto_cronico_12m` | 23,54% |
| `n_metodos_pago_distintos_12m` | 23,35% |
| `n_categorias_distintas_12m` | 11,81% |
| `monto_total_12m` | 10,00% |
| `n_visitas_12m` | 9,65% |
| `tendencia_gasto` | 9,18% |
| `n_dias_activo_12m` | 8,60% |
| `ticket_promedio` | 6,71% |
| `recency_dias` | 5,94% |
| `n_bodegas_distintas_12m` | 5,02% |
| `antiguedad_meses` | 0,17% |
| `umbral_propio_dias` | 0,00% (ya acotado por diseño a [14, 180] días) |

Los porcentajes de outliers son altos en `pct_gasto_cronico_12m` y
`n_metodos_pago_distintos_12m` porque ambas son variables con distribución muy
asimétrica (muchos clientes en 0% o en un solo método de pago, y una cola de
clientes con comportamiento distinto), no por errores de captura. **Decisión
de preprocesamiento (ver `arquitectura.md`):** para el pipeline de árboles
(LightGBM, A1/A2) no se recortan los outliers — los árboles son robustos a
valores extremos por construcción. Para el pipeline de regresión logística
(B) sí se aplica winsorización (recorte a percentiles extremos) y
transformación logarítmica con signo, ajustadas solo con train, precisamente
para controlar el efecto de estos outliers sobre un modelo lineal.

## Datos faltantes

Solo una columna tiene nulos: **`tendencia_gasto` (1,96%)** — ocurre cuando el
promedio mensual de los 9 meses base es cero (cliente nuevo con poco
historial) y la razón queda indefinida. El resto de las 13 columnas
(features + `churn`) no tiene ningún nulo. **Manejo:** no se imputa; LightGBM
soporta valores faltantes nativamente (los usa como una categoría más de
división en cada árbol), consistente con la prueba de estrés de
`optimizacion.md` que muestra que el modelo degrada de forma gradual, no
abrupta, ante datos faltantes. Para el pipeline de regresión logística (que sí
requiere datos completos) estos nulos se imputan con la mediana de train,
documentado en `notebooks/02_preprocesamiento.ipynb`.

## Manejo del desbalance de clases

Tasa base de `churn` ≈ 39% (no es un desbalance extremo como en fraude, pero sí
asimétrico). **No se usa SMOTE** (regla no negociable #6): generaría clientes
sintéticos inexistentes y rompería la calibración de probabilidades. En su
lugar: (a) las métricas de evaluación son siempre PR-AUC y precision@k/lift@k,
nunca `accuracy` (regla no negociable #4), que es la forma correcta de medir
cuando las clases están desbalanceadas; y (b) se verificó mediante una prueba
de estrés (`optimizacion.md`) qué tan sensible es el modelo a desbalances aún
más extremos (tasa base forzada a 5% y 1%), encontrando una caída fuerte de
PR-AUC como es esperable — evidencia de que el desbalance real del proyecto
(39%) está lejos del punto donde el modelo deja de ser útil.

## Estabilidad temporal (sección 4.2) — **no cumplida, sin explicar**

La tasa base de `churn` varía entre cortes más de lo esperado:

| Corte | Tasa de churn |
|---|---|
| C1 | 35,27% |
| C2 | 35,79% |
| C3 | 40,27% |
| C4 | 41,87% |
| C5 | 40,11% |

Rango = 6,60 puntos porcentuales, por encima del límite de 5 p.p. que la
propia sección 4.2 usa como criterio de estabilidad (ver `eda_03_churn_por_corte.png`).
C3/C4/C5 (los cortes más recientes) están varios puntos por encima de C1/C2. Dos
hipótesis sin descartar: estacionalidad (los cortes caen en distintas épocas del
año) o un artefacto de la simplificación del SQL (el cambio de `y_evento` — gap
máximo en toda la ventana — a `churn` — solo la primera compra futura — puede
comportarse distinto entre cortes). **Pendiente T2.1 / P10** en
`PLAN_DE_ACCION.md`: investigar antes de escribir la metodología de la tesis,
porque condiciona cómo se interpretan las diferencias entre C1-C2 y C3-C5.

## Censura (sección 4.3) — cumplida

Censura global 15,82% (cercana al 14,7% del diseño anterior), concentrada en
el segmento de baja frecuencia:

| Segmento | % censura |
|---|---|
| Baja (4-6 visitas/año) | 33,54% |
| Media (7-12 visitas/año) | 6,62% |
| Alta (13+ visitas/año) | 0,06% |

Dentro del rango aceptable (10%-20%). Esta concentración estructural en el
segmento bajo es la razón por la que el análisis de supervivencia (T3.8,
pendiente) se considera el diferenciador académico del trabajo: los
censurados llevan información parcial que el enfoque binario descarta.

## Fuga de información (sección 4.11)

Verificado: ninguna columna de etiqueta (`churn`, `censurado`) está dentro de
las features. `umbral_propio_dias` sí participa en la definición de la
etiqueta (es la base del umbral personalizado p90) y se mantiene como
feature — de ahí el análisis de sensibilidad obligatorio en la fase de
modelado (T3.4, ver `optimizacion.md`), no un descarte automático de la
columna.

## Esquema de columnas

16 columnas del SQL simplificado (9 CTEs): columnas de control
(`id_cliente`, `fecha_corte`, `corte_id`, `segmento_frecuencia`,
`censurado`), la etiqueta (`churn`) y 13 features numéricas + 1 categórica
(`metodo_pago_principal_12m`). Detalle completo del mapeo columna por columna
en `CONTEXTO_PROYECTO.md` sección 6 y en `contexto_esquema_tablas.md`
(Proyecto de Claude).

## Validaciones del esquema aún pendientes (Fase 0)

Tres de cuatro validaciones fallan en silencio (el dataset se genera igual,
solo que mal): P1 (valores de `estado`/`tipo_transaccion` que son ventas
reales), P3 (valores de `producto_cronico`/`generico`), P4 (cardinalidad de
`id_linea1`). Ver `PLAN_DE_ACCION.md` Fase 0 para las consultas exactas.
