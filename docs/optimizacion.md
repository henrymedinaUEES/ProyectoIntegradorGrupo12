# Optimización de hiperparámetros y validación (Workshop S5)

Detalle de `notebooks/04_optimizacion.ipynb` (T3.3 y T3.4 de
`PLAN_DE_ACCION.md`), más la evaluación estadística robusta, la sensibilidad
de hiperparámetros con Optuna y las pruebas de estrés agregadas el
17/10/2026 para cumplir con el contenido de la Semana 5 ("Validación en
Contextos Reales o Simulados") y con el criterio "Optimización de
Hiperparámetros" de la rúbrica del proyecto final 2026.

## Método (búsqueda sobre el dataset completo, configuración de producción)

Búsqueda **aleatoria** (no exhaustiva) de 12 configuraciones por variante
(A1 y A2), misma semilla y mismo espacio de búsqueda en ambas para que la
comparación sea justa:

| Hiperparámetro | Valores explorados |
|---|---|
| `num_leaves` | 15, 31, 63, 127 |
| `learning_rate` | 0,03, 0,05, 0,08, 0,10 |
| `min_child_samples` | 20, 50, 100, 200 |
| `subsample` | 0,7, 0,8, 0,9, 1,0 |
| `colsample_bytree` | 0,6, 0,8, 1,0 |
| `reg_lambda` | 0,0, 0,1, 1,0, 5,0 |

`n_estimators=800` con **early stopping** (`patience=30`) sobre PR-AUC de
**validación únicamente** — nunca sobre test, regla no negociable #2. El
resto de hiperparámetros no explorados quedan en su valor por defecto de
`lightgbm`.

### Resultado — configuración ganadora (A2, dataset completo)

```
num_leaves        = 127
learning_rate     = 0.05
min_child_samples = 20
subsample         = 0.8
colsample_bytree  = 1.0
reg_lambda        = 0.0
```

PR-AUC de validación = 0,6975 (mejor iteración: 56, de un techo de 800).
Congelados en `src/model.py` (`MEJORES_HIPERPARAMETROS`) y reproducibles
corriendo `python src/train.py`.

### Decisión A1 vs. A2

Empate técnico: PR-AUC de validación 0,6975 en ambas variantes, diferencia
solo en la 4ª cifra decimal — ni siquiera alcanza a ser una diferencia
significativa dado el tamaño del set de validación. Se eligió **A2**
(categóricas nativas) por simplicidad de pipeline (14 columnas vs. 18 de A1,
sin necesidad de expandir one-hot ante categorías nuevas de método de pago),
una decisión explícitamente documentada como de mantenibilidad, no de
desempeño.

### Regresión logística (nivel 2, referencia)

Grid de 6 configuraciones (L2 con `lbfgs`, C en {0,01; 0,1; 1; 10}; L1 con
`liblinear`, C en {0,1; 1}) dio PR-AUC de validación prácticamente idéntico
en las seis (~0,6836) — el pipeline B ya deja las features bien
comportadas, así que la regularización aporta poco. Congelado: L2, C=0,01.

### Análisis de sensibilidad de `umbral_propio_dias` (T3.4, no opcional)

Reentrenando LightGBM A2 **sin** la columna `umbral_propio_dias` (única
columna del bloque ligado a la definición de la etiqueta en el esquema
simplificado):

- PR-AUC de validación **con** la columna: 0,6975
- PR-AUC de validación **sin** la columna: 0,6840
- Caída relativa: **1,94%**

**Conclusión:** el rendimiento no se derrumba al quitar la columna. El
modelo retiene casi todo su poder predictivo con el resto de features —
evidencia de que no está solo memorizando la regla de etiquetado.

## Diagnóstico de overfitting / underfitting

Hecho como validación complementaria (ver `arquitectura.md` para el resumen):
**veredicto ÓPTIMO**, sin evidencia de sobreajuste ni subajuste en la
configuración de producción. Reporte completo en
[`diagnostic_report_overfitting.pdf`](diagnostic_report_overfitting.pdf).

---

## Evaluación estadística robusta (17/10/2026)

El diseño del proyecto usa una partición fija por cliente y **fuera de
tiempo** (C1-C3 train, C4 validación, C5 test, tocado una sola vez) — no
folds aleatorios i.i.d. Por eso la validación cruzada repetida tal como la
describe la Semana 5 no se aplica literalmente: mezclar folds temporales
violaría la regla no negociable de partición out-of-time (regla #1/#3 de
`PLAN_DE_ACCION.md`). En su lugar se usó **bootstrap sobre el conjunto de
test ya congelado** (n=2.000 remuestras) para estimar la incertidumbre del
PR-AUC, y **McNemar** sobre predicciones a umbral top-10% para una prueba de
significancia pareada. Ninguno de los dos modelos comparados se volvió a
entrenar: ambos ya estaban congelados (`best_model.pkl`, `model_v1.pkl`).

| Modelo | PR-AUC (test) | IC 95% (bootstrap) |
|---|---|---|
| Nivel 0 — recencia | 0,6403 | [0,6343 – 0,6465] |
| Regresión logística | 0,6701 | [0,6643 – 0,6761] |
| **LightGBM A2** | **0,6875** | **[0,6819 – 0,6929]** |

Diferencia pareada (misma remuestra bootstrap) LightGBM vs. cada rival:

| Comparación | Diferencia media de PR-AUC | IC 95% | ¿Evidencia de mejora? |
|---|---|---|---|
| LightGBM A2 vs. regresión logística | +0,0173 | [0,0151 – 0,0194] | Sí (el IC no cruza 0) |
| LightGBM A2 vs. nivel 0 (recencia) | +0,0471 | [0,0439 – 0,0502] | Sí (el IC no cruza 0) |

Prueba de significancia (McNemar exacto, umbral top-10%, mismo tamaño de
campaña para ambos modelos):

| Comparación | Solo LightGBM acierta | Solo el rival acierta | p-valor |
|---|---|---|---|
| LightGBM A2 vs. regresión logística | 1.485 | 1.189 | 1,13 × 10⁻⁸ |
| LightGBM A2 vs. nivel 0 (recencia) | 2.900 | 2.018 | 2,21 × 10⁻³⁶ |

**Conclusión:** la mejora de LightGBM sobre ambos baselines es estadísticamente
significativa y no es un artefacto de la variabilidad muestral — ningún
intervalo de confianza se solapa, y ambos p-valores son muy inferiores a
0,05. Resultados completos en
`results/metrics/evaluacion_estadistica_robusta.json` y figura en
`results/figures/pr_auc_ci_comparacion.png`.

## Sensibilidad de hiperparámetros con Optuna (Workshop S5, 17/10/2026)

La búsqueda de producción (arriba) usó una búsqueda aleatoria manual de 12
configuraciones sobre el dataset completo. Para el análisis de sensibilidad
que pide explícitamente el Workshop 2 de la Semana 5 (partial dependence,
ranking de importancia de hiperparámetros y análisis de interacciones) se
corrió un estudio adicional con **Optuna** (40 pruebas, muestreador TPE,
mismo espacio de búsqueda de T3.3) sobre una **submuestra de 150.000 filas de
train** (de las 796.044 disponibles), para poder completar las 40 pruebas en
tiempos razonables. **Esta corrida es solo para el análisis de sensibilidad:
no reemplaza ni cambia la configuración de producción**, que sigue siendo la
de la sección anterior, elegida sobre el dataset completo.

Mejor PR-AUC(val) sobre la submuestra: 0,6970 — prácticamente idéntico al
0,6975 obtenido con el dataset completo, lo que confirma que la submuestra es
representativa para este análisis.

### Ranking de importancia de hiperparámetros

| Hiperparámetro | Importancia relativa |
|---|---|
| `subsample` | 57,2% |
| `reg_lambda` | 29,2% |
| `learning_rate` | 5,7% |
| `min_child_samples` | 4,2% |
| `colsample_bytree` | 1,9% |
| `num_leaves` | 1,8% |

Figura: `results/figures/optuna_importancia_hiperparametros.png`.

### Cómo interpretar (partial dependence / slice)

Figura: `results/figures/optuna_slice_hiperparametros.png`. `subsample` y
`reg_lambda` muestran curvas con pendiente marcada — el valor elegido sí
importa. `num_leaves` y `colsample_bytree` muestran curvas casi planas en el
rango explorado: **se podrían fijar en un valor razonable sin perder
desempeño**, liberando presupuesto de búsqueda para los hiperparámetros que
sí importan. Esto es consistente con el hallazgo de `analisis_datos.md` de
que ninguna variable individual tiene correlación fuerte con `churn`: el
modelo depende más de **cómo combina evidencia entre árboles** (`subsample`,
regularización) que de la complejidad de un árbol individual (`num_leaves`).

### Análisis de interacciones

Figura: `results/figures/optuna_contour_num_leaves_min_child.png` (contorno
`num_leaves` × `min_child_samples`). No se observa un valle diagonal marcado:
el efecto de cada hiperparámetro sobre el PR-AUC es aproximadamente
independiente del otro en el rango explorado, es decir, no hace falta
ajustarlos en conjunto con una malla fina.

Trials completos (parámetros + PR-AUC de cada una de las 40 pruebas) en
`results/metrics/optuna_trials_sensibilidad.csv`; resumen en
`results/metrics/optuna_sensibilidad_resumen.json`.

## Pruebas de estrés / validación en escenarios simulados (Workshop 1, 17/10/2026)

Corridas sobre el modelo de producción (`best_model.pkl`, sin reentrenar) y el
conjunto de **validación** (nunca test, para no romper la regla de tocarlo
una sola vez). Baseline: PR-AUC = 0,6975, precision@10% = 0,852, lift@10% =
2,04.

| Escenario | Cómo se simuló | Degradación de PR-AUC |
|---|---|---|
| Ruido gaussiano | σ relativo a la desviación estándar de cada feature: 0,1 / 0,2 / 0,5 / 1,0 | 1,8% / 4,1% / 12,0% / 20,4% |
| Datos faltantes | Enmascarar (NaN) 10% / 30% / 50% de las celdas numéricas | 8,2% / 16,8% / 21,2% |
| Desbalance extremo | Submuestrear la clase positiva a tasa base 5% / 1% | PR-AUC cae a 0,228 / 0,059 (el ranking sigue siendo mejor que el azar, pero la utilidad práctica baja mucho) |
| Corrimiento de distribución | Desplazar `recency_dias` +15 / +30 / +60 días (clientes que tardan sistemáticamente más en volver) | 0,6% / 3,1% / 9,9% |
| Caída de feature clave | `recency_dias`, `umbral_propio_dias` o `n_dias_activo_12m` fijadas a su mediana | 9,2% / 5,0% / 1,7% |

**Umbral de degradación aceptable propuesto:** una caída de PR-AUC mayor al
10% en producción debería disparar una revisión del pipeline de datos antes
de confiar en la lista generada. Bajo ese criterio, el modelo **tolera bien**
ruido moderado (σ ≤ 0,2), hasta 10% de datos faltantes, y corrimientos de
recencia de hasta 30 días; pero es **sensible** a: más de 30% de datos
faltantes, desbalances mucho más extremos que el real (no es un riesgo
esperado, el desbalance real del proyecto es ~39%), y a perder por completo
la señal de `recency_dias` (la feature más importante según SHAP). **Plan de
mitigación:** si en producción se detecta más de 10% de valores faltantes en
una corrida, o un corrimiento de recencia sostenido, el pipeline debería
alertar antes de publicar la lista, en vez de generarla igual con menor
confiabilidad silenciosa.

Figura: `results/figures/pruebas_estres_degradacion.png`. Resultados
completos en `results/metrics/pruebas_estres.json`.

## Pendiente

**T3.5 — Calibración**, todavía no hecha: isotonic regression o Platt
scaling, con curva de confiabilidad y Brier score, necesaria antes de usar
un umbral de probabilidad (en vez de un ranking top-k) para decidir a quién
contactar.
