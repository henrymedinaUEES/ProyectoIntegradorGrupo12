# Informe de evaluación final — modelo de producción

Generado el 26/09/2026 a partir de `notebooks/05_evaluacion.ipynb` y
`src/evaluate.py`. Resume la evaluación única sobre test (C5) del modelo
elegido: **LightGBM (A2, categóricas nativas)**.

## Métricas sobre test (C5, una sola pasada)

| Métrica | Valor |
|---|---|
| PR-AUC | 0,6875 |
| Precision@1% | 0,9774 |
| Lift@1% | 2,43 |
| Precision@5% | 0,9121 |
| Lift@5% | 2,27 |
| Precision@10% | 0,8403 |
| Lift@10% | 2,09 |
| Precision@20% | 0,7403 |
| Lift@20% | 1,84 |
| Precision@30% | 0,6748 |
| Lift@30% | 1,68 |

Comparación contra los demás niveles de la escalera (baselines y regresión
logística) en `results/metrics/_tabla_final_test.csv` y en
`docs/arquitectura.md`.

## Desglose por segmento de frecuencia

| Segmento | n | Tasa base | PR-AUC | Precision@10% | Lift@10% |
|---|---|---|---|---|---|
| Baja | 21.090 | 57,16% | 0,7733 | 0,9137 | 1,60 |
| Media | 21.398 | 38,27% | 0,6102 | 0,7509 | 1,96 |
| Alta | 23.933 | 26,85% | 0,5504 | 0,6841 | 2,55 |

Lectura: el PR-AUC absoluto es mayor en el segmento bajo (tasa base más
alta), pero el modelo discrimina mejor **en términos relativos** en el
segmento alto (mayor lift).

## Desglose por comprador crónico

| Grupo | n | Tasa base | PR-AUC | Precision@10% | Lift@10% |
|---|---|---|---|---|---|
| No crónico | 65.791 | 40,10% | 0,6867 | 0,8398 | 2,09 |
| Crónico | 630 (0,9%) | 45,08% | 0,7458 | 0,8571 | 1,90 |

Muestra pequeña en el grupo crónico: leer con cautela (ver
`docs/consideraciones_eticas.md`).

## Análisis de sensibilidad (T3.4)

Reentrenando sin `umbral_propio_dias`: PR-AUC de validación cae de 0,6975 a
0,6840 (**-1,94%**). El modelo no depende críticamente de esa columna — no
está memorizando la regla de etiquetado.

## Explicabilidad (SHAP, T3.7)

Orden de importancia: `recency_dias` > `umbral_propio_dias` >
`n_dias_activo_12m` > `tendencia_gasto`. `es_comprador_cronico` y
`pct_gasto_cronico_12m` tienen importancia casi nula — limitación declarada,
no resuelta (ver `docs/consideraciones_eticas.md`).

## Pendiente antes de producción

- T3.5 — Calibración de probabilidades.
- T3.8 — Análisis de supervivencia.

Gráficos de soporte en `results/figures/` (lift por decil, SHAP bar/beeswarm,
waterfalls individuales, coeficientes de la logística).
