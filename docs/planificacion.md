# Planificación

## Problema y objetivos

Predecir la deserción (*churn*) de clientes de un retail farmacéutico para que
Marketing pueda priorizar a quién contactar, en un contexto **no contractual**:
no hay una fecha de "cancelación" explícita como en telco o suscripciones, así
que hay que definir qué significa "desertar" a partir del propio comportamiento
de compra de cada cliente.

Objetivos concretos:

1. Definir y justificar con evidencia medida (no supuestos) qué es un cliente
   "desertor" en un negocio sin contrato.
2. Construir un dataset temporal, sin fuga de información, a partir de datos
   transaccionales reales.
3. Entrenar y comparar una escalera de modelos (regla de negocio → RFM →
   regresión logística → LightGBM), aceptando IA más compleja solo si mejora
   con evidencia estadística a lo anterior.
4. Evaluar el modelo con rigor (métricas del dominio, intervalos de confianza,
   pruebas de significancia, pruebas de estrés) y con consideraciones éticas
   explícitas.
5. Entregar una interfaz funcional para que Marketing use la lista de riesgo.

## Justificación de la relevancia

El cliente no avisa que se fue: espacia sus compras hasta desaparecer. Cuando
Marketing detecta la inactividad, el cliente suele ya comprar en otra cadena —
el costo de cambio es casi nulo. Anticipar el riesgo antes de que ocurra permite
actuar en la ventana corta en la que todavía es posible retener al cliente.

## Alcance

**Incluye:**
- Retail farmacéutico, dataset transaccional real (`FACT_VENTA_*` / `DIM_*`).
- Cobertura: clientes con 4 o más visitas en 12 meses (~26 % de los clientes
  con actividad; el resto queda fuera por tener 1-3 compras — limitación
  declarada, ver `consideraciones_eticas.md`).
- 5 cortes espaciados 120 días (C1-C5), partición **por cliente** y
  **fuera de tiempo** (nunca cortes mezclados al azar).
- Evaluación estadística robusta (bootstrap, significancia), análisis de sesgo
  por segmento (Fairlearn) y pruebas de estrés sobre el modelo de producción.
- Una interfaz de usuario funcional (Streamlit/Gradio/Flask) para consultar la
  lista priorizada.

**No incluye (fuera de alcance, declarado):**
- Análisis de incrementalidad / *uplift* (a quién *conviene* contactar, no solo
  a quién tiene riesgo alto) — requiere datos de campañas con grupo de control.
- Explicar la causa exacta de la deserción (fin de tratamiento médico vs. fuga
  comercial producen el mismo patrón — limitación central del proyecto).
- Predecir compradores esporádicos (menos de 4 compras/año).
- Escribir de vuelta a la base de datos de origen: la credencial de conexión es
  de solo lectura (regla no negociable #10).

## Cronograma (planificado vs. real)

| Fase | Planificado | Real | Estado |
|---|---|---|---|
| Ideación, propuesta y requerimientos | 12–18 sept 2026 | 12–17 sept 2026: definición del problema, calibración de parámetros con evidencia medida (diagnósticos Q1–Q9) | Cerrado |
| Fase 0 — Desbloquear el esquema de datos | 18–20 sept 2026 | 19 sept 2026: auto-detección de tablas, regla de PDVs válidos (`DIM_BODEGA`), SQL del dataset optimizado y luego simplificado (9 CTEs, 16 columnas) | Cerrado (P1/P3/P4 quedan como validaciones menores, no bloqueantes) |
| Fase 1 — Generar el dataset | 20–22 sept 2026 | 19–20 sept 2026: dataset generado (2.417.698 filas), notebook dividido en recolección + EDA/prep | Cerrado |
| Fase 2 — EDA | 22–24 sept 2026 | 20 sept 2026: censura y cobertura validadas; variación de tasa de `churn` entre cortes (6,6 p.p.) identificada y documentada como hallazgo a investigar | Cerrado, con un hallazgo abierto no bloqueante (ver `analisis_datos.md`) |
| Fase 3 — Modelado | 24–27 sept 2026 | 26 sept 2026: baselines, regresión logística, LightGBM (A1/A2), sensibilidad del umbral, SHAP, evaluación única sobre test; reorganización en `PROYECTO_ESTRUCTURADO/` | Cerrado |
| Validación metodológica adicional (Semana 5) | 29 sept – 17 oct 2026 | 29 sept 2026: diagnóstico de overfitting/underfitting. 17 oct 2026: evaluación estadística robusta (bootstrap + McNemar), análisis de sesgo (Fairlearn), pruebas de estrés, sensibilidad de hiperparámetros con Optuna, documentación reescrita | Cerrado |
| Fase 4 — Entrega: interfaz, repositorio, video | 17–26 oct 2026 | En curso | **Pendiente** |
| Pruebas finales y video pitch/preguntas | 26–27 oct 2026 | — | **Pendiente** |

El proyecto corrió prácticamente en línea con lo planificado: el único
corrimiento real fue dedicar tiempo adicional (29 sept – 17 oct) a la validación
metodológica (overfitting, rigor estadístico, sesgo, pruebas de estrés) que no
estaba en el cronograma original de la tesis, pero que sí exige la rúbrica del
proyecto final 2026. Eso deja menos margen del inicialmente previsto para
construir la interfaz y grabar los videos, concentrado ahora en la última
semana y media (17–27 oct).

## Recursos necesarios

| Recurso | Detalle |
|---|---|
| Datos | Acceso de solo lectura a la base SQL Server del retail farmacéutico (`.env` con credenciales, nunca versionado). Sin ese acceso, el plan B es replicar el pipeline sobre el dataset público **Online Retail II (UCI)** — retail no contractual, análogo al caso de estudio. |
| Hardware | Una laptop estándar es suficiente: el entrenamiento de LightGBM sobre ~800.000 filas × 14 columnas toma segundos; no se necesitó GPU ni un servidor dedicado. |
| Software | Python 3.10+, `scikit-learn`, `lightgbm`, `shap`, `fairlearn`, `optuna`, `pandas`/`numpy`/`pyarrow`, `pytest` (ver `requirements.txt`); Jupyter para los notebooks; `pyodbc`/`sqlalchemy` solo para la recolección (fuera de esta carpeta estructurada). |
| Equipo humano | Dos integrantes (ver [Autores y contribuciones](../README.md#autores-y-contribuciones) en el README): uno a cargo de datos/modelado, otro de desarrollo/interfaz. |

## Riesgos identificados y mitigación

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Se corta el acceso a los datos de origen (credenciales, cambios de permisos) | Baja | Alto | Dataset y splits ya cacheados en `data/processed/`; plan B documentado con Online Retail II (UCI) si hiciera falta reconstruir desde cero. |
| Validaciones de esquema pendientes (P1, P3, P4: valores de `estado`, `producto_cronico`, cardinalidad de `id_linea1`) resultan distintas a lo asumido | Media | Medio — alguna feature podría quedar en cero o mal definida sin que el pipeline falle (fallan en silencio) | Quedaron documentadas explícitamente como pendientes no bloqueantes en `analisis_datos.md`; se recomienda confirmarlas antes de poner el modelo en producción real. |
| Interfaz de usuario y video no se completan a tiempo para el 26–27 oct | Media | Alto — son dos de los componentes obligatorios de la entrega final | Quedan como las dos únicas tareas pendientes del proyecto, con una semana y media de margen y sin dependencias técnicas bloqueantes (el modelo y los datos ya están listos para conectarse a una interfaz). |
| El modelo no logra una mejora significativa sobre los baselines de negocio | Baja (ya descartado) | Alto si ocurriera: el hallazgo sería "no hacía falta IA" | Ya evaluado: LightGBM supera a la regla de recencia y a la regresión logística con intervalos de confianza que no se solapan y significancia estadística (McNemar p < 0,0001) — ver `optimizacion.md` y `results/metrics/evaluacion_estadistica_robusta.json`. |
| Sesgo/inequidad entre segmentos de clientes en la lista priorizada | Media (ya detectado) | Medio — afecta a qué clientes llegan las campañas de retención | Cuantificado con Fairlearn (`MetricFrame`) y documentado en `consideraciones_eticas.md`, con una recomendación explícita de mitigación (umbral de selección por segmento en vez de un umbral global). |
| No crear objetos en la base de datos de origen (restricción de solo lectura) | — (regla, no riesgo probabilístico) | Alto si se violara | Ninguna sentencia de creación en SQL en ningún punto del pipeline (regla no negociable #10). |

## Equipo

- **Henry Medina (HM)** — datos y base de datos: diseño del dataset, SQL,
  calibración de parámetros, EDA, modelado, optimización y evaluación.
- **William Andrade (WA)** — desarrollo: explicabilidad (SHAP), reorganización
  del proyecto en la estructura estándar, interfaz de usuario.

## Fases (ver `PLAN_DE_ACCION.md` del Proyecto de Claude para el detalle
tarea por tarea, criterios de aceptación y estado de cada una)

| Fase | Objetivo | Estado (17/10/2026) |
|---|---|---|
| 0 — Desbloquear el esquema | Validar supuestos del SQL que fallan en silencio | P2 resuelta; P1, P3, P4 pendientes (no bloqueantes) |
| 1 — Generar el dataset | SQL simplificado (16 columnas), 5 cortes cacheados | Hecho |
| 2 — EDA | Estabilidad temporal, censura, hallazgos | T2.2 cumplida; T2.1 (variación de tasa entre cortes) documentada, sin resolver |
| 3 — Modelado | Baselines, logística, LightGBM, sensibilidad, SHAP, evaluación final | Hecho (T3.1-T3.4, T3.6, T3.7) |
| Validación metodológica (Semana 5) | Rigor estadístico, sesgo, pruebas de estrés, sensibilidad de hiperparámetros | Hecho (17/10/2026) |
| 4 — Entrega | Interfaz, repositorio GitHub, videos | **Pendiente, a cargo del equipo antes del 26-27 oct 2026** |

## Reglas que no se negocian

1. Filtrar siempre por `periodo` además de por fecha.
2. El test se toca **una sola vez**, al final, tras congelar todo.
3. Partición **por cliente**, no por fila.
4. Nunca `accuracy`. PR-AUC y precision@k con *k* igual al tamaño real de la
   campaña.
5. Métricas siempre desglosadas por segmento de frecuencia y por comprador
   crónico.
6. Nada de SMOTE.
7. Ninguna feature calculada después del corte.
8. Los transformadores se ajustan solo con train.
9. Los datos personales de `DIM_CLIENTE` no salen en ninguna extracción de
   análisis.
10. Ninguna sentencia de creación en SQL (`CREATE TABLE`, `CREATE INDEX`,
    `SELECT ... INTO`, tablas temporales). La credencial de conexión es de
    solo lectura.

Fuente completa y actualizada de estas reglas y del estado de cada tarea:
`PLAN_DE_ACCION.md` y `CONTEXTO_PROYECTO.md` (Proyecto de Claude "MODELO DE
PREDICCIÓN DE DESERCIÓN"). Este archivo es un resumen para quien abra el
repositorio sin acceso a ese Proyecto; ante cualquier diferencia, esos dos
documentos son la fuente de verdad.
