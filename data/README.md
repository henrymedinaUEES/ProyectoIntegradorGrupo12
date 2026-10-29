# Datos

Dataset transaccional de un retail farmacéutico, 5 cortes
espaciados 120 días (C1-C5), etiqueta de deserción por umbral personalizado
(percentil 90 de los intervalos entre compras de cada cliente, piso 14 días
/ techo 180 días). Detalle completo del diseño y las reglas no negociables
en `PLAN_DE_ACCION.md` y `CONTEXTO_PROYECTO.md` (Proyecto de Claude "MODELO
DE PREDICCIÓN DE DESERCIÓN").

**Ningún dato personal de `DIM_CLIENTE`** (nombre, documento, teléfono,
correo, dirección) está en ninguno de estos archivos — regla no negociable
#9 del plan de acción.

## `raw/`

- `ds_desercion_real.parquet` — el dataset construido por el SQL simplificado
  (9 CTEs, 16 columnas), tal como sale de la base, sin ningún preprocesamiento.
  2.417.698 filas (clientes × cortes). Generado por un notebook de recolección
  fuera de esta carpeta estructurada (ver nota en `README.md` de la raíz).

## `processed/`

Generados por `notebooks/02_preprocesamiento.ipynb`:

| Archivo | Contenido |
|---|---|
| `ds_entrenable.parquet` | Dataset completo tras excluir censurados (2.035.200 filas): columnas de control (`segmento_frecuencia`, `es_comprador_cronico`, `split`, ...) + features + `churn`. Fuente para el desglose de métricas por segmento/crónico |
| `XA1_{train,val,test}.parquet` | Features para árboles, categóricas con one-hot (18 columnas) |
| `XA2_{train,val,test}.parquet` | Features para árboles, categóricas nativas de LightGBM (14 columnas) — **la que usa el modelo de producción** |
| `XB_{train,val,test}.parquet` | Features para la regresión logística: winsorizadas, log con signo, escaladas y one-hot (18 columnas) |
| `y_{train,val,test}.parquet` | Etiqueta `churn` (0/1) de cada split |

Partición: **por cliente** (nunca por fila) y **out-of-time** (C1-C3 = train,
C4 = validación, C5 = test). Filas: train 796.044 / val 68.164 / test 66.421.
Los transformadores (agrupación de categorías raras, winsorización, escalado)
se ajustaron **solo con train** — ver `preprocesamiento.py` / `src/preprocesamiento.py`
y la sección 5 de `notebooks/02_preprocesamiento.ipynb`.
