# Manual de usuario

Guía para quien va a **usar** los resultados del modelo (por ejemplo,
Marketing) o correr el proyecto sin haberlo construido. No repite el diseño
técnico (ver `arquitectura.md`) ni la optimización (ver `optimizacion.md`).

## Qué hace el modelo

Ordena a los clientes cubiertos (4 o más visitas en 12 meses) de mayor a
menor riesgo de dejar de comprar en los próximos 120 días. No dice "sí" o
"no": da un **score** que sirve para ordenar una lista y decidir cuántos
contactar (top 5%, top 10%, top 20%...) según el tamaño real de la campaña.

**Importante:** el score todavía no está calibrado como una probabilidad
exacta (pendiente T3.5, ver `consideraciones_eticas.md`). Puede usarse con
confianza para **ordenar**, no todavía para leer "72% de probabilidad de
irse" como un número absoluto.

## Cómo leer las métricas de una lista

- **Precision@k%**: de los clientes en el top k% de la lista, qué fracción
  realmente desertó (en la evaluación histórica). Ej.: precision@10% = 0,84
  quiere decir que, de cada 100 clientes en el top 10%, unos 84 efectivamente
  dejaron de comprar.
- **Lift@k%**: cuántas veces mejor que elegir al azar. Lift@10% = 2,09
  significa que el top 10% de la lista tiene el doble de desertores reales
  que un 10% elegido al azar.

Estas cifras están desglosadas por segmento de frecuencia y por comprador
crónico en `notebooks/05_evaluacion.ipynb` y en `results/metrics/`; conviene
mirar el desglose, no solo el número global, antes de decidir el tamaño de
campaña para un segmento específico.

## Cómo explicarle un caso a alguien de negocio (SHAP)

Cada cliente en la lista se puede explicar en una frase, sin tecnicismos.
Ejemplo real (`notebooks/05_evaluacion.ipynb`): *"este cliente compra
típicamente cada dos semanas y lleva varios meses sin volver"* — el modelo
lo marca de alto riesgo porque su recencia (días desde la última compra) es
muy alta **respecto a su propio ritmo habitual**, no respecto a un umbral
fijo para todos.

## Cómo correr el proyecto

1. Instalar dependencias: `pip install -r requirements.txt`.
2. Los datos y el modelo ya vienen generados en `data/processed/` y
   `models/best_model.pkl`. Para reproducir desde cero:
   - `notebooks/01_exploracion.ipynb` → `02_preprocesamiento.ipynb` →
     `03_modelado.ipynb` → `04_optimizacion.ipynb` → `05_evaluacion.ipynb`,
     en ese orden, o
   - `python src/train.py` (entrena con los hiperparámetros ya congelados)
     seguido de `python src/evaluate.py` (evalúa sobre test).
3. `python -m pytest tests/` corre las pruebas automáticas de
   `src/data_processing.py` y `src/model.py`/`src/utils.py`.

## Interfaz web

Desde la raíz del proyecto, con el entorno virtual activo:

```bash
streamlit run app/app.py
```

La interfaz se abre normalmente en `http://localhost:8501` y se organiza en
tres pestañas:

- **Generar ranking**: demostración con datos de test o carga de un archivo
  CSV/Excel propio (validación de sus 14 features), selección del porcentaje
  de clientes que admite la campaña, tabla de ranking con score y nivel de
  riesgo, descarga del resultado completo en Excel (`.xlsx`) y, justo debajo,
  el panel **"Exploración de su base (EDA)"**: un panorama sencillo (pensado
  para un usuario no técnico) de la base recién analizada — totales de
  registros/columnas y % de nulos, nulos por columna (con la etiqueta de %
  sobre cada barra), estadísticas descriptivas, histogramas de las 13
  variables numéricas del modelo, matriz de correlación y distribución del
  método de pago principal (con la cantidad de clientes sobre cada barra).
  Este panel aparece automáticamente, en la misma pantalla, apenas se
  presiona "Analizar clientes" y el ranking se genera correctamente — no es
  necesario cambiar de pestaña para verlo.
- **Evaluar un cliente**: formulario para consultar el riesgo de un caso
  puntual.
- **Acerca del modelo**: métricas de referencia (PR-AUC, precision@10%,
  lift@10%) y advertencias de uso.

El CSV o Excel cargado puede incluir una columna `id_cliente`, que se
conserva en la salida, y debe incluir las 14 columnas de entrada de A2. La
propia pantalla permite descargar una plantilla — en formato **Excel**
(`plantilla_clientes.xlsx`), no CSV. No deben cargarse nombres, documentos,
teléfonos, correos ni direcciones. En el modo de demostración, los datos de
prueba (`data/processed/XA2_test.parquet`) no traen `id_cliente` por sí
solos (el id no es una feature del modelo), así que la interfaz lo recupera
cruzando por posición con `data/processed/ds_entrenable.parquet` (filtrado a
`split == "test"`, que conserva el orden original y el id real). Si ese
archivo no está disponible o el conteo de filas no coincide, se usa en su
lugar un identificador sintético (`DEMO-00001`, ...) para no mostrar un id
incorrecto. El ranking se descarga como archivo Excel (`.xlsx`), no CSV.

La capa backend está en `app/prediction_service.py`: carga una vez
`models/best_model.pkl`, valida tipos y columnas, restaura las categorías de
LightGBM y genera el ranking. El procesamiento se realiza localmente.

## Qué todavía falta

- **Calibración de probabilidades** (T3.5) y **análisis de supervivencia**
  (T3.8). Por eso la interfaz describe la salida como score de ordenamiento,
  no como una probabilidad absoluta.
- Integración automática con SQL Server y autenticación multiusuario para un
  despliegue corporativo; la versión actual recibe Parquet de demostración o
  un CSV cargado por el usuario.
