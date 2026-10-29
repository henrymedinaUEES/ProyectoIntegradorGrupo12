# Predicción de deserción de clientes — retail farmacéutico

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![LightGBM](https://img.shields.io/badge/modelo-LightGBM-orange)
![Estado](https://img.shields.io/badge/estado-en%20desarrollo-yellow)

Sistema de IA que predice la **deserción de clientes** (churn no contractual) de un
retail farmacéutico a partir de su historial transaccional, y entrega una lista de
clientes ordenada por riesgo para que el equipo de Marketing priorice a quién
contactar. Es el proyecto final de la Maestría en Inteligencia Artificial, con
alcance de tesis.

## Tabla de contenidos

1. [Descripción del problema](#descripción-del-problema)
2. [Dataset](#dataset)
3. [Metodología](#metodología)
4. [Resultados](#resultados)
5. [Instalación y uso](#instalación-y-uso)
6. [Interfaz de usuario](#interfaz-de-usuario)
7. [Estructura del proyecto](#estructura-del-proyecto)
8. [Consideraciones éticas](#consideraciones-éticas)
9. [Autores y contribuciones](#autores-y-contribuciones)
10. [Licencia](#licencia)
11. [Agradecimientos y referencias](#agradecimientos-y-referencias)
12. [Videos](#videos)

## Descripción del problema

En un retail farmacéutico el cliente no avisa que se fue: no hay contrato que
cancelar ni baja que registrar, simplemente espacia sus compras hasta desaparecer.
Cuando Marketing detecta la inactividad y lanza una campaña de retención, con
frecuencia el cliente ya compra en otra cadena — el costo de cambio es casi nulo y
la ventana de intervención es corta.

Este es un problema de **churn no contractual**: no existe un evento de baja
observable, así que la deserción no se mide directamente, se *construye* a partir
del propio ritmo de compra de cada cliente (ver [`docs/arquitectura.md`](docs/arquitectura.md)
para la definición completa). El sistema estima, para cada cliente activo en una
fecha dada, la probabilidad de que interrumpa su ritmo habitual de compra dentro de
los 120 días siguientes.

**Por qué importa:** permite actuar antes de perder al cliente, en vez de
reaccionar después. **Usuarios objetivo:** el equipo de Marketing del retail
farmacéutico, que usa la lista priorizada para decidir a quién y en qué orden
contactar con campañas de retención.

**El sistema no hace:** no decide a quién *conviene* contactar (eso requeriría
modelos de efecto incremental con datos de campañas y grupo de control, fuera de
alcance), no explica la causa de la deserción (fin de tratamiento médico y fuga
comercial producen el mismo patrón — limitación central, ver
[Consideraciones éticas](#consideraciones-éticas)), y no cubre a compradores
esporádicos (menos de 4 compras al año no tienen ritmo estimable).

## Dataset

Dataset transaccional **real y privado** de un retail farmacéutico (no público, no
compartible — contiene operaciones de un negocio real). Por esa razón no hay un
link de descarga; el dataset crudo vive en `data/raw/` dentro de este repositorio
solo como referencia de esquema, y las credenciales de conexión a la base de
origen nunca se versionan (ver `.gitignore`).

- **Tamaño:** 2.417.698 filas (clientes × 5 cortes temporales), 2.035.200 tras
  excluir clientes censurados.
- **Cobertura:** clientes con 4 o más visitas en 12 meses (~26 % de los clientes
  con actividad).
- **Variables:** 13 features numéricas de comportamiento (recencia, frecuencia,
  monto, canasta farmacéutica, antigüedad) + 1 categórica (método de pago
  principal). **No contiene ningún dato personal** (nombre, documento, teléfono,
  correo, dirección quedan excluidos por diseño).
- **Etiqueta:** `churn` — definida por un umbral de inactividad personalizado por
  cliente (percentil 90 de sus propios intervalos entre compras), no por una regla
  fija. Detalle completo en [`docs/analisis_datos.md`](docs/analisis_datos.md) y
  [`data/README.md`](data/README.md).

## Metodología

- **Modelo:** LightGBM (variante A2, categóricas nativas), elegido sobre una
  escalera de 4 niveles (regla de recencia → RFM simple → regresión logística →
  LightGBM), comparando siempre con los mismos niveles anteriores antes de aceptar
  una IA más compleja.
- **Partición:** por cliente **y** fuera de tiempo (*out-of-time*): 3 cortes de
  entrenamiento, 1 de validación, 1 de test — el test se toca una sola vez.
- **Preprocesamiento:** doble pipeline (crudo para árboles, escalado/winsorizado
  para la regresión logística), ajustado solo con entrenamiento.
- **Optimización de hiperparámetros:** búsqueda aleatoria sobre el dataset
  completo para la configuración de producción, más un estudio adicional con
  Optuna (40 pruebas) para el análisis de sensibilidad (partial dependence,
  importancia y efectos de interacción entre hiperparámetros). Detalle en
  [`docs/optimizacion.md`](docs/optimizacion.md).
- **Métricas:** PR-AUC y precision@k/lift@k (nunca *accuracy*, por el
  desbalance de clases), con intervalos de confianza por bootstrap y pruebas de
  significancia (McNemar) frente a los baselines. Siempre desglosadas por
  segmento de frecuencia y por comprador crónico.
- **Explicabilidad:** SHAP (`TreeExplainer`) sobre el modelo de producción.
- **Validación adicional:** diagnóstico de overfitting/underfitting (ver
  [`docs/diagnostic_report_overfitting.pdf`](docs/diagnostic_report_overfitting.pdf))
  y pruebas de estrés simulando ruido, datos faltantes, desbalance extremo y
  corrimiento de distribución.

## Resultados

Métricas sobre test (C5, una sola pasada), con intervalo de confianza 95% por
bootstrap (n=2.000) y significancia frente a los baselines (ver
[`results/metrics/evaluacion_estadistica_robusta.json`](results/metrics/evaluacion_estadistica_robusta.json)):

| Modelo | PR-AUC | IC 95% | Precision@10% | Lift@10% |
|---|---|---|---|---|
| Nivel 0 — regla de recencia | 0,6403 | [0,6343 – 0,6465] | — | — |
| Regresión logística | 0,6701 | [0,6643 – 0,6761] | — | — |
| **LightGBM A2 (producción)** | **0,6875** | **[0,6819 – 0,6929]** | **0,8403** | **2,09** |

LightGBM supera a ambos baselines con intervalos de confianza que **no se
solapan**, y la diferencia es estadísticamente significativa (McNemar,
p < 0,0001 en ambos casos). Ver la figura
[`results/figures/pr_auc_ci_comparacion.png`](results/figures/pr_auc_ci_comparacion.png)
y el detalle completo, incluido el desglose por segmento y por comprador crónico,
en [`results/reports/informe_evaluacion_final.md`](results/reports/informe_evaluacion_final.md).

## Instalación y uso

Requisitos: Python 3.10+.

```bash
pip install -r requirements.txt
python -m pytest tests/          # pruebas automáticas, deben pasar
python src/train.py              # reentrena LightGBM A2 (~10 s)
python src/evaluate.py           # evalúa sobre test, reproduce las cifras de arriba
```

También se puede seguir el proceso completo abriendo los notebooks en
`notebooks/` en orden (01 → 05). Más detalle paso a paso en
[`docs/manual_usuario.md`](docs/manual_usuario.md).

## Interfaz de usuario

El dashboard local está construido con **Streamlit** y conectado directamente
al modelo LightGBM de `models/best_model.pkl`. Permite probar una muestra del
conjunto de test, cargar un CSV propio, elegir el tamaño de la campaña,
descargar el ranking y evaluar manualmente un cliente.

```bash
streamlit run app/app.py
```

La interfaz se abre normalmente en `http://localhost:8501`. El contrato del
CSV y la operación detallada están en [`docs/manual_usuario.md`](docs/manual_usuario.md).

## Estructura del proyecto

```
PROYECTO_ESTRUCTURADO/
├── README.md              este archivo — carta de presentación del proyecto
├── README_HANDOFF.md      nota técnica de traspaso entre el equipo (histórico de decisiones)
├── requirements.txt       dependencias de Python
├── LICENSE                licencia MIT
├── docs/                  documentación (planificación, EDA, arquitectura, optimización, ética, manual de usuario)
├── data/                  raw/ (dataset crudo) y processed/ (splits ya preprocesados)
├── notebooks/             01_exploracion → 02_preprocesamiento → 03_modelado → 04_optimizacion → 05_evaluacion
├── src/                   módulos reutilizables (data_processing, model, train, evaluate, utils, preprocesamiento)
├── models/                best_model.pkl (LightGBM A2) + versiones anteriores
├── app/                   interfaz de usuario (pendiente, ver arriba)
├── tests/                 pruebas de src/ con pytest
└── results/               figures/, metrics/, reports/
```

Descripción de cada carpeta en detalle dentro de su propio `README.md`
(`data/README.md`, `models/README.md`) o en `docs/arquitectura.md`.

## Consideraciones éticas

El modelo produce un **ranking de riesgo**, no una decisión automática, y no usa
ningún dato personal. Las limitaciones más importantes: no distingue entre fuga
comercial y fin de tratamiento médico, cubre solo ~26 % de los clientes con
actividad, y su ranking global favorece de forma desigual a los clientes de baja
frecuencia frente a los de alta frecuencia al priorizar una lista de contacto
(ver el análisis de sesgo con Fairlearn). Detalle completo, incluidas las cifras
de equidad y el plan de responsabilidad/monitoreo, en
[`docs/consideraciones_eticas.md`](docs/consideraciones_eticas.md).

## Autores y contribuciones

| Integrante | Rol |
|---|---|
| **Henry Medina** | Datos y base de datos: diseño del dataset, SQL, EDA, modelado, optimización de hiperparámetros y evaluación |
| **William Andrade** | Desarrollo: explicabilidad (SHAP), reorganización del proyecto en la estructura estándar, interfaz de usuario |

Proyecto desarrollado como parte de la Maestría en Inteligencia Artificial.

## Licencia

Este proyecto se distribuye bajo la licencia MIT — ver [`LICENSE`](LICENSE) para
el texto completo.

## Agradecimientos y referencias

- Dataset transaccional cedido por el retail farmacéutico para fines académicos.
- Librerías principales: [scikit-learn](https://scikit-learn.org/),
  [LightGBM](https://lightgbm.readthedocs.io/), [SHAP](https://shap.readthedocs.io/),
  [Fairlearn](https://fairlearn.org/), [Optuna](https://optuna.org/).
- Material de referencia de la Semana 5 (Validación en Contextos Reales o
  Simulados) de la Maestría en Inteligencia Artificial, usado como guía
  metodológica para la evaluación estadística robusta, el análisis de sesgo y
  las pruebas de estrés de este proyecto.

## Videos

**Pendientes**, a cargo del equipo antes del cierre del proyecto (26–27 de
octubre de 2026):

- Video pitch (5 minutos): _link pendiente_
- Video de respuestas a preguntas: _link pendiente_
