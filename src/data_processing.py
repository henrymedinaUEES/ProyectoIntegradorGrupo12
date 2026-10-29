"""
Funciones de carga de los datos ya procesados (partición por cliente +
out-of-time, agrupacion de categorias raras, pipelines A/A2/B) generados por
notebooks/02_preprocesamiento.ipynb y cacheados en data/processed/.

Este modulo NO recalcula el preprocesamiento desde el dataset crudo: eso vive
en notebooks/02_preprocesamiento.ipynb, que es la fuente de verdad para como
se construyen estos parquets (agrupacion de categorias con menos del 1% en
train, "OTROS"/"DESCONOCIDO", winsorizacion + log con signo para la variante
B, etc.). Los transformadores en si (necesarios solo para volver a
transformar datos NUEVOS con el pipeline B/logistica, no para leer los
parquets ya transformados) estan en `preprocesamiento.py`, en este mismo
directorio.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

DIR_DATOS_PROCESADOS = Path(__file__).resolve().parents[1] / "data" / "processed"

# Splits temporales, por cliente (regla no negociable #1 y #3 de
# PLAN_DE_ACCION.md): C1-C3 entrenamiento, C4 validacion, C5 test (una sola
# pasada, al final).
CORTES_TRAIN = ["C1", "C2", "C3"]
CORTE_VAL = "C4"
CORTE_TEST = "C5"


def cargar_features_target(variante: str = "A2", split: str = "train",
                            dir_datos: Path = DIR_DATOS_PROCESADOS) -> tuple[pd.DataFrame, pd.Series]:
    """Carga X_<variante>_<split>.parquet + y_<split>.parquet.

    variante: "A1" (arboles, one-hot), "A2" (arboles, categoricas nativas,
              la que usa el modelo de produccion) o "B" (logistica, escalada).
    split:    "train", "val" o "test".
    """
    assert variante in {"A1", "A2", "B"}, f"variante desconocida: {variante}"
    assert split in {"train", "val", "test"}, f"split desconocido: {split}"

    X = pd.read_parquet(dir_datos / f"X{variante}_{split}.parquet")
    y = pd.read_parquet(dir_datos / f"y_{split}.parquet")[f"y_{split}"]
    return X, y


def cargar_control(split: str | None = None, dir_datos: Path = DIR_DATOS_PROCESADOS) -> pd.DataFrame:
    """Carga ds_entrenable.parquet (columnas de control + features + churn +
    split), opcionalmente filtrado a un split. Se usa para el desglose de
    metricas por segmento_frecuencia / es_comprador_cronico (regla no
    negociable #5): esas columnas no estan en los X_<variante>_<split>."""
    df = pd.read_parquet(dir_datos / "ds_entrenable.parquet")
    if split is not None:
        etiqueta = {"train": "train", "val": "validacion", "test": "test"}[split]
        df = df.loc[df["split"] == etiqueta].reset_index(drop=True)
    return df
