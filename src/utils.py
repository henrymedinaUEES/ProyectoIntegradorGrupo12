"""
Funciones auxiliares compartidas por train.py / evaluate.py y por los
notebooks 03-05. Son las mismas funciones de metricas usadas en todo el
proyecto (03_desercion_modelado.ipynb -> notebooks/03_modelado.ipynb,
04_optimizacion.ipynb y 05_evaluacion.ipynb): PR-AUC, precision@k y lift@k,
nunca accuracy (regla no negociable #4 de PLAN_DE_ACCION.md, ver el
Proyecto de Claude "MODELO DE PREDICCION DE DESERCION").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Tamanos de campana tipicos evaluados en todo el proyecto.
K_GRID = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30]


def precision_en_k(y_true: np.ndarray, score: np.ndarray, k_frac: float) -> float:
    """Fraccion de verdaderos desertores entre el k_frac superior por score."""
    n = len(y_true)
    k = max(1, int(round(n * k_frac)))
    idx = np.argsort(-score)[:k]
    return float(np.asarray(y_true)[idx].mean())


def lift_en_k(y_true: np.ndarray, score: np.ndarray, k_frac: float) -> float:
    """precision@k dividido por la tasa base (que tan por encima del azar)."""
    base = np.asarray(y_true).mean()
    return precision_en_k(y_true, score, k_frac) / base if base > 0 else float("nan")


def tabla_metricas(nombre: str, y_true: np.ndarray, score: np.ndarray) -> dict:
    """Una fila de metricas (PR-AUC + precision@k/lift@k para K_GRID)."""
    fila = {"modelo": nombre, "pr_auc": average_precision_score(y_true, score)}
    for k in K_GRID:
        fila[f"precision@{int(k*100)}%"] = precision_en_k(y_true, score, k)
        fila[f"lift@{int(k*100)}%"] = lift_en_k(y_true, score, k)
    return fila


def lift_por_decil(y_true: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Tabla de lift por decil de score (decil 0 = 10% de mayor score)."""
    dec = pd.qcut(pd.Series(score).rank(method="first"), 10, labels=False)
    t = pd.DataFrame({"y": y_true, "decil": 9 - dec})
    base = np.asarray(y_true).mean()
    g = t.groupby("decil")["y"].agg(["size", "mean"])
    g["lift"] = g["mean"] / base
    return g.sort_index()


def desglose_por(df_control: pd.DataFrame, col: str, y_col: str = "y", score_col: str = "score") -> pd.DataFrame:
    """Metricas desglosadas por una columna de control (segmento_frecuencia,
    es_comprador_cronico, ...). Regla no negociable #5 de PLAN_DE_ACCION.md:
    ninguna metrica se reporta sin este desglose."""
    filas = []
    for val, g in df_control.groupby(col, observed=True):
        y_g, s_g = g[y_col].to_numpy(), g[score_col].to_numpy()
        filas.append({
            col: val, "n": len(g), "tasa_base": y_g.mean(),
            "pr_auc": average_precision_score(y_g, s_g),
            "precision@10%": precision_en_k(y_g, s_g, 0.10),
            "lift@10%": lift_en_k(y_g, s_g, 0.10),
        })
    return pd.DataFrame(filas).round(4)
