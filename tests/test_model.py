"""
Pruebas de src/model.py y src/utils.py. Verifican que el modelo se
construye con los hiperparametros congelados en notebooks/04_optimizacion.ipynb
y que las funciones de metricas de negocio (PR-AUC, precision@k, lift@k)
se comportan correctamente en casos conocidos -- nunca se prueba `accuracy`
(regla no negociable #4 de PLAN_DE_ACCION.md).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import construir_modelo, MEJORES_HIPERPARAMETROS, BLOQUE_UMBRAL
from utils import precision_en_k, lift_en_k, tabla_metricas, lift_por_decil, K_GRID


def test_construir_modelo_devuelve_lgbm_con_hiperparametros_congelados():
    m = construir_modelo()
    assert isinstance(m, LGBMClassifier)
    for k, v in MEJORES_HIPERPARAMETROS.items():
        assert getattr(m, k) == v


def test_bloque_umbral_es_la_columna_de_la_etiqueta():
    assert BLOQUE_UMBRAL == ["umbral_propio_dias"]


def test_precision_en_k_ranking_perfecto():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])  # ranking perfecto
    assert precision_en_k(y, score, 0.20) == 1.0  # los 2 mejores son los 2 positivos


def test_precision_en_k_ranking_al_azar_es_peor_que_perfecto():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    score_malo = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])  # ordena al reves
    assert precision_en_k(y, score_malo, 0.20) == 0.0


def test_lift_en_k_es_uno_cuando_el_score_no_discrimina():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    score = rng.random(2000)  # score sin relacion con y
    lift = lift_en_k(y, score, 0.30)
    assert 0.7 < lift < 1.3  # cerca de 1 (azar), con margen por ruido


def test_tabla_metricas_incluye_todas_las_k_del_grid():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=500)
    score = rng.random(500)
    fila = tabla_metricas("modelo_prueba", y, score)
    for k in K_GRID:
        assert f"precision@{int(k*100)}%" in fila
        assert f"lift@{int(k*100)}%" in fila
    assert "pr_auc" in fila
    assert "accuracy" not in fila  # regla no negociable: nunca accuracy


def test_lift_por_decil_decil_0_es_el_de_mayor_score():
    rng = np.random.default_rng(2)
    score = rng.random(1000)
    y = (score > np.quantile(score, 0.9)).astype(int)  # los positivos son el 10% de mayor score
    tabla = lift_por_decil(y, score)
    assert tabla.loc[0, "lift"] > tabla.loc[9, "lift"]
