"""
Pruebas de src/data_processing.py contra los datos ya cacheados en
data/processed/ (generados por notebooks/02_preprocesamiento.ipynb). No
recalculan el preprocesamiento: solo verifican que lo ya guardado se puede
leer, tiene la forma esperada y respeta las reglas no negociables del plan
de accion (particion por cliente/out-of-time, sin fuga de la etiqueta).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_processing import cargar_features_target, cargar_control, CORTES_TRAIN, CORTE_VAL, CORTE_TEST


@pytest.mark.parametrize("variante", ["A1", "A2", "B"])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_carga_features_target_formas_consistentes(variante, split):
    X, y = cargar_features_target(variante, split)
    assert len(X) == len(y)
    assert len(X) > 0
    assert not X.empty


def test_conteos_de_filas_conocidos():
    # Tal como quedaron documentados en CONTEXTO_PROYECTO.md / PLAN_DE_ACCION.md
    # (26/09/2026): train 796,044 / val 68,164 / test 66,421 filas.
    X_tr, y_tr = cargar_features_target("A2", "train")
    X_va, y_va = cargar_features_target("A2", "val")
    X_te, y_te = cargar_features_target("A2", "test")
    assert len(X_tr) == 796_044
    assert len(X_va) == 68_164
    assert len(X_te) == 66_421


def test_umbral_propio_dias_esta_en_features():
    # BLOQUE_UMBRAL (columna que participa en la definicion de la etiqueta,
    # ver model.py) debe seguir presente en A2 para poder excluirla en el
    # analisis de sensibilidad de notebooks/04_optimizacion.ipynb.
    X, _ = cargar_features_target("A2", "train")
    assert "umbral_propio_dias" in X.columns


def test_etiqueta_fuera_de_las_features():
    # Regla no negociable: ninguna columna de etiqueta dentro de las features.
    X, _ = cargar_features_target("A2", "train")
    assert "churn" not in X.columns
    assert "censurado" not in X.columns


def test_cargar_control_filtra_por_split_y_alinea_con_features():
    X_te, y_te = cargar_features_target("A2", "test")
    ctrl_te = cargar_control("test")
    assert len(ctrl_te) == len(X_te)
    assert set(ctrl_te["split"].unique()) == {"test"}
    # Verificacion de alineacion fila a fila (misma usada en evaluate.py):
    assert np.allclose(ctrl_te["recency_dias"].to_numpy(), X_te["recency_dias"].to_numpy())


def test_cortes_definidos_segun_diseno_out_of_time():
    # Regla no negociable: out-of-time, nunca cortes mezclados al azar.
    assert CORTES_TRAIN == ["C1", "C2", "C3"]
    assert CORTE_VAL == "C4"
    assert CORTE_TEST == "C5"
