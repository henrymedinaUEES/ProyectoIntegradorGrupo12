"""
Script de evaluacion del modelo de produccion (LightGBM A2) sobre el
conjunto de test (C5). Reproduce, fuera de un notebook, exactamente el
mismo calculo de notebooks/05_evaluacion.ipynb: PR-AUC, precision@k, lift@k
(K_GRID de utils.py, nunca accuracy) y el desglose por segmento_frecuencia /
es_comprador_cronico (reglas no negociables #4 y #5 de PLAN_DE_ACCION.md).

Regla no negociable #2: el test se toca una sola vez. Este script no vuelve
a tocar test para decidir nada -- solo repite, de forma reproducible, la
medicion ya congelada.

Uso:
    python src/evaluate.py

Escribe:
    results/metrics/tabla_final_test.csv
    results/metrics/desglose_segmento_test.csv
    results/metrics/desglose_cronico_test.csv
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from data_processing import cargar_features_target, cargar_control
from utils import tabla_metricas, desglose_por

RAIZ = Path(__file__).resolve().parents[1]
DIR_MODELOS = RAIZ / "models"
DIR_RESULTADOS = RAIZ / "results"


def evaluar() -> dict:
    paquete = joblib.load(DIR_MODELOS / "best_model.pkl")
    modelo = paquete["modelo"]

    X_te, y_te = cargar_features_target("A2", "test")
    score_te = modelo.predict_proba(X_te)[:, 1]

    fila = tabla_metricas(f"LightGBM {paquete['variante']} (TEST, produccion)", y_te.to_numpy(), score_te)

    ctrl = cargar_control("test")
    assert len(ctrl) == len(X_te), "ds_entrenable(test) y X_te no tienen el mismo tamano"
    ctrl = ctrl[["segmento_frecuencia", "es_comprador_cronico"]].copy()
    ctrl["score"] = score_te
    ctrl["y"] = y_te.to_numpy()

    seg = desglose_por(ctrl, "segmento_frecuencia")
    cro = desglose_por(ctrl, "es_comprador_cronico")

    return {"fila": fila, "segmento": seg, "cronico": cro}


def main():
    resultado = evaluar()
    tabla = pd.DataFrame([resultado["fila"]]).round(4)
    print(tabla.to_string(index=False))
    print("\n-- por segmento de frecuencia --")
    print(resultado["segmento"].to_string(index=False))
    print("\n-- por comprador cronico --")
    print(resultado["cronico"].to_string(index=False))

    dir_metricas = DIR_RESULTADOS / "metrics"
    dir_metricas.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(dir_metricas / "tabla_final_test.csv", index=False)
    resultado["segmento"].to_csv(dir_metricas / "desglose_segmento_test.csv", index=False)
    resultado["cronico"].to_csv(dir_metricas / "desglose_cronico_test.csv", index=False)
    print(f"\nresultados guardados en {dir_metricas}")


if __name__ == "__main__":
    main()
