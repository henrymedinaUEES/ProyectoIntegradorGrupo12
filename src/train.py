"""
Script de entrenamiento del modelo de produccion (LightGBM A2, categoricas
nativas) con los hiperparametros ya congelados en
notebooks/04_optimizacion.ipynb (ver model.py). No vuelve a correr la
busqueda aleatoria de 12 configuraciones -- esa busqueda es un experimento de
notebook, documentado y no se repite en cada entrenamiento de produccion --
solo entrena la configuracion ganadora con early stopping sobre validacion.

Uso:
    python src/train.py

Guarda el modelo entrenado en models/best_model.pkl con el mismo formato de
diccionario que ya usan los notebooks (modelo, params, pr_auc_val, columnas,
variante), para que evaluate.py y notebooks/05_evaluacion.ipynb lo lean sin
cambios.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
from sklearn.metrics import average_precision_score

from data_processing import cargar_features_target
from model import construir_modelo

DIR_MODELOS = Path(__file__).resolve().parents[1] / "models"


def entrenar() -> dict:
    X_tr, y_tr = cargar_features_target("A2", "train")
    X_va, y_va = cargar_features_target("A2", "val")

    modelo = construir_modelo()
    modelo.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    pr_auc_val = average_precision_score(y_va, modelo.predict_proba(X_va)[:, 1])
    paquete = {
        "modelo": modelo,
        "params": {k: getattr(modelo, k) for k in
                    ["num_leaves", "learning_rate", "min_child_samples",
                     "subsample", "colsample_bytree", "reg_lambda"]},
        "pr_auc_val": pr_auc_val,
        "columnas": list(X_tr.columns),
        "variante": "A2",
    }
    return paquete


def main():
    paquete = entrenar()
    print(f"PR-AUC de validacion: {paquete['pr_auc_val']:.4f}  "
          f"(mejor iteracion: {paquete['modelo'].best_iteration_})")
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, DIR_MODELOS / "best_model.pkl")
    print(f"modelo guardado en {DIR_MODELOS / 'best_model.pkl'}")


if __name__ == "__main__":
    main()
