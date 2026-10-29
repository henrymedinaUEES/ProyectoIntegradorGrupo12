"""
Definicion del modelo de produccion: LightGBM sobre la variante A2
(categoricas nativas, sin one-hot), elegido en notebooks/04_optimizacion.ipynb
sobre la regresion logistica y los dos baselines de negocio (ver
docs/optimizacion.md y docs/arquitectura.md para el detalle completo de la
decision).

Los hiperparametros de abajo son los que salieron ganadores de la busqueda
aleatoria de 12 configuraciones (early stopping sobre validacion, nunca
sobre test) documentada en artefactos/_grid_lgbm_A2.json de la carpeta de
trabajo original y en notebooks/04_optimizacion.ipynb.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier

# Congelados en 04_optimizacion.ipynb (T3.3), PR-AUC de validacion = 0.6975
MEJORES_HIPERPARAMETROS = dict(
    num_leaves=127,
    learning_rate=0.05,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=1.0,
    reg_lambda=0.0,
)

# Columna que participa en la definicion de la etiqueta (umbral personalizado
# p90 de intervalos entre compras). Se excluye en el analisis de sensibilidad
# de notebooks/04_optimizacion.ipynb (T3.4), no en el entrenamiento normal.
BLOQUE_UMBRAL = ["umbral_propio_dias"]


def construir_modelo(semilla: int = 42, n_estimators: int = 500) -> LGBMClassifier:
    """Instancia un LightGBM con los hiperparametros ya elegidos. `n_estimators`
    es un techo alto porque el entrenamiento real usa early stopping sobre el
    conjunto de validacion (ver train.py); no se ajusta aqui."""
    return LGBMClassifier(
        n_estimators=n_estimators,
        random_state=semilla,
        verbosity=-1,
        **MEJORES_HIPERPARAMETROS,
    )
