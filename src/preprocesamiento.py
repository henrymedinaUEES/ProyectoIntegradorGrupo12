"""
preprocesamiento.py
====================
Transformadores de sklearn compartidos por `02_desercion_eda_prep.ipynb` y
`03_desercion_modelado.ipynb`.

Se movieron aquí (Fase 3, "Aviso técnico" de PLAN_DE_ACCION.md) porque el
joblib `artefactos/preprocesamiento.joblib` referencia estas clases: para
cargarlo con `joblib.load` hace falta que estén definidas/importables ANTES
de deserializarlo, y una lambda o una clase definida solo dentro de un
notebook no es importable desde otro. Este módulo es la fuente única de
verdad para estas cuatro clases; los notebooks las importan de aquí en vez
de volver a definirlas.

No cambia ninguna lógica respecto a como estaban en
`02_desercion_eda_prep.ipynb` (celdas 5.2 y 5.4, 20/09/2026): mismo
comportamiento, mismos nombres, mismos parámetros por defecto.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MIN_FREQ = 0.01          # 1 % de los clientes de train
ETIQ_RARA = "OTROS"
ETIQ_NULA = "DESCONOCIDO"


class AgrupadorRaras(BaseEstimator, TransformerMixin):
    """Colapsa en OTROS las categorías con frecuencia < min_freq en el fit."""

    def __init__(self, min_freq=0.01, etiq_rara=ETIQ_RARA, etiq_nula=ETIQ_NULA):
        self.min_freq, self.etiq_rara, self.etiq_nula = min_freq, etiq_rara, etiq_nula

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columnas_ = list(X.columns)
        self.mantener_ = {}
        for c in self.columnas_:
            s = X[c].astype("object").where(X[c].notna(), self.etiq_nula).astype(str)
            fr = s.value_counts(normalize=True)
            mant = set(fr[fr >= self.min_freq].index)
            if self.etiq_nula in fr.index:      # solo si hay nulos de verdad
                mant.add(self.etiq_nula)
            self.mantener_[c] = mant
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.columnas_:
            s = X[c].astype("object").where(X[c].notna(), self.etiq_nula).astype(str)
            X[c] = np.where(s.isin(self.mantener_[c]), s, self.etiq_rara)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columnas_, dtype=object)


class _PasaNombres:
    """Mixin: la transformación es 1 a 1, así que los nombres no cambian."""

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(getattr(self, "nombres_", []), dtype=object)
        return np.asarray(input_features, dtype=object)


class LimpiaInfinitos(_PasaNombres, BaseEstimator, TransformerMixin):
    """Convierte +-inf en nulo para que el imputador los trate."""

    def fit(self, X, y=None):
        self.nombres_ = list(getattr(X, "columns", []))
        return self

    def transform(self, X):
        A = np.asarray(X, dtype="float64")
        return np.where(np.isinf(A), np.nan, A)


class Winsorizador(_PasaNombres, BaseEstimator, TransformerMixin):
    """Acota cada columna a los percentiles aprendidos en el fit (solo train)."""

    def __init__(self, inferior=0.01, superior=0.99):
        self.inferior, self.superior = inferior, superior

    def fit(self, X, y=None):
        self.nombres_ = list(getattr(X, "columns", []))
        A = np.asarray(X, dtype="float64")
        self.lo_ = np.nanquantile(A, self.inferior, axis=0)
        self.hi_ = np.nanquantile(A, self.superior, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype="float64"), self.lo_, self.hi_)


class LogConSigno(_PasaNombres, BaseEstimator, TransformerMixin):
    """sign(x)*log(1+|x|): linealiza colas y admite valores negativos."""

    def fit(self, X, y=None):
        self.nombres_ = list(getattr(X, "columns", []))
        return self

    def transform(self, X):
        A = np.asarray(X, dtype="float64")
        return np.sign(A) * np.log1p(np.abs(A))
