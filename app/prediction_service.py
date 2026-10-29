"""Capa de inferencia para conectar el modelo LightGBM con interfaces web."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "best_model.pkl"
DEFAULT_DEMO_PATH = ROOT / "data" / "processed" / "XA2_test.parquet"
DEFAULT_DEMO_IDS_PATH = ROOT / "data" / "processed" / "ds_entrenable.parquet"

NUMERIC_FEATURES = [
    "recency_dias",
    "n_visitas_12m",
    "n_dias_activo_12m",
    "monto_total_12m",
    "ticket_promedio",
    "tendencia_gasto",
    "pct_gasto_cronico_12m",
    "es_comprador_cronico",
    "n_categorias_distintas_12m",
    "n_bodegas_distintas_12m",
    "antiguedad_meses",
    "n_metodos_pago_distintos_12m",
    "umbral_propio_dias",
]
CATEGORICAL_FEATURE = "metodo_pago_principal_12m"


class PredictionInputError(ValueError):
    """Error comprensible para el usuario en los datos de entrada."""


@dataclass(frozen=True)
class PredictionSummary:
    total: int
    selected: int
    cutoff_score: float
    mean_score: float


class PredictionService:
    """Carga una vez el modelo, valida datos y genera rankings de riesgo."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {self.model_path}")
        package = joblib.load(self.model_path)
        if not isinstance(package, dict) or "modelo" not in package:
            raise RuntimeError("best_model.pkl no tiene el formato esperado")
        self.model = package["modelo"]
        self.variant = package.get("variante", "A2")
        self.validation_pr_auc = package.get("pr_auc_val")
        self.features = list(package.get("columnas") or getattr(self.model, "feature_name_", []))
        if not self.features:
            raise RuntimeError("El modelo no declara sus columnas de entrada")

    @property
    def categorical_values(self) -> list[str]:
        booster = getattr(self.model, "booster_", None)
        values = getattr(booster, "pandas_categorical", None)
        if values and values[0]:
            return [str(value) for value in values[0]]
        return ["DESCONOCIDO", "EFECTIVO", "TARJETA", "OTROS"]

    def _prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            raise PredictionInputError("El archivo no contiene filas")
        missing = [column for column in self.features if column not in data.columns]
        if missing:
            raise PredictionInputError("Faltan columnas requeridas: " + ", ".join(missing))

        result = data[self.features].copy()
        for column in self.features:
            if column == CATEGORICAL_FEATURE:
                continue
            result[column] = pd.to_numeric(result[column], errors="coerce")
            invalid = result[column].isna() & data[column].notna()
            if invalid.any():
                raise PredictionInputError(
                    f"La columna '{column}' contiene valores que no son numéricos"
                )

        if CATEGORICAL_FEATURE in result:
            # Convertir primero a object evita que pandas intente insertar una
            # categoría nueva en un Categorical proveniente de Parquet.
            raw = result[CATEGORICAL_FEATURE].astype("object").fillna("DESCONOCIDO").astype(str)
            categories = self.categorical_values
            fallback = "OTROS" if "OTROS" in categories else categories[0]
            raw = raw.where(raw.isin(categories), fallback)
            result[CATEGORICAL_FEATURE] = pd.Categorical(raw, categories=categories)
        return result

    def predict(self, data: pd.DataFrame, top_fraction: float = 0.10) -> tuple[pd.DataFrame, PredictionSummary]:
        if not 0 < top_fraction <= 1:
            raise PredictionInputError("El tamaño de campaña debe estar entre 1% y 100%")
        prepared = self._prepare(data)
        scores = self.model.predict_proba(prepared)[:, 1]

        output = data.copy()
        output["score_riesgo"] = scores
        output = output.sort_values("score_riesgo", ascending=False).reset_index(drop=True)
        output.insert(0, "prioridad", np.arange(1, len(output) + 1))
        selected = max(1, int(np.ceil(len(output) * top_fraction)))
        output["seleccionado_campana"] = output["prioridad"] <= selected
        output["nivel_riesgo"] = np.select(
            [output["prioridad"] <= selected, output["prioridad"] <= max(selected, int(np.ceil(len(output) * 0.30)))],
            ["Prioridad", "Vigilancia"],
            default="Resto",
        )
        summary = PredictionSummary(
            total=len(output),
            selected=selected,
            cutoff_score=float(output.loc[selected - 1, "score_riesgo"]),
            mean_score=float(np.mean(scores)),
        )
        return output, summary

    def load_demo(self, rows: int = 5000) -> pd.DataFrame:
        if not DEFAULT_DEMO_PATH.exists():
            raise FileNotFoundError(f"No se encontraron datos de demostración: {DEFAULT_DEMO_PATH}")
        demo = pd.read_parquet(DEFAULT_DEMO_PATH).reset_index(drop=True)
        if "id_cliente" not in demo.columns:
            demo.insert(0, "id_cliente", self._demo_ids(len(demo)))
        return demo.head(rows)

    @staticmethod
    def _demo_ids(n_filas: int):
        """Recupera el id_cliente real de las filas de XA2_test.parquet.

        XA2_test.parquet solo trae las features del modelo (el id no participa
        del entrenamiento), pero se genera a partir de `ds_entrenable.parquet`
        sin reordenar filas: filtrando ese archivo por `split == "test"` se
        recupera el id_cliente real en la misma posición (verificado: mismo
        conteo de filas, 66.421, y los mismos valores de features en varias
        filas de control repartidas en todo el archivo). Si el archivo de
        control no existe o el conteo no coincide (por ejemplo, si se
        regeneran los datos y cambia el orden), se cae a un identificador
        sintético para no mostrar un id incorrecto.
        """
        if DEFAULT_DEMO_IDS_PATH.exists():
            try:
                control = pd.read_parquet(DEFAULT_DEMO_IDS_PATH, columns=["id_cliente", "split"])
                ids_test = control.loc[control["split"] == "test", "id_cliente"].reset_index(drop=True)
                if len(ids_test) == n_filas:
                    return ids_test
            except Exception:
                pass
        return [f"DEMO-{i + 1:05d}" for i in range(n_filas)]


def read_csv(upload: BinaryIO) -> pd.DataFrame:
    """Lee CSV UTF-8 o Latin-1 y acepta coma o punto y coma."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            upload.seek(0)
            return pd.read_csv(upload, sep=None, engine="python", encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise PredictionInputError(f"No se pudo leer el CSV: {last_error}")


def read_excel(upload: BinaryIO) -> pd.DataFrame:
    """Lee la primera hoja de un archivo Excel (.xlsx/.xlsm)."""
    try:
        upload.seek(0)
        return pd.read_excel(upload, sheet_name=0, engine="openpyxl")
    except Exception as exc:
        raise PredictionInputError(f"No se pudo leer el Excel: {exc}")


def read_uploaded_file(upload: BinaryIO, filename: str) -> pd.DataFrame:
    """Lee un archivo subido (CSV o Excel) según la extensión de `filename`."""
    suffix = Path(filename).suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        return read_excel(upload)
    if suffix == ".csv":
        return read_csv(upload)
    raise PredictionInputError(
        f"Formato de archivo no soportado: '{suffix}'. Use .csv o .xlsx"
    )
