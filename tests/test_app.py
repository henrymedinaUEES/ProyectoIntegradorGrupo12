"""Pruebas de la capa backend usada por la interfaz Streamlit."""
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from app.prediction_service import PredictionInputError, PredictionService, read_csv, read_excel, read_uploaded_file


class FakeModel:
    feature_name_ = ["a", "metodo_pago_principal_12m"]

    def predict_proba(self, data):
        score = np.asarray(data["a"], dtype=float) / 10
        return np.column_stack([1 - score, score])


def service_without_artifact():
    service = object.__new__(PredictionService)
    service.model = FakeModel()
    service.features = FakeModel.feature_name_
    service.variant = "A2"
    service.validation_pr_auc = 0.7
    return service


def test_predict_ordena_y_selecciona_top():
    service = service_without_artifact()
    data = pd.DataFrame({"id_cliente": ["A", "B", "C"], "a": [2, 9, 5], "metodo_pago_principal_12m": ["EFECTIVO"] * 3})
    ranking, summary = service.predict(data, top_fraction=0.34)
    assert ranking["id_cliente"].tolist() == ["B", "C", "A"]
    assert ranking["seleccionado_campana"].tolist() == [True, True, False]
    assert summary.selected == 2


def test_predict_reporta_columnas_faltantes():
    service = service_without_artifact()
    with pytest.raises(PredictionInputError, match="Faltan columnas"):
        service.predict(pd.DataFrame({"a": [1]}))


def test_read_csv_detecta_separador():
    data = read_csv(BytesIO("a;metodo_pago_principal_12m\n1;EFECTIVO\n".encode()))
    assert list(data.columns) == ["a", "metodo_pago_principal_12m"]


def test_read_excel_lee_primera_hoja():
    buffer = BytesIO()
    pd.DataFrame(
        {"a": [1, 2], "metodo_pago_principal_12m": ["EFECTIVO", "TARJETA"]}
    ).to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    data = read_excel(buffer)
    assert list(data.columns) == ["a", "metodo_pago_principal_12m"]
    assert len(data) == 2


def test_read_uploaded_file_detecta_formato_por_extension():
    csv_buffer = BytesIO("a;metodo_pago_principal_12m\n1;EFECTIVO\n".encode())
    assert list(read_uploaded_file(csv_buffer, "clientes.csv").columns) == ["a", "metodo_pago_principal_12m"]

    xlsx_buffer = BytesIO()
    pd.DataFrame({"a": [1], "metodo_pago_principal_12m": ["EFECTIVO"]}).to_excel(
        xlsx_buffer, index=False, engine="openpyxl"
    )
    xlsx_buffer.seek(0)
    assert list(read_uploaded_file(xlsx_buffer, "clientes.xlsx").columns) == ["a", "metodo_pago_principal_12m"]


def test_read_uploaded_file_rechaza_formato_no_soportado():
    with pytest.raises(PredictionInputError, match="no soportado"):
        read_uploaded_file(BytesIO(b"contenido"), "clientes.txt")
