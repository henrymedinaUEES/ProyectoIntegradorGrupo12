"""Dashboard Streamlit para generar rankings de riesgo de deserción."""
from __future__ import annotations

import io
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from prediction_service import (  # noqa: E402
    CATEGORICAL_FEATURE,
    NUMERIC_FEATURES,
    PredictionInputError,
    PredictionService,
    read_uploaded_file,
)


st.set_page_config(page_title="Radar de Retención", page_icon="💊", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink:#102a43; --brand:#087f8c; --soft:#e8f6f7; --warm:#f59e0b; }
    html, body, [data-testid="stAppViewContainer"], .stApp {
      color:#102a43 !important;
      background:linear-gradient(135deg,#f7fbfc 0%,#ffffff 45%,#f8fafc 100%);
    }
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] span:not([data-baseweb="tag"]),
    [data-testid="stMarkdownContainer"] { color:#102a43; }
    [data-testid="stSidebar"] { background:#e8f3f5; }
    [data-testid="stSidebar"] * { color:#102a43; }
    [data-baseweb="tab-list"] { gap:.35rem; }
    [data-baseweb="tab"] { color:#365568 !important; background:#edf5f6; border-radius:10px 10px 0 0; }
    [aria-selected="true"][data-baseweb="tab"] { color:#075f68 !important; background:#d8eeee; font-weight:700; }
    [data-baseweb="input"] input,
    [data-baseweb="select"] input,
    [data-baseweb="textarea"] textarea { color:#102a43 !important; -webkit-text-fill-color:#102a43 !important; }
    [data-baseweb="input"], [data-baseweb="select"] > div { background:white !important; }
    [data-testid="stFileUploaderDropzone"] { background:#f2f8f9; color:#102a43; border-color:#91bfc4; }
    .hero { padding:1.5rem 1.7rem; border-radius:20px; color:white;
      background:linear-gradient(120deg,#073b4c,#087f8c 65%,#0f9b8e); margin-bottom:1rem; }
    .hero h1 { margin:0; font-size:2.15rem; letter-spacing:-.04em; }
    .hero p { margin:.4rem 0 0; color:#d8f3f1; max-width:760px; }
    .hero h1, .hero p, .hero * { color:white !important; }
    .note { padding:.8rem 1rem; border-left:4px solid var(--warm); background:#fffbeb; border-radius:8px; }
    div[data-testid="stMetric"] { color:#102a43; background:white; border:1px solid #dce8eb; padding:1rem; border-radius:14px; }
    div[data-testid="stMetric"] * { color:#102a43 !important; }
    .stButton>button, .stDownloadButton>button { border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_service() -> PredictionService:
    return PredictionService()


@st.cache_data
def plantilla_excel() -> bytes:
    example = {feature: [0] for feature in NUMERIC_FEATURES}
    example.update({
        "id_cliente": ["CLIENTE-001"],
        CATEGORICAL_FEATURE: ["EFE"],
    })
    columns = ["id_cliente", *NUMERIC_FEATURES, CATEGORICAL_FEATURE]
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(example)[columns].to_excel(writer, index=False, sheet_name="Plantilla")
    return buffer.getvalue()


def render_eda_dashboard(df: pd.DataFrame) -> None:
    """Panel exploratorio (EDA) de la base cargada: nulos, distribuciones y
    correlacion de las variables numericas, y la categorica. Pensado para un
    publico no tecnico: graficos simples con la etiqueta del dato a la vista."""
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_col = CATEGORICAL_FEATURE if CATEGORICAL_FEATURE in df.columns else None
    columnas_modelo = numeric_cols + ([cat_col] if cat_col else [])

    st.markdown("**Panorama general de la base**")
    nulos_pct = float(df[columnas_modelo].isna().mean().mean() * 100) if columnas_modelo else 0.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros", f"{len(df):,}")
    m2.metric("Columnas totales", f"{df.shape[1]:,}")
    m3.metric("Variables del modelo", f"{len(columnas_modelo)}")
    m4.metric("Nulos promedio", f"{nulos_pct:.2f}%")

    if columnas_modelo:
        nulos_col = df[columnas_modelo].isna().mean().mul(100).sort_values(ascending=False)
        nulos_col = nulos_col[nulos_col > 0]
        if not nulos_col.empty:
            st.markdown("**Valores nulos por columna**")
            fig, ax = plt.subplots(figsize=(6, max(2, 0.3 * len(nulos_col))))
            barras = ax.barh(nulos_col.index, nulos_col.values, color="#f59e0b")
            ax.bar_label(barras, labels=[f"{v:.1f}%" for v in nulos_col.values], fontsize=7, padding=3)
            ax.invert_yaxis()
            ax.set_xlabel("% de nulos", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
            ax.margins(x=0.15)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True, width="content")
        else:
            st.caption("No se detectaron valores nulos en las columnas que usa el modelo.")

    if numeric_cols:
        st.markdown("**Estadisticas descriptivas**")
        st.dataframe(df[numeric_cols].describe().T.round(2), use_container_width=True)

        st.markdown("**Distribucion de variables numericas**")
        n_cols_grid = 3
        n_rows_grid = -(-len(numeric_cols) // n_cols_grid)
        fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(3.6 * n_cols_grid, 2.4 * n_rows_grid))
        axes = axes.flatten()
        for ax, col in zip(axes, numeric_cols):
            ax.hist(df[col].dropna(), bins=25, color="#087f8c")
            ax.set_title(col, fontsize=8)
            ax.tick_params(axis="both", labelsize=6)
        for ax in axes[len(numeric_cols):]:
            ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True, width="stretch")

        st.markdown("**Relacion entre variables (matriz de correlacion)**")
        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=90, fontsize=6)
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_yticklabels(numeric_cols, fontsize=6)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True, width="content")

    if cat_col:
        st.markdown(f"**Distribucion de `{cat_col}`**")
        conteo = df[cat_col].astype(str).fillna("DESCONOCIDO").value_counts().head(10)
        fig, ax = plt.subplots(figsize=(5, 3))
        barras = ax.bar(conteo.index, conteo.values, color="#073b4c")
        ax.bar_label(barras, labels=[f"{v:,.0f}" for v in conteo.values], fontsize=7, padding=3)
        ax.set_ylabel("Clientes", fontsize=9)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.margins(y=0.15)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True, width="content")


try:
    service = get_service()
except Exception as exc:
    st.error(f"No fue posible iniciar el motor de predicción: {exc}")
    st.stop()

st.markdown(
    """<section class="hero"><h1>Radar de Retención</h1>
    <p>Convierte comportamiento transaccional en una lista accionable de clientes para campañas de retención.</p>
    </section>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Configuración")
    campaign_pct = st.slider("Tamaño de campaña", 1, 50, 10, 1, format="%d%%")
    st.caption(f"Modelo LightGBM · variante {service.variant}")
    if service.validation_pr_auc is not None:
        st.caption(f"PR-AUC de validación: {service.validation_pr_auc:.4f}")
    st.markdown("---")
    st.markdown("**Privacidad**")
    st.caption("El procesamiento ocurre localmente. No cargue nombres, teléfonos, documentos ni correos.")

tab_rank, tab_manual, tab_about = st.tabs(
    ["Generar ranking", "Evaluar un cliente", "Acerca del modelo"]
)

with tab_rank:
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Fuente de datos")
        source = st.radio(
            "Seleccione una opción",
            ["Demostración con datos de test", "Cargar archivo CSV o Excel"],
            horizontal=True,
            label_visibility="collapsed",
        )
        uploaded = None
        if source == "Cargar archivo CSV o Excel":
            uploaded = st.file_uploader("Archivo con una fila por cliente", type=["csv", "xlsx"])
            st.caption("Acepta CSV o Excel (.xlsx). Si el Excel tiene varias hojas, se usa solo la primera.")
            st.download_button(
                "Descargar plantilla Excel",
                plantilla_excel(),
                "plantilla_clientes.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            demo_rows = st.select_slider("Clientes de demostración", options=[100, 500, 1000, 5000, 10000], value=1000)
    with right:
        st.markdown(
            """<div class="note"><b>Cómo leer el resultado</b><br>
            El score es útil para ordenar clientes, pero aún no es una probabilidad calibrada.
            “Prioridad” identifica el porcentaje superior definido para la campaña.</div>""",
            unsafe_allow_html=True,
        )

    run = st.button("Analizar clientes", type="primary", use_container_width=True)
    if run:
        try:
            if source == "Cargar archivo CSV o Excel":
                if uploaded is None:
                    raise PredictionInputError("Seleccione un archivo CSV o Excel antes de analizar")
                input_data = read_uploaded_file(uploaded, uploaded.name)
            else:
                input_data = service.load_demo(demo_rows)
            st.session_state["input_data"] = input_data
            ranking, summary = service.predict(input_data, campaign_pct / 100)
            st.session_state["ranking"] = ranking
            st.session_state["summary"] = summary
        except (PredictionInputError, FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"La predicción no pudo completarse: {exc}")

    if "ranking" in st.session_state:
        ranking = st.session_state["ranking"]
        summary = st.session_state["summary"]
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clientes analizados", f"{summary.total:,}")
        m2.metric("Para contactar", f"{summary.selected:,}")
        m3.metric("Score de corte", f"{summary.cutoff_score:.3f}")
        m4.metric("Score promedio", f"{summary.mean_score:.3f}")

        st.subheader("Distribución de clientes por nivel de riesgo")
        resumen_niveles = ranking.groupby("nivel_riesgo")["score_riesgo"].agg(cantidad="count", minimo="min", maximo="max")
        orden_niveles = [nivel for nivel in ["Prioridad", "Vigilancia", "Resto"] if nivel in resumen_niveles.index]
        resumen_niveles = resumen_niveles.reindex(orden_niveles)
        colores_nivel = {"Prioridad": "#087f8c", "Vigilancia": "#f59e0b", "Resto": "#cbd5e1"}

        fig, ax = plt.subplots(figsize=(4.5, 3))
        barras = ax.bar(
            resumen_niveles.index,
            resumen_niveles["cantidad"],
            color=[colores_nivel.get(nivel, "#cbd5e1") for nivel in resumen_niveles.index],
        )
        for barra, (_, fila) in zip(barras, resumen_niveles.iterrows()):
            ax.annotate(
                f"{int(fila['cantidad']):,}\nscore {fila['minimo']:.2f} – {fila['maximo']:.2f}",
                xy=(barra.get_x() + barra.get_width() / 2, barra.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_xlabel("Nivel de riesgo", fontsize=9)
        ax.set_ylabel("Cantidad de clientes", fontsize=9)
        ax.set_title("Clientes por nivel de riesgo (con rango de score)", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        ax.margins(y=0.2)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True, width="content")

        display_columns = [column for column in ["prioridad", "id_cliente", "score_riesgo", "nivel_riesgo", "seleccionado_campana", "recency_dias", "n_visitas_12m"] if column in ranking]
        st.dataframe(
            ranking[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "score_riesgo": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.3f"),
                "seleccionado_campana": st.column_config.CheckboxColumn("Contactar"),
            },
        )
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            ranking.to_excel(writer, index=False, sheet_name="Ranking")
        st.download_button(
            "Descargar ranking completo",
            excel_buffer.getvalue(),
            "ranking_retencion.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

        st.markdown("---")
        st.subheader("Exploracion de su base (EDA)")
        st.caption("Panorama sencillo de los datos que acaba de analizar.")
        if "input_data" in st.session_state:
            render_eda_dashboard(st.session_state["input_data"])

with tab_manual:
    st.subheader("Evaluación individual")
    st.caption("Complete el comportamiento agregado del cliente durante los últimos 12 meses.")
    with st.form("manual_prediction"):
        c1, c2, c3 = st.columns(3)
        values: dict[str, float | str] = {}
        labels = {
            "recency_dias": "Días desde última compra",
            "n_visitas_12m": "Visitas en 12 meses",
            "n_dias_activo_12m": "Días activos en 12 meses",
            "monto_total_12m": "Monto total en 12 meses",
            "ticket_promedio": "Ticket promedio",
            "tendencia_gasto": "Tendencia de gasto",
            "pct_gasto_cronico_12m": "% gasto crónico (0–1)",
            "es_comprador_cronico": "Comprador crónico (0/1)",
            "n_categorias_distintas_12m": "Categorías distintas",
            "n_bodegas_distintas_12m": "Locales distintos",
            "antiguedad_meses": "Antigüedad en meses",
            "n_metodos_pago_distintos_12m": "Métodos de pago distintos",
            "umbral_propio_dias": "Umbral personal de inactividad",
        }
        defaults = [30, 12, 10, 500, 40, 0, 0, 0, 4, 3, 24, 2, 60]
        for index, (feature, default) in enumerate(zip(NUMERIC_FEATURES, defaults)):
            with (c1, c2, c3)[index % 3]:
                values[feature] = st.number_input(labels[feature], value=float(default), step=1.0)
        with c2:
            values[CATEGORICAL_FEATURE] = st.selectbox("Método de pago principal", service.categorical_values)
        submitted = st.form_submit_button("Calcular riesgo", type="primary", use_container_width=True)
    if submitted:
        try:
            result, _ = service.predict(pd.DataFrame([values]), 1.0)
            score = float(result.loc[0, "score_riesgo"])
            st.metric("Score de riesgo", f"{score:.3f}")
            st.progress(score)
            st.info("Este resultado sirve para priorización relativa. Compárelo dentro de una cartera de clientes antes de tomar una decisión.")
        except Exception as exc:
            st.error(f"No se pudo calcular el score: {exc}")

with tab_about:
    st.subheader("Qué hace este sistema")
    st.write(
        "El modelo identifica patrones compatibles con interrupción del ritmo habitual de compra durante los próximos 120 días. "
        "Fue entrenado con clientes que registran al menos cuatro visitas anuales."
    )
    a, b, c = st.columns(3)
    a.metric("PR-AUC test", "0.6875")
    b.metric("Precision@10%", "84.03%")
    c.metric("Lift@10%", "2.09×")
    st.warning(
        "No determina la causa de la inactividad, no recomienda una oferta específica y no sustituye el criterio del equipo de Marketing."
    )
