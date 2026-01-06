import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import altair as alt
from pathlib import Path
from streamlit.runtime.scriptrunner import get_script_run_ctx

# ============================================================
# CONFIGURACIÓN GENERAL (DEBE SER LO PRIMERO)
# ============================================================
st.set_page_config(
    page_title="Precios Nodales Honduras",
    layout="wide",
)

# ============================================================
# HEALTHCHECK GUARD (CRÍTICO PARA STREAMLIT CLOUD)
# ============================================================
if get_script_run_ctx() is None:
    # Estamos en /healthz → no cargar datos pesados
    st.stop()

# ============================================================
# AUTH (SEGURO)
# ============================================================
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 Acceso restringido")

    APP_PASSWORD = st.secrets.get("APP_PASSWORD")
    if APP_PASSWORD is None:
        st.error("❌ Falta configurar APP_PASSWORD en los secrets de Streamlit Cloud")
        st.stop()

    password = st.text_input("Ingrese contraseña", type="password")

    if password:
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta")

    return False


if not check_password():
    st.stop()

st.title("📍 Precios Nodales – Análisis Espacial y Temporal")

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data_processed"
STATIC_DIR = ROOT / "data_static"

PRICES_PATH = PROCESSED_DIR / "precios_nodales_clean.parquet"
QUALITY_PATH = PROCESSED_DIR / "node_quality.parquet"
NODES_PATH = STATIC_DIR / "nodes_real.csv"

# ============================================================
# CARGA DE DATOS (DIFERIDA Y SEGURA)
# ============================================================
@st.cache_data(ttl=300)
def load_data():
    missing = [p for p in [PRICES_PATH, QUALITY_PATH, NODES_PATH] if not p.exists()]
    if missing:
        return None

    prices = pd.read_parquet(PRICES_PATH)
    quality = pd.read_parquet(QUALITY_PATH)
    nodes = pd.read_csv(NODES_PATH)

    df = (
        prices
        .merge(quality, on="nodo", how="left")
        .merge(nodes, on="nodo", how="left")
    )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    return df


df = load_data()

if df is None:
    st.warning("⚠️ Datos aún no generados")
    st.markdown(
        """
        Esta aplicación necesita que el pipeline se ejecute al menos una vez.

        **Pasos:**
        1. Ve a **Actualizar Datos**
        2. Sube el archivo Excel
        3. Ejecuta el pipeline
        """
    )
    st.stop()

# ============================================================
# FUNCIONES ROBUSTAS
# ============================================================
def robust_exceedance_price(x, poe):
    x = x.dropna()
    if len(x) < 10:
        return np.nan
    cap = x.quantile(0.99)
    x = x[x <= cap]
    return np.percentile(x, 100 - poe)


def robust_volatility(x):
    x = x.dropna()
    if len(x) < 10:
        return np.nan
    return np.percentile(x, 90) - np.percentile(x, 10)

# ============================================================
# SIDEBAR – FILTROS
# ============================================================
st.sidebar.header("⏱️ Filtros de tiempo")

min_date = df["date"].min()
max_date = df["date"].max()

date_start, date_end = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_date, max_date),
)

hour_start, hour_end = st.sidebar.slider("Rango de horas", 0, 23, (0, 23))

st.sidebar.header("📊 Métrica del mapa")

metric = st.sidebar.radio(
    "Selecciona métrica",
    [
        "Promedio",
        "Máximo",
        "Probabilidad de excedencia",
        "Volatilidad (P90 − P10)",
    ],
)

poe = 20
if metric == "Probabilidad de excedencia":
    poe = st.sidebar.slider("POE (%)", 5, 95, 20, step=5)

# ============================================================
# FILTRADO BASE
# ============================================================
df_filt = df[
    (df["date"] >= date_start) &
    (df["date"] <= date_end) &
    (df["hour"] >= hour_start) &
    (df["hour"] <= hour_end)
]

df_filt = df_filt.dropna(subset=["precio", "lat", "lon"])

if df_filt.empty:
    st.warning("No hay datos para el rango seleccionado.")
    st.stop()

# ============================================================
# ESTADÍSTICAS
# ============================================================
df_stats = (
    df_filt
    .groupby("nodo")
    .agg(
        precio_promedio=("precio", "mean"),
        precio_max=("precio", "max"),
        precio_poe=("precio", lambda x: robust_exceedance_price(x, poe)),
        volatilidad=("precio", robust_volatility),
    )
    .reset_index()
)

# ============================================================
# MAPA
# ============================================================
df_map = df_stats.merge(
    df[["nodo", "lat", "lon"]].drop_duplicates(),
    on="nodo",
    how="left"
)

v_min, v_max = df_map["precio_promedio"].min(), df_map["precio_promedio"].max()
df_map["norm"] = (df_map["precio_promedio"] - v_min) / (v_max - v_min) if v_max != v_min else 0
df_map["radius"] = 4000 + df_map["norm"] * 16000

st.subheader("🗺️ Precio promedio")

st.pydeck_chart(
    pdk.Deck(
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="[255 * norm, 60, 255 * (1 - norm), 180]",
                pickable=True,
            )
        ],
        initial_view_state=pdk.ViewState(
            latitude=14.85,
            longitude=-86.6,
            zoom=6,
        ),
        tooltip={"html": "<b>{nodo}</b><br/>Precio promedio"},
    )
)
