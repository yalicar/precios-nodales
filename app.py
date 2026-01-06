import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import altair as alt
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Precios Nodales Honduras", layout="wide")

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data_processed"
STATIC_DIR = ROOT / "data_static"

PRICES_PATH = PROCESSED_DIR / "precios_nodales_clean.parquet"
QUALITY_PATH = PROCESSED_DIR / "node_quality.parquet"
NODES_PATH = STATIC_DIR / "nodes_real.csv"

# ============================================================
# AUTH (safe secrets)
# ============================================================
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 Acceso restringido")

    app_pw = st.secrets.get("APP_PASSWORD")
    if app_pw is None:
        st.error("❌ Falta configurar APP_PASSWORD en Streamlit Cloud → Settings → Secrets")
        st.stop()

    pw = st.text_input("Ingrese contraseña", type="password")
    if pw:
        if pw == app_pw:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta")

    return False

if not check_password():
    st.stop()

st.title("📍 Precios Nodales – Análisis Espacial y Temporal")

# ============================================================
# Helpers robustos
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
# Sidebar filtros (UI SIEMPRE ARRANCA)
# ============================================================
st.sidebar.header("⏱️ Filtros de tiempo")

# Si no hay datos, ponemos defaults “seguros”
today = pd.Timestamp.utcnow().date()
date_start, date_end = st.sidebar.date_input("Rango de fechas", value=(today, today))
hour_start, hour_end = st.sidebar.slider("Rango de horas", 0, 23, (0, 23))

st.sidebar.header("📊 Métrica del mapa")
metric = st.sidebar.radio(
    "Selecciona métrica",
    ["Promedio", "Máximo", "Probabilidad de excedencia", "Volatilidad (P90 − P10)"],
)

poe = 20
if metric == "Probabilidad de excedencia":
    poe = st.sidebar.slider("POE (%)", 5, 95, 20, step=5)

st.sidebar.header("⚙️ Calidad")
show_low_coverage = st.sidebar.checkbox("Mostrar nodos con baja cobertura (≥90% NaN)", value=False)

# BOTÓN CLAVE: evita cargar en el arranque (healthz no “clickea”)
st.sidebar.divider()
load_now = st.sidebar.button("📥 Cargar datos", type="primary")

# Estado para no recargar cada rerun
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# ============================================================
# Loaders (NO se ejecutan hasta que el usuario lo pida)
# ============================================================
@st.cache_data(ttl=300)
def load_nodes_and_quality():
    missing = [p for p in [QUALITY_PATH, NODES_PATH] if not p.exists()]
    if missing:
        return None, missing
    quality = pd.read_parquet(QUALITY_PATH)
    nodes = pd.read_csv(NODES_PATH)
    nodes["lat"] = pd.to_numeric(nodes["lat"], errors="coerce")
    nodes["lon"] = pd.to_numeric(nodes["lon"], errors="coerce")
    return (quality, nodes), None

@st.cache_data(ttl=300)
def load_prices_filtered(date_start_, date_end_, hour_start_, hour_end_):
    # Si falta parquet grande, no rompemos la app
    if not PRICES_PATH.exists():
        return None, [PRICES_PATH]

    # Convertimos a timestamps para filtros
    start_ts = pd.Timestamp(date_start_)
    end_ts = pd.Timestamp(date_end_) + pd.Timedelta(days=1)  # inclusivo por fecha

    # Intento con predicate pushdown (pyarrow via pandas)
    # Nota: filtros por hour los hacemos después para no complicar engines
    df = pd.read_parquet(
        PRICES_PATH,
        columns=["datetime", "nodo", "precio"],
        filters=[("datetime", ">=", start_ts), ("datetime", "<", end_ts)],
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    df = df[(df["hour"] >= hour_start_) & (df["hour"] <= hour_end_)]
    return df, None

# ============================================================
# Si no presiona cargar, mostramos instrucciones (arranque sano)
# ============================================================
if not load_now and not st.session_state.data_loaded:
    st.info(
        "La app arrancó sin cargar el parquet grande para evitar crashes en Streamlit Cloud.\n\n"
        "1) Ve a **Actualizar Datos** y corre el pipeline (si aún no lo hiciste)\n"
        "2) Regresa aquí y presiona **📥 Cargar datos**"
    )
    st.stop()

# Marcamos que el usuario quiere datos (permite reruns sin tocar el botón)
st.session_state.data_loaded = True

# ============================================================
# Cargar calidad/nodos
# ============================================================
(qn, missing_qn) = load_nodes_and_quality()
if qn is None:
    st.error("❌ Faltan archivos base (calidad/nodos).")
    st.code("\n".join(str(p) for p in missing_qn))
    st.stop()

quality, nodes = qn

# Cargar precios filtrados
with st.spinner("Cargando datos filtrados..."):
    prices, missing_prices = load_prices_filtered(date_start, date_end, hour_start, hour_end)

if prices is None:
    st.warning("⚠️ No existe el parquet procesado aún.")
    st.code("\n".join(str(p) for p in missing_prices))
    st.info("Ve a **Actualizar Datos** y ejecuta el pipeline.")
    st.stop()

# Merge final
df = (
    prices
    .merge(quality, on="nodo", how="left")
    .merge(nodes, on="nodo", how="left")
)

# Limpieza mínima
df = df.dropna(subset=["precio", "lat", "lon"])

if df.empty:
    st.warning("No hay datos para el rango seleccionado.")
    st.stop()

# ============================================================
# Stats por nodo
# ============================================================
# flags opcionales si existen en tu parquet de calidad
if "is_dead" in df.columns:
    df = df[df["is_dead"] != True]
if (not show_low_coverage) and ("is_low_coverage" in df.columns):
    df = df[df["is_low_coverage"] != True]

df_stats = (
    df.groupby("nodo")
      .agg(
          precio_promedio=("precio", "mean"),
          precio_min=("precio", "min"),
          precio_max=("precio", "max"),
          cobertura_nan=("nan_pct", "mean") if "nan_pct" in df.columns else ("precio", "size"),
          precio_poe=("precio", lambda x: robust_exceedance_price(x, poe)),
          volatilidad=("precio", robust_volatility),
      )
      .reset_index()
)

# ============================================================
# Map metric
# ============================================================
if metric == "Promedio":
    df_map = df_stats.rename(columns={"precio_promedio": "valor"})
    metric_label = "Precio promedio [USD/MWh]"
elif metric == "Máximo":
    df_map = df_stats.rename(columns={"precio_max": "valor"})
    metric_label = "Precio máximo [USD/MWh]"
elif metric == "Probabilidad de excedencia":
    df_map = df_stats.rename(columns={"precio_poe": "valor"})
    metric_label = f"Precio POE {poe}% [USD/MWh]"
else:
    df_map = df_stats.rename(columns={"volatilidad": "valor"})
    metric_label = "Volatilidad (P90 − P10) [USD/MWh]"

df_map = df_map.merge(nodes[["nodo", "lat", "lon"]].drop_duplicates(), on="nodo", how="left")
df_map = df_map.dropna(subset=["valor", "lat", "lon"])

v_min, v_max = df_map["valor"].min(), df_map["valor"].max()
df_map["norm"] = (df_map["valor"] - v_min) / (v_max - v_min) if v_max != v_min else 0.0
df_map["radius"] = 4000 + df_map["norm"] * 16000
df_map["color_r"] = (255 * df_map["norm"]).astype(int)
df_map["color_g"] = 60
df_map["color_b"] = (255 * (1 - df_map["norm"])).astype(int)

# ============================================================
# MAPA
# ============================================================
st.subheader(f"🗺️ {metric_label}")

st.pydeck_chart(
    pdk.Deck(
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="[color_r, color_g, color_b, 180]",
                pickable=True,
            )
        ],
        initial_view_state=pdk.ViewState(
            latitude=14.85,
            longitude=-86.60,
            zoom=6,
            bearing=0,
            pitch=0,
        ),
        tooltip={"html": "<b>{nodo}</b><br/>Valor: {valor}"},
    )
)

# ============================================================
# TABLA
# ============================================================
st.subheader("📋 Resumen por nodo")

df_table = (
    df_stats
      .rename(columns={
          "precio_promedio": "Promedio",
          "precio_min": "Mínimo",
          "precio_max": "Máximo",
          "precio_poe": f"Precio POE {poe}%",
          "volatilidad": "Volatilidad (P90−P10)",
          "cobertura_nan": "Cobertura NaN (%)",
      })
      .round(2)
      .sort_values("Promedio", ascending=False)
)
st.dataframe(df_table, use_container_width=True)

# ============================================================
# SERIE TEMPORAL (1 o 2 nodos)
# ============================================================
st.subheader("📈 Serie temporal por nodo")

nodos_list = sorted(df["nodo"].unique())
col1, col2 = st.columns([2, 1])
with col1:
    nodo_1 = st.selectbox("Nodo principal", nodos_list)
with col2:
    comparar = st.checkbox("Comparar con otro nodo")

nodo_2 = None
if comparar:
    nodo_2 = st.selectbox("Nodo de comparación", [n for n in nodos_list if n != nodo_1])

resolucion = st.selectbox("Resolución temporal", ["Horaria", "Diaria", "Mensual", "Anual"])

def aggregate(df_node):
    if resolucion == "Horaria":
        return df_node[["datetime", "precio"]]
    rule = {"Diaria": "D", "Mensual": "M", "Anual": "Y"}[resolucion]
    return (
        df_node.set_index("datetime")
              .resample(rule)["precio"].mean()
              .reset_index()
    )

series = []
df1 = aggregate(df[df["nodo"] == nodo_1].copy())
df1["Nodo"] = nodo_1
series.append(df1)

if comparar and nodo_2:
    df2 = aggregate(df[df["nodo"] == nodo_2].copy())
    df2["Nodo"] = nodo_2
    series.append(df2)

df_plot = pd.concat(series, ignore_index=True)

chart = (
    alt.Chart(df_plot)
      .mark_line()
      .encode(
          x="datetime:T",
          y=alt.Y("precio:Q", title="Precio [USD/MWh]"),
          color=alt.Color("Nodo:N", legend=alt.Legend(title="Nodo")),
          tooltip=["datetime:T", "Nodo:N", "precio:Q"]
      )
      .properties(height=400)
)

st.altair_chart(chart, use_container_width=True)
