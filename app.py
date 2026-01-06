import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import altair as alt
from pathlib import Path

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 Acceso restringido")

    password = st.text_input(
        "Ingrese contraseña",
        type="password"
    )

    if password:
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta")

    return False


if not check_password():
    st.stop()


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Precios Nodales Honduras",
    layout="wide",
)

st.title("📍 Precios Nodales – Análisis Espacial y Temporal")

ROOT = Path(__file__).resolve().parent

# ============================================================
# CARGA DE DATOS
# ============================================================
@st.cache_data
def load_data():
    prices = pd.read_parquet(ROOT / "data_processed" / "precios_nodales_clean.parquet")
    quality = pd.read_parquet(ROOT / "data_processed" / "node_quality.parquet")
    nodes = pd.read_csv(ROOT / "data_static" / "nodes_real.csv")


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
    """Volatilidad robusta = P90 - P10"""
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

hour_start, hour_end = st.sidebar.slider(
    "Rango de horas",
    0, 23, (0, 23)
)

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
    poe = st.sidebar.slider(
        "POE – Probabilidad de excedencia (%)",
        5, 95, 20, step=5
    )
    st.sidebar.caption(
        "POE bajo → precios altos raros | POE alto → precios bajos frecuentes"
    )

st.sidebar.header("⚙️ Calidad")

show_low_coverage = st.sidebar.checkbox(
    "Mostrar nodos con baja cobertura (≥90% NaN)",
    value=False,
)

# ============================================================
# FILTRADO BASE
# ============================================================
df_filt = df[
    (df["date"] >= date_start) &
    (df["date"] <= date_end) &
    (df["hour"] >= hour_start) &
    (df["hour"] <= hour_end)
]

df_filt = df_filt[df_filt["is_dead"] != True]

if not show_low_coverage:
    df_filt = df_filt[df_filt["is_low_coverage"] != True]

df_filt = df_filt.dropna(subset=["precio", "lat", "lon"])

if df_filt.empty:
    st.warning("No hay datos para el rango seleccionado.")
    st.stop()

# ============================================================
# ESTADÍSTICAS BASE POR NODO
# ============================================================
df_stats = (
    df_filt
    .groupby("nodo")
    .agg(
        precio_promedio=("precio", "mean"),
        precio_min=("precio", "min"),
        precio_max=("precio", "max"),
        cobertura_nan=("nan_pct", "mean"),
        precio_poe=("precio", lambda x: robust_exceedance_price(x, poe)),
        volatilidad=("precio", robust_volatility),
    )
    .reset_index()
)

# ============================================================
# DATA PARA MAPA
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

df_map = df_map.merge(
    df[["nodo", "lat", "lon"]].drop_duplicates(),
    on="nodo",
    how="left"
).dropna(subset=["valor", "lat", "lon"])

# ============================================================
# ESCALADO VISUAL
# ============================================================
v_min, v_max = df_map["valor"].min(), df_map["valor"].max()
df_map["norm"] = (df_map["valor"] - v_min) / (v_max - v_min) if v_max != v_min else 0
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

        tooltip={
            "html": f"""
                <b>{{nodo}}</b><br/>
                {metric_label}: <b>{{valor:.2f}}</b><br/>
                Cobertura NaN: {{cobertura_nan:.1f}} %
            """
        },
    )
)

# ============================================================
# TABLA ANALÍTICA
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
# SERIE TEMPORAL (CON COMPARACIÓN)
# ============================================================
st.subheader("📈 Serie temporal por nodo")

nodos = sorted(df_filt["nodo"].unique())

col1, col2 = st.columns([2, 1])

with col1:
    nodo_1 = st.selectbox("Nodo principal", nodos)

with col2:
    comparar = st.checkbox("Comparar con otro nodo")

nodo_2 = None
if comparar:
    nodo_2 = st.selectbox(
        "Nodo de comparación",
        [n for n in nodos if n != nodo_1]
    )

resolucion = st.selectbox(
    "Resolución temporal",
    ["Horaria", "Diaria", "Mensual", "Anual"]
)

# -------- resumen nodo principal ----------
row = df_stats[df_stats["nodo"] == nodo_1].iloc[0]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Promedio", f"{row.precio_promedio:.2f}")
c2.metric("Mínimo", f"{row.precio_min:.2f}")
c3.metric("Máximo", f"{row.precio_max:.2f}")
c4.metric(f"POE {poe}%", f"{row.precio_poe:.2f}")
c5.metric("Volatilidad", f"{row.volatilidad:.2f}")
c6.metric("NaN (%)", f"{row.cobertura_nan:.1f}")

# -------- agregación ----------
def aggregate(df_node):
    if resolucion == "Horaria":
        return df_node[["datetime", "precio"]]
    rule = {"Diaria": "D", "Mensual": "M", "Anual": "Y"}[resolucion]
    return (
        df_node
        .set_index("datetime")
        .resample(rule)["precio"]
        .mean()
        .reset_index()
    )

series = []

df1 = aggregate(df_filt[df_filt["nodo"] == nodo_1])
df1["Nodo"] = nodo_1
series.append(df1)

if comparar and nodo_2:
    df2 = aggregate(df_filt[df_filt["nodo"] == nodo_2])
    df2["Nodo"] = nodo_2
    series.append(df2)

df_plot = pd.concat(series)

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
