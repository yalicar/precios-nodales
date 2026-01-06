import streamlit as st
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
from auth import check_auth

if not check_auth():
    st.error("🔐 Debe iniciar sesión para acceder a esta página")
    st.stop()

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="Actualizar datos", layout="centered")
st.title("🔄 Actualización de datos")

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data_raw" / "Costos_Marginales.xlsx"
PROCESSED = ROOT / "data_processed"
META = ROOT / "metadata.json"

ETL = ROOT / "pipeline" / "etl.py"
CLEAN = ROOT / "pipeline" / "clean.py"
SANITY = ROOT / "pipeline" / "sanity.py"

# ============================================================
# METADATA
# ============================================================
def load_meta():
    default = {
        "raw_file": str(RAW),
        "last_run": None,
        "status": "never",
    }

    if not META.exists():
        return default

    try:
        text = META.read_text().strip()
        if not text:
            return default
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def save_meta(meta):
    META.write_text(json.dumps(meta, indent=2))


meta = load_meta()

st.subheader("📌 Estado actual")
st.json({
    "Archivo activo": Path(meta["raw_file"]).name,
    "Última ejecución": meta["last_run"],
    "Estado": meta["status"],
})

# ============================================================
# UPLOAD DE ARCHIVO
# ============================================================
st.divider()
st.subheader("📤 Cargar nuevo archivo Excel")

uploaded_file = st.file_uploader(
    "Selecciona el archivo Excel de costos marginales",
    type=["xlsx"],
)

if uploaded_file is not None:
    RAW.parent.mkdir(exist_ok=True)

    with open(RAW, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(
        f"Archivo cargado correctamente: {uploaded_file.name}\n\n"
        "Ahora puedes ejecutar el pipeline."
    )

# ============================================================
# EJECUCIÓN DEL PIPELINE
# ============================================================
st.divider()
st.subheader("▶ Ejecutar pipeline")

st.info(
    "Usa este botón cuando el archivo Excel haya sido actualizado.\n"
    "Se ejecutará ETL → limpieza → sanity checks."
)

def run_step(name, cmd):
    st.write(f"▶ Ejecutando **{name}**")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        st.error(f"❌ Error en {name}")
        if result.stdout:
            st.code(result.stdout, language="text")
        if result.stderr:
            st.code(result.stderr, language="text")
        raise RuntimeError(f"Falló {name}")

    if result.stdout:
        st.code(result.stdout, language="text")


if st.button("Actualizar datos ahora"):
    if not RAW.exists():
        st.error("❌ No se encontró el archivo Excel")
        st.stop()

    with st.spinner("Ejecutando pipeline completo..."):
        try:
            PROCESSED.mkdir(exist_ok=True)

            run_step(
                "ETL",
                [
                    sys.executable, str(ETL),
                    "--input", str(RAW),
                    "--output", str(PROCESSED / "precios_nodales.parquet"),
                ],
            )

            run_step(
                "Limpieza",
                [
                    sys.executable, str(CLEAN),
                    "--input", str(PROCESSED / "precios_nodales.parquet"),
                    "--output", str(PROCESSED / "precios_nodales_clean.parquet"),
                ],
            )

            run_step(
                "Sanity checks",
                [
                    sys.executable, str(SANITY),
                    "--input", str(PROCESSED / "precios_nodales_clean.parquet"),
                    "--output", str(PROCESSED / "precios_nodales_flags.parquet"),
                ],
            )

            meta.update({
                "last_run": datetime.utcnow().isoformat(),
                "status": "ok",
            })
            save_meta(meta)

            st.success("✅ Pipeline ejecutado correctamente")
            st.cache_data.clear()

        except Exception as e:
            meta["status"] = "error"
            save_meta(meta)
            st.error("❌ Pipeline falló")
            st.exception(e)
