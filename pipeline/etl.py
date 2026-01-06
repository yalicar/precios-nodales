#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd


def normalize_fecha(fecha):
    if pd.isna(fecha):
        return None
    return str(fecha).split(" ")[0]


def normalize_hora(hora):
    if pd.isna(hora):
        return None
    s = str(hora)
    # casos tipo 1900-01-01 03:00:00
    if " " in s:
        s = s.split(" ")[1]
    # dejar HH:MM
    return s[:5]


def run_etl(input_path: Path, output_path: Path, sheet="24 HRS"):
    df = pd.read_excel(input_path, sheet_name=sheet, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    if "Fecha" not in df.columns or "Hora" not in df.columns:
        raise ValueError("Faltan columnas Fecha/Hora")

    # Normalizar Fecha y Hora como strings limpias
    fecha_str = df["Fecha"].apply(normalize_fecha)
    hora_str = df["Hora"].apply(normalize_hora)

    dt_str = fecha_str + " " + hora_str
    df["datetime"] = pd.to_datetime(dt_str, errors="coerce")

    if df["datetime"].isna().any():
        print("⚠️ Filas con datetime inválido:")
        print(df[df["datetime"].isna()][["Fecha", "Hora"]].head(10))
        raise ValueError("No se pudo construir datetime")

    node_cols = [c for c in df.columns if c not in ("Fecha", "Hora", "datetime")]

    tidy = df.melt(
        id_vars=["datetime"],
        value_vars=node_cols,
        var_name="nodo",
        value_name="precio",
    )

    tidy["precio"] = pd.to_numeric(tidy["precio"], errors="coerce")
    tidy.loc[tidy["precio"] == 0, "precio"] = pd.NA

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(output_path, index=False)

    print("✅ ETL OK")
    print(f"Filas tidy: {len(tidy):,}")
    print(f"Rango: {tidy['datetime'].min()} → {tidy['datetime'].max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    run_etl(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
