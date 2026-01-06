#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd


def sanity(input_parquet: Path, output_flags: Path):
    df = pd.read_parquet(input_parquet)

    # --- 1) Precios negativos ---
    neg = df["precio"] < 0

    # --- 2) Outliers duros (percentil 99.9 por nodo) ---
    q = (
        df.groupby("nodo")["precio"]
        .quantile(0.999)
        .rename("p999")
        .reset_index()
    )
    df = df.merge(q, on="nodo", how="left")
    hard_outlier = df["precio"] > df["p999"]

    # --- 3) Saltos horarios absurdos ---
    df = df.sort_values(["nodo", "datetime"])
    df["delta"] = df.groupby("nodo")["precio"].diff().abs()
    spike = df["delta"] > 300  # umbral conservador (ajustable)

    # --- Flags ---
    flags = pd.DataFrame({
        "neg_price": neg,
        "hard_outlier": hard_outlier,
        "hourly_spike": spike,
    })

    df_flags = df[["datetime", "nodo", "precio"]].copy()
    df_flags = pd.concat([df_flags, flags], axis=1)

    # --- Resumen ---
    print("✅ SANITY CHECKS (A3)")
    print(f"Precios negativos : {int(neg.sum()):,}")
    print(f"Outliers duros    : {int(hard_outlier.sum()):,}")
    print(f"Saltos horarios   : {int(spike.sum()):,}")

    # Guardar flags
    output_flags.parent.mkdir(parents=True, exist_ok=True)
    df_flags.to_parquet(output_flags, index=False)
    print(f"💾 Flags guardadas en: {output_flags}")


def main():
    ap = argparse.ArgumentParser(description="Sanity checks precios nodales")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sanity(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
