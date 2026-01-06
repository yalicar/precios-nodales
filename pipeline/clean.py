#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd


def clean(input_parquet: Path, output_parquet: Path):
    df = pd.read_parquet(input_parquet)

    # --- 1) Resolver duplicados (quedarse con el último) ---
    before = len(df)
    df = df.sort_values("datetime")
    df = df.drop_duplicates(subset=["datetime", "nodo"], keep="last")
    after = len(df)

    # --- 2) Stats de calidad por nodo ---
    stats = (
        df.groupby("nodo")["precio"]
        .agg(
            n="size",
            n_nan=lambda s: int(s.isna().sum()),
            min="min",
            max="max",
        )
        .reset_index()
    )
    stats["nan_pct"] = (stats["n_nan"] / stats["n"]) * 100.0

    # Flags de calidad (NO se eliminan)
    stats["is_dead"] = stats["nan_pct"] == 100.0
    stats["is_low_coverage"] = stats["nan_pct"] >= 90.0

    # --- 3) Guardar dataset completo ---
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)

    # Guardar metadata de nodos
    stats_path = output_parquet.parent / "node_quality.parquet"
    stats.to_parquet(stats_path, index=False)

    print("✅ LIMPIEZA A2 COMPLETADA (SIN BORRAR NODOS)")
    print(f"Filas antes     : {before:,}")
    print(f"Filas después   : {after:,}")
    print(f"Duplicados resueltos: {before - after:,}")
    print(f"Nodos totales   : {stats.shape[0]}")
    print(f"Nodos 100% NaN  : {int(stats['is_dead'].sum())}")
    print(f"Nodos ≥90% NaN  : {int(stats['is_low_coverage'].sum())}")
    print(f"Salida datos    : {output_parquet}")
    print(f"Salida calidad  : {stats_path}")


def main():
    ap = argparse.ArgumentParser(description="Limpieza precios nodales (no destructiva)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    clean(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
