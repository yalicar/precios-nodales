#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd


def validate(parquet_path: Path):
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)

    required = {"datetime", "nodo", "precio"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {missing}. Encontradas: {df.columns.tolist()}")

    # Tipos básicos
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if df["datetime"].isna().any():
        bad = df[df["datetime"].isna()].head(10)
        raise ValueError(f"Hay datetime inválidos. Ejemplos:\n{bad}")

    # 1) Duplicados por clave (datetime, nodo)
    dup_mask = df.duplicated(subset=["datetime", "nodo"], keep=False)
    dup_count = int(dup_mask.sum())

    # 2) Nodos “muertos” (todo NaN o todo 0 si quedaron)
    by_node = df.groupby("nodo", dropna=False)["precio"]
    node_stats = by_node.agg(
        n="size",
        n_nan=lambda s: int(s.isna().sum()),
        n_zero=lambda s: int((s == 0).sum()),
        min="min",
        max="max",
    )
    node_stats["nan_pct"] = (node_stats["n_nan"] / node_stats["n"]) * 100.0

    dead_all_nan = node_stats[node_stats["n_nan"] == node_stats["n"]].sort_values("n", ascending=False)
    mostly_nan = node_stats[(node_stats["nan_pct"] >= 90) & (node_stats["n_nan"] < node_stats["n"])].sort_values("nan_pct", ascending=False)

    # 3) Resumen global
    dt_min = df["datetime"].min()
    dt_max = df["datetime"].max()
    n_rows = len(df)
    n_nodes = df["nodo"].nunique(dropna=True)
    nan_total = int(df["precio"].isna().sum())
    nan_pct_total = (nan_total / n_rows) * 100.0

    print("✅ VALIDACIÓN (A1)")
    print(f"Archivo      : {parquet_path}")
    print(f"Filas        : {n_rows:,}")
    print(f"Nodos únicos : {n_nodes:,}")
    print(f"Rango tiempo : {dt_min} → {dt_max}")
    print(f"Precio NaN   : {nan_total:,} ({nan_pct_total:.2f}%)")
    print(f"Duplicados (datetime,nodo): {dup_count:,}")

    # Mostrar top problemas (sin spam)
    if dup_count:
        print("\n⚠️ Ejemplo duplicados (10 filas):")
        print(df.loc[dup_mask, ["datetime", "nodo", "precio"]].head(10))

    print("\n📌 Top 10 nodos con más NaN:")
    print(node_stats.sort_values("n_nan", ascending=False).head(10))

    if len(dead_all_nan):
        print(f"\n🚫 Nodos 100% NaN: {len(dead_all_nan)} (top 10)")
        print(dead_all_nan.head(10))

    if len(mostly_nan):
        print(f"\n⚠️ Nodos >=90% NaN: {len(mostly_nan)} (top 10)")
        print(mostly_nan.head(10))

    # Guardar stats para siguiente fase
    out = parquet_path.parent / "node_stats.parquet"
    node_stats.reset_index().to_parquet(out, index=False)
    print(f"\n💾 Guardé stats por nodo en: {out}")


def main():
    ap = argparse.ArgumentParser(description="Validación de precios nodales (parquet tidy)")
    ap.add_argument("--input", required=True, help="Ruta al out/precios_nodales.parquet")
    args = ap.parse_args()
    validate(Path(args.input))


if __name__ == "__main__":
    main()
