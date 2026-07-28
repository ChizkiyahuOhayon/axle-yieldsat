"""Collect metrics.json files under a directory into one results table.

Scans recursively for ``metrics.json`` (written by ``axle.train``), builds a tidy
table keyed by (data, model, loss, protocol, crop, members), and prints it plus a
pivot of the headline metric. Writes ``results.csv`` and a markdown table.

``members`` is part of the key so single models (members=1) and deep ensembles
(members>1) never merge. Rows without a ``run`` block (e.g. stale synthetic-demo
outputs from before this field existed) are dropped with a warning.

    python scripts/collect_results.py multirun/ --metric field_r2
    python scripts/collect_results.py multirun/ outputs/ --metric field_r2 --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

KEYS = ["data", "model", "loss", "protocol", "crop", "members"]


def load(dirs: list[str]) -> tuple[pd.DataFrame, int]:
    rows, dropped = [], 0
    for d in dirs:
        for mj in Path(d).rglob("metrics.json"):
            try:
                m = json.loads(mj.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            run = m.get("run")
            if not run or run.get("data") is None:  # stale / pre-run-block file
                dropped += 1
                continue
            row = {**{k: run.get(k) for k in [*KEYS, "seed"]}, "path": str(mj.parent)}
            row.update({k: v for k, v in m.items() if isinstance(v, (int, float))})
            rows.append(row)
    return pd.DataFrame(rows), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="directories to scan (e.g. multirun/ outputs/)")
    ap.add_argument("--metric", default="field_r2", help="headline metric for the pivot (base name)")
    ap.add_argument("--out", default="results", help="output basename (writes .csv and .md)")
    args = ap.parse_args()

    df, dropped = load(args.dirs)
    if dropped:
        print(f"[note] skipped {dropped} metrics.json without a run block (stale/demo outputs)")
    if df.empty:
        print("no usable metrics.json found under:", ", ".join(args.dirs))
        return

    # average any repeated (config x seed) runs so each config appears once
    num = df.select_dtypes("number").columns
    df = df.groupby(KEYS, as_index=False, dropna=False)[list(num)].mean()

    mean_col = f"{args.metric}_mean"
    show = [c for c in [*KEYS, mean_col, f"{args.metric}_std", "pixel_r2_mean",
                        "pixel_picp90_mean", "reliability_gap_mean", "n_folds"] if c in df]
    table = df[show].sort_values([c for c in KEYS if c in df]).reset_index(drop=True)
    pd.set_option("display.width", 180, "display.max_columns", 40)
    print(table.to_string(index=False))

    df.to_csv(f"{args.out}.csv", index=False)
    Path(f"{args.out}.md").write_text(table.to_markdown(index=False))
    if mean_col in df:
        piv = df.pivot_table(index=["data", "model", "crop", "members", "loss"],
                             columns="protocol", values=mean_col)
        print(f"\n=== {mean_col} by loss x protocol (members separated) ===")
        print(piv.to_string())
    print(f"\nwrote {args.out}.csv and {args.out}.md ({len(df)} configs)")


if __name__ == "__main__":
    main()
