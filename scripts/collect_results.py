"""Collect metrics.json files under a directory into one results table.

Scans recursively for ``metrics.json`` (written by ``axle.train``), builds a tidy
table keyed by (data, model, loss, protocol, crop), and prints it plus a pivot of
the headline metric. Writes ``results.csv`` and a markdown table for the paper.

    python scripts/collect_results.py outputs/ multirun/ --metric field_r2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(dirs: list[str]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for mj in Path(d).rglob("metrics.json"):
            try:
                m = json.loads(mj.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            run = m.get("run", {})
            row = {**{k: run.get(k) for k in ("data", "model", "loss", "protocol", "crop", "seed")},
                   "path": str(mj.parent)}
            row.update({k: v for k, v in m.items() if isinstance(v, (int, float))})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="directories to scan (e.g. outputs/ multirun/)")
    ap.add_argument("--metric", default="field_r2", help="headline metric for the pivot (base name)")
    ap.add_argument("--out", default="results", help="output basename (writes .csv and .md)")
    args = ap.parse_args()

    df = load(args.dirs)
    if df.empty:
        print("no metrics.json found under:", ", ".join(args.dirs))
        return
    keys = ["data", "model", "loss", "protocol", "crop"]
    mean_col = f"{args.metric}_mean"
    show = [c for c in [*keys, mean_col, f"{args.metric}_std", "pixel_r2_mean",
                        "pixel_picp90_mean", "reliability_gap_mean", "n_folds"] if c in df]
    table = df[show].sort_values([c for c in keys if c in df]).reset_index(drop=True)
    pd.set_option("display.width", 160, "display.max_columns", 40)
    print(table.to_string(index=False))

    df.to_csv(f"{args.out}.csv", index=False)
    Path(f"{args.out}.md").write_text(table.to_markdown(index=False))
    # headline pivot: loss (rows) x protocol (cols), averaged over seeds
    if mean_col in df:
        piv = df.pivot_table(index=["data", "model", "crop", "loss"], columns="protocol",
                             values=mean_col, aggfunc="mean")
        print(f"\n=== {mean_col} by loss x protocol ===")
        print(piv.to_string())
    print(f"\nwrote {args.out}.csv and {args.out}.md ({len(df)} runs)")


if __name__ == "__main__":
    main()
