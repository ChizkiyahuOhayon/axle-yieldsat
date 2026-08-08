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


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a markdown table without pandas' optional ``tabulate`` dependency.

    ``DataFrame.to_markdown`` needs tabulate, which is not worth a hard dependency (and
    its absence used to crash this script *after* the csv was already written).
    """
    def cell(v):
        return floatfmt.format(v) if isinstance(v, float) else str(v)

    head = list(df.columns)
    rows = [[cell(v) for v in row] for row in df.itertuples(index=False)]
    width = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
             for i, h in enumerate(head)]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, width)) + " |"
    return "\n".join([line(head), "|" + "|".join("-" * (w + 2) for w in width) + "|",
                      *(line(r) for r in rows)]) + "\n"


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

    # collapse repeated (config x seed) runs so each config appears once, and report the
    # spread *across seeds* -- the error bar a paper needs, distinct from the fold std
    mean_col = f"{args.metric}_mean"
    seed_std_col = f"{args.metric}_seed_std"
    num = [c for c in df.select_dtypes("number").columns if c != "seed"]
    grp = df.groupby(KEYS, as_index=False, dropna=False)
    spread = grp.agg(**{seed_std_col: (mean_col, "std"), "n_seeds": ("seed", "nunique")}) \
        if mean_col in df else None
    df = grp[num].mean()
    if spread is not None:
        df = df.merge(spread, on=KEYS, how="left")

    show = [c for c in [*KEYS, mean_col, seed_std_col, "n_seeds", f"{args.metric}_std",
                        "pixel_r2_mean", "pixel_picp90_mean", "reliability_gap_mean",
                        "n_folds"] if c in df]
    table = df[show].sort_values([c for c in KEYS if c in df]).reset_index(drop=True)
    pd.set_option("display.width", 180, "display.max_columns", 40)
    print(table.to_string(index=False))

    df.to_csv(f"{args.out}.csv", index=False)
    Path(f"{args.out}.md").write_text(to_markdown(table))
    if mean_col in df:
        piv = df.pivot_table(index=["data", "model", "crop", "members", "loss"],
                             columns="protocol", values=mean_col)
        print(f"\n=== {mean_col} by loss x protocol (members separated) ===")
        print(piv.to_string())
    print(f"\nwrote {args.out}.csv and {args.out}.md ({len(df)} configs)")


if __name__ == "__main__":
    main()
