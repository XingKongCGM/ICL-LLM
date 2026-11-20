#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oor_extract.py

Created on Tue Aug 26 14:17:09 2025

@author: Carlos Gonzalez
Material Data Division,
National Institute of Standatds and Technology,
Gaithersburg, MD. 20889


==============

Pull out-of-range and reject statistics from the blinded cells of a
blinding run, so they can be reported without shipping binary .npz
files anywhere.

Out-of-range (OOR) counts exist only for blinded (generic/agnostic)
cells; the native/identity arm cannot be out of range and is skipped.
With the zscore transform every output is in range by construction, so
OOR counts are zero and only rejects are informative.

Rates
-----
Each of the M_c cycles scores the full test partition once, so the
number of TEST queries in a cell is  n_test * n_cycles  and

    oor_rate_test = sum(n_out_of_range_test) / (n_test * n_cycles)
    rej_rate_test = sum(n_rejects_test)      / (n_test * n_cycles)

where n_cycles counts cycles that produced metrics.  n_test is read from
the cell's testing_summary.txt ("Testing dataset: N molecules"); if it
cannot be found the rates are left blank and only counts are reported.

Usage
-----
    python oor_extract.py --run-dir RESULTS/blinding_runs/{run_id}
    python oor_extract.py --results-root RESULTS --run-id {run_id}

Writes oor_summary.csv into the run directory and prints a table.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np


_N_TEST_RE = re.compile(r"Testing dataset:\s*(\d+)\s+molecules")


def _n_test_from_summary(cell_dir):
    """Return the test-partition size recorded in testing_summary.txt,
    or None if unavailable."""
    p = Path(cell_dir) / "testing_summary.txt"
    if not p.exists():
        return None
    m = _N_TEST_RE.search(p.read_text(errors="replace"))
    return int(m.group(1)) if m else None


def load_cell(npz_path):
    # Strings are stored as unicode arrays; pickle is never needed.
    with np.load(npz_path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}

    def scalar(k, default=np.nan):
        if k not in d:
            return default
        v = d[k]
        return v.item() if getattr(v, "ndim", 1) == 0 else v

    cond = str(scalar("condition", ""))
    trans = str(scalar("transform", ""))
    oor_test = np.asarray(d.get("n_out_of_range_test", []), dtype=float)
    rej_test = np.asarray(d.get("n_rejects_test", []), dtype=float)
    n_cycles = int(np.sum(~np.isnan(oor_test))) if oor_test.size else 0
    oor_total = float(np.nansum(oor_test)) if oor_test.size else float("nan")
    rej_total = float(np.nansum(rej_test)) if rej_test.size else float("nan")

    n_test = _n_test_from_summary(Path(npz_path).parent)
    if n_test and n_cycles:
        denom = n_test * n_cycles
        oor_rate = oor_total / denom
        rej_rate = rej_total / denom
    else:
        oor_rate = rej_rate = float("nan")

    def _stat(fn, a):
        a = a[~np.isnan(a)]
        return float(fn(a)) if a.size else float("nan")

    return {
        "condition": cond,
        "transform": trans,
        "n_cycles": n_cycles,
        "n_test": n_test if n_test is not None else "",
        "oor_test_total": oor_total,
        "oor_test_mean_per_cycle": _stat(np.mean, oor_test),
        "oor_test_max_per_cycle": _stat(np.max, oor_test),
        "oor_rate_test": oor_rate,
        "rej_test_total": rej_total,
        "rej_rate_test": rej_rate,
        "oor_val": float(np.asarray(scalar("n_out_of_range_val", np.nan))),
        "rej_val": float(np.asarray(scalar("n_rejects_val", np.nan))),
        "mae_val": float(np.asarray(scalar("MAE_val", np.nan))),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Out-of-range / reject summary for a blinding run.")
    ap.add_argument("--run-dir", default=None,
                    help="Path to RESULTS/blinding_runs/{run_id}")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.results_root and args.run_id:
        run_dir = Path(args.results_root) / "blinding_runs" / args.run_id
    else:
        ap.error("Give either --run-dir, or both --results-root and "
                 "--run-id.")
    if not run_dir.is_dir():
        ap.error(f"Run directory not found: {run_dir}")

    rows = []
    for npz_path in sorted(glob.glob(str(run_dir / "*" / "cycles_blinded.npz"))):
        info = load_cell(npz_path)
        if info["condition"].lower() in ("native", ""):
            continue  # native arm has no out-of-range concept
        info["cell"] = Path(npz_path).parent.name
        rows.append(info)

    if not rows:
        print("No blinded cells found under", run_dir)
        return

    out_csv = run_dir / "oor_summary.csv"
    cols = ["cell", "condition", "transform", "n_cycles", "n_test",
            "oor_test_total", "oor_test_mean_per_cycle",
            "oor_test_max_per_cycle", "oor_rate_test",
            "rej_test_total", "rej_rate_test",
            "oor_val", "rej_val", "mae_val"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    def pct(x):
        return f"{100 * x:.2f}%" if x == x else "n/a"

    print(f"\nOut-of-range / reject summary  ({len(rows)} blinded cells)")
    print("=" * 100)
    print(f"{'cell':<40}{'cyc':>4}{'oor_tot':>9}{'oor/cyc':>9}"
          f"{'oor_rate':>10}{'rej_tot':>9}{'rej_rate':>10}{'oor_val':>9}")
    print("-" * 100)
    for r in rows:
        print(f"{r['cell']:<40}{r['n_cycles']:>4}"
              f"{r['oor_test_total']:>9.0f}"
              f"{r['oor_test_mean_per_cycle']:>9.2f}"
              f"{pct(r['oor_rate_test']):>10}"
              f"{r['rej_test_total']:>9.0f}"
              f"{pct(r['rej_rate_test']):>10}"
              f"{r['oor_val']:>9.0f}")
    print("=" * 100)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
