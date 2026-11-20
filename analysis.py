#!/usr/bin/env python3
"""
analysis.py

Created on Tue Aug 26 14:17:09 2025

@author: Carlos Gonzalez
Material Data Division,
National Institute of Standatds and Technology,
Gaithersburg, MD. 20889


============

Post-hoc analysis of a completed ICL sweep run.  Consumes the NPZ files
and summary.csv in a run directory and produces:

  * Per-property MAE box plots across models
  * Scaling fits:  log MAE  vs.  log(effective params)
  * Property-difficulty ranking with permutation-null baseline
  * Property-property correlation matrix (across models)
  * IQR / quartile skewness table
  * Mutual information estimates (Gaussian rate-distortion lower bound)
  * A framework-ready summary CSV with normalized metrics

Metrics used
------------

For each cell we report five (mostly) independent scores:

  median MAE       robust central tendency of the 50 per-cycle MAEs
  IQR of MAE       Q3 - Q1 across cycles, robust reproducibility measure
  R^2 (validation) coefficient of determination on the held-out val set
  Info in nats     Gaussian rate-distortion LOWER bound on I(Y ; Y_hat):
                       I >= (1/2) ln( Var(Y) / MSE )
                   (exact bound for Gaussian Y; here Var(Y) is
                   approximated from the range, so treat as an estimate)
  Binary accuracy  ordinal-ranking skill against dataset median

If a permutation-control cell exists for the same (model, property),
we normalize each metric by that null.  For MAE this is

    skill(m, p) = 1 - MAE(m, p) / MAE_null(p).

Usage
-----

Either point at the run directory directly:

    python analysis.py --run-dir /path/to/RESULTS/runs/{run_id}

or give the results root and run id (the form used by run_icl_sweep.sh
and icl_dashboard.py):

    python analysis.py --results-root /path/to/RESULTS --run-id {run_id}

Output goes into  {results_root}/analysis/{run_id}/  (override with
--analysis-root), created if it doesn't exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model effective-parameter counts (used for scaling fits).  For MoE
# models we use ACTIVE parameters per token, not total capacity, as
# this is what's actually processing each query.  Note the 400 B
# stored capacity of Maverick is still relevant for retrieval-style
# ICL; a curious reader may want to plot both.
# ---------------------------------------------------------------------------

MODEL_PARAMS_ACTIVE_B = {
    "llama3.1-8b-q4"            :   8.0,
    "llama3.3-70b-q4"           :  70.0,
    "nemotron-mini-4b-bf16"     :   4.0,
    "llama3.1-nemotron-70b-q4"  :  70.0,
    "gemma-4-31b-optiq-q4"      :  31.0,
    "mistral-small-3.1-24b-q4"  :  24.0,
    "maverick-17b-128e-fp8"     :  17.0,   # active per token
    "llama4-scout-17b-16e-q4"   :  17.0,   # active per token (same as Maverick)
}

MODEL_PARAMS_TOTAL_B = {
    "llama3.1-8b-q4"            :   8.0,
    "llama3.3-70b-q4"           :  70.0,
    "nemotron-mini-4b-bf16"     :   4.0,
    "llama3.1-nemotron-70b-q4"  :  70.0,
    "gemma-4-31b-optiq-q4"      :  31.0,
    "mistral-small-3.1-24b-q4"  :  24.0,
    "maverick-17b-128e-fp8"     : 400.0,   # total across 128 experts
    "llama4-scout-17b-16e-q4"   : 109.0,   # total across 16 experts
}


# ---------------------------------------------------------------------------
# NPZ-level loading
# ---------------------------------------------------------------------------

def _as_bool(v):
    """Parse a summary.csv boolean robustly.  pandas may hand back a real
    bool, a numpy bool, or the strings 'True'/'False' (bool('False') is
    True, so a plain bool() cast is wrong for strings)."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    return bool(v)


def load_run(run_dir):
    """Return (summary_df, cells_dict) where cells_dict maps
    (model_slug, property_key, permutation_bool) -> numpy dict."""
    run_dir = Path(run_dir)
    summary_csv = run_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"No summary.csv in {run_dir}")

    summary_df = pd.read_csv(summary_csv)

    cells = {}
    for _, row in summary_df.iterrows():
        slug = row["model_slug"]
        prop = row["property"]
        perm = _as_bool(row.get("permutation", False))
        if (slug, prop, perm) in cells:
            continue   # duplicate row (e.g. a re-run appended to the CSV)
        folder = f"{slug}_{prop}" + ("_perm" if perm else "")
        npz_path = run_dir / folder / "cycles.npz"
        if npz_path.exists():
            data = dict(np.load(npz_path, allow_pickle=False))
            cells[(slug, prop, perm)] = data
        else:
            print(f"[warn] Missing cycles.npz for {folder}")

    return summary_df, cells


# ---------------------------------------------------------------------------
# Metrics computed post-hoc from the per-cycle arrays
# ---------------------------------------------------------------------------

def cell_stats(cell_arrays):
    """Compute the 5-metric summary + IQR + Bowley skewness from a
    single cell's cycles.npz dict.  Returns a dict of scalars."""
    mae = np.asarray(cell_arrays["MAE_test"], dtype=float)
    mae_valid = mae[~np.isnan(mae)]

    def _scalar(key):
        try:
            return float(cell_arrays.get(key, np.nan))
        except (TypeError, ValueError):
            return float("nan")

    # Mutual information, Gaussian rate-distortion LOWER bound:
    #     I(Y; Y_hat) >= (1/2) ln( Var(Y) / MSE )
    # (exact for Gaussian Y at mean-squared distortion MSE).
    y_min = _scalar("y_min")
    y_max = _scalar("y_max")
    # Use validation MSE (single scalar) for the info bound
    mse_val = _scalar("MSE_val")
    # Var(Y) is approximated from the range (range ~ 4 sigma for
    # near-normal data, so Var ~ ((y_max - y_min)/4)^2).  For a proper
    # estimate we'd need the raw y values; if the framework wants
    # tight bounds this is where to add them.
    if (y_max - y_min) > 0 and mse_val > 0 and not np.isnan(mse_val):
        var_approx = ((y_max - y_min) / 4.0) ** 2
        ratio = var_approx / mse_val
        info_nats = 0.5 * np.log(ratio) if ratio > 0 else float("nan")
    else:
        info_nats = float("nan")

    if len(mae_valid) < 4:
        # Not enough valid cycles for meaningful quartiles.  Return the
        # SAME key set as the normal path so the framework table always
        # has every column.
        return {
            "median_MAE"    : float("nan"),
            "IQR_MAE"       : float("nan"),
            "bowley_skew"   : float("nan"),
            "R2_val"        : _scalar("R2_val"),
            "MAE_val"       : _scalar("MAE_val"),
            "MSE_val"       : mse_val,
            "Acc_val"       : _scalar("Acc_val"),
            "F1_val"        : _scalar("F1_val"),
            "NRMSE_range_val": _scalar("NRMSE_range_val"),
            "info_nats_val" : info_nats,
            "n_valid_cycles": len(mae_valid),
        }

    q1, q2, q3 = np.percentile(mae_valid, [25, 50, 75])
    iqr = q3 - q1
    # Bowley skewness: [(Q3 - Q2) - (Q2 - Q1)] / IQR
    bowley = ((q3 - q2) - (q2 - q1)) / iqr if iqr > 0 else float("nan")

    return {
        "median_MAE"    : float(q2),
        "IQR_MAE"       : float(iqr),
        "bowley_skew"   : float(bowley),
        "R2_val"        : float(cell_arrays.get("R2_val", np.nan)),
        "MAE_val"       : float(cell_arrays.get("MAE_val", np.nan)),
        "MSE_val"       : mse_val,
        "Acc_val"       : float(cell_arrays.get("Acc_val", np.nan)),
        "F1_val"        : float(cell_arrays.get("F1_val", np.nan)),
        "NRMSE_range_val": float(cell_arrays.get("NRMSE_range_val", np.nan)),
        "info_nats_val" : info_nats,
        "n_valid_cycles": len(mae_valid),
    }


# ---------------------------------------------------------------------------
# Framework table: one row per (model, property) with all reportable
# metrics, normalized against permutation null where available
# ---------------------------------------------------------------------------

def build_framework_table(summary_df, cells):
    """Return a DataFrame with one row per non-permutation cell and
    columns for each metric plus a permutation-normalized skill."""
    rows = []
    # Group real vs. null cells for each (model, property)
    real_cells = {(s, p): a for (s, p, perm), a in cells.items()
                  if not perm}
    null_cells = {(s, p): a for (s, p, perm), a in cells.items()
                  if perm}

    for (slug, prop), real in real_cells.items():
        stats_real = cell_stats(real)
        null = null_cells.get((slug, prop))
        stats_null = cell_stats(null) if null is not None else None

        # Metrological skill: (MAE_null - MAE_real) / MAE_null
        if stats_null is not None:
            mae_r = stats_real["median_MAE"]
            mae_n = stats_null["median_MAE"]
            if mae_n > 0 and not np.isnan(mae_r):
                mae_skill = 1.0 - mae_r / mae_n
            else:
                mae_skill = float("nan")
        else:
            mae_skill = float("nan")

        rows.append({
            "model_slug"    : slug,
            "property"      : prop,
            "N_active_B"    : MODEL_PARAMS_ACTIVE_B.get(slug, np.nan),
            "N_total_B"     : MODEL_PARAMS_TOTAL_B.get(slug, np.nan),
            **{k: stats_real[k] for k in stats_real},
            "null_median_MAE": stats_null["median_MAE"] if stats_null else np.nan,
            "MAE_skill"     : mae_skill,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_property_boxplots(cells, out_dir, only_real=True):
    """For each property, box-plot per-cycle MAEs across models."""
    # Group cells by property
    by_prop = {}
    for (slug, prop, perm), arr in cells.items():
        if only_real and perm:
            continue
        by_prop.setdefault(prop, []).append((slug, arr))

    for prop, items in by_prop.items():
        items.sort(key=lambda x: MODEL_PARAMS_ACTIVE_B.get(x[0], 999))
        slugs = [s for s, _ in items]
        maes  = [np.asarray(a["MAE_test"])[~np.isnan(a["MAE_test"])]
                 for _, a in items]
        if not maes or all(len(m) < 4 for m in maes):
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(slugs) * 1.2), 5))
        # matplotlib >= 3.9 renamed `labels` to `tick_labels` (the old
        # name was removed in 3.11); support both.
        try:
            bp = ax.boxplot(maes, tick_labels=slugs, showfliers=True,
                            patch_artist=True)
        except TypeError:
            bp = ax.boxplot(maes, labels=slugs, showfliers=True,
                            patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#3b8bba")
            patch.set_alpha(0.6)
        ax.set_ylabel("MAE per cycle", fontweight="bold")
        ax.set_xlabel("Model", fontweight="bold")
        ax.set_title(f"Per-cycle MAE distribution for {prop}",
                      fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(out_dir / f"boxplot_{prop}.png", dpi=150)
        plt.close(fig)


def plot_scaling_fits(framework_df, out_dir, param_col="N_active_B"):
    """For each property, log-log plot of median MAE vs. model size,
    with a per-property power-law fit and residuals annotated."""
    props = sorted(framework_df["property"].unique())
    n_props = len(props)
    n_cols = 3
    n_rows = int(np.ceil(n_props / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4, n_rows * 3.5),
                              squeeze=False)

    for i, prop in enumerate(props):
        ax = axes[i // n_cols][i % n_cols]
        sub = framework_df[framework_df["property"] == prop]
        sub = sub.dropna(subset=[param_col, "median_MAE"])
        if len(sub) < 2:
            ax.set_title(f"{prop}  (insufficient data)",
                          fontweight="bold")
            continue
        x = np.log10(sub[param_col].values)
        y = np.log10(sub["median_MAE"].values)
        # Fit  log10(MAE) = a + b * log10(N)
        b, a = np.polyfit(x, y, 1)
        ax.scatter(sub[param_col], sub["median_MAE"],
                    s=60, c="#c94a3e", zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["model_slug"].replace("-q4", ""),
                         (row[param_col], row["median_MAE"]),
                         fontsize=7, alpha=0.7,
                         xytext=(4, 3), textcoords="offset points")
        xs = np.logspace(x.min(), x.max(), 50)
        ys = 10 ** (a + b * np.log10(xs))
        ax.plot(xs, ys, "--", c="#666", lw=1.2,
                 label=f"$\\alpha = {-b:.2f}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (B)", fontweight="bold")
        ax.set_ylabel("median MAE", fontweight="bold")
        ax.set_title(f"{prop}", fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # Hide unused axes
    for j in range(n_props, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    plt.suptitle(f"ICL scaling per property   MAE $\\propto N^{{-\\alpha}}$",
                  fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / f"scaling_fits_{param_col}.png",
                 dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_property_correlation(framework_df, out_dir):
    """Heatmap of MAE-across-models correlation, property × property."""
    pivot = framework_df.pivot_table(index="model_slug",
                                       columns="property",
                                       values="median_MAE")
    if pivot.shape[1] < 2 or pivot.shape[0] < 2:
        print("[analysis] Not enough cells for property correlation")
        return

    corr = pivot.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(max(6, corr.shape[0] * 0.7),
                                     max(5, corr.shape[0] * 0.6)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}",
                    ha="center", va="center",
                    color="white" if abs(corr.values[i,j]) > 0.5 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, label="Spearman $\\rho$")
    ax.set_title("Property difficulty correlation (across models)",
                  fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "property_correlation_matrix.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc analysis of an ICL sweep run.")
    parser.add_argument("--run-dir", default=None,
                        help="Path to RESULTS/runs/{run_id}. "
                             "Alternative to --results-root + --run-id.")
    parser.add_argument("--results-root", default=None,
                        help="Results root (contains runs/). Use together "
                             "with --run-id.")
    parser.add_argument("--run-id", default=None,
                        help="Run id under {results_root}/runs/.")
    parser.add_argument("--analysis-root", default=None,
                        help="Where to write analysis outputs. "
                             "Default: {results_root}/analysis/{run_id}")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.results_root and args.run_id:
        run_dir = (Path(args.results_root) / "runs" / args.run_id).resolve()
    else:
        parser.error("Give either --run-dir, or both --results-root "
                     "and --run-id.")
    if not run_dir.is_dir():
        parser.error(f"Run directory not found: {run_dir}")
    run_id = run_dir.name

    if args.analysis_root:
        out_dir = Path(args.analysis_root) / run_id
    else:
        results_root = run_dir.parent.parent   # .../runs/{run_id} -> .../
        out_dir = results_root / "analysis" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analysis] Loading run from {run_dir}")
    summary_df, cells = load_run(run_dir)
    print(f"[analysis] Found {len(cells)} cells "
          f"({sum(1 for _,_,p in cells if p)} permutation control)")

    framework_df = build_framework_table(summary_df, cells)
    framework_csv = out_dir / "framework_table.csv"
    framework_df.to_csv(framework_csv, index=False)
    print(f"[analysis] Framework table -> {framework_csv}")

    plot_property_boxplots(cells, out_dir)
    print(f"[analysis] Box plots -> {out_dir}/boxplot_*.png")

    if framework_df.empty:
        print("[analysis] No real (non-permutation) cells; skipping "
              "scaling fits and correlation matrix.")
    else:
        plot_scaling_fits(framework_df, out_dir, param_col="N_active_B")
        plot_scaling_fits(framework_df, out_dir, param_col="N_total_B")
        print(f"[analysis] Scaling fits -> {out_dir}/scaling_fits_*.png")

        plot_property_correlation(framework_df, out_dir)
        print(f"[analysis] Correlation matrix -> "
              f"{out_dir}/property_correlation_matrix.png")

    # Print top-line rollup
    print("\n" + "=" * 68)
    print("  FRAMEWORK ROLLUP")
    print("=" * 68)
    key_cols = ["model_slug", "property", "median_MAE", "IQR_MAE",
                "R2_val", "info_nats_val", "MAE_skill"]
    if framework_df.empty:
        print("(no non-permutation cells with cycles.npz found)")
    else:
        key_cols = [c for c in key_cols if c in framework_df.columns]
        print(framework_df[key_cols].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
