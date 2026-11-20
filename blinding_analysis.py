#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blinding_analysis.py

Created on Tue Aug 26 14:17:09 2025

@author: Carlos Gonzalez
Material Data Division,
National Institute of Standatds and Technology,
Gaithersburg, MD. 20889


====================

Post-hoc analysis of a blinding sweep produced by blinding_driver.py.

Computes, per (model, property, blinded-condition) relative to the
paired native arm:

  * Retention ratio    R = median_c[MAE_native] / median_c[MAE_blinded]
  * Paired per-cycle   dMAE_c = MAE_blinded_c - MAE_native_c
  * Median dMAE, IQR of dMAE
  * Paired bootstrap 95% CI on median dMAE
  * Wilcoxon signed-rank test (paired) and a paired permutation
    (sign-flip) test on dMAE
  * Benjamini-Hochberg FDR adjustment across the principal contrasts
  * An interpretation label: no-significant-change, or, for significant
    contrasts, improved / retained / degraded according to whether the
    retention ratio lies above, inside, or below the tolerance band
    [tol, 1/tol] (default tol = 0.9)

It also emits the compact main-text table and a four-panel figure
(Native vs Generic vs Agnostic) per property, with bold axis labels
(project plotting style).

Pairing
-------

The driver runs every condition with the SAME seed_base, so cycle k of
the native arm and cycle k of a blinded arm share the same split,
demonstration identities and ordering, and query order.  We therefore
pair by cycle index (seed).  If seeds do not align (e.g. runs made with
different seed_base), paired tests are invalid: the contrast is flagged
paired=False, a warning is printed, retention and medians are still
reported, and the paired statistics (CI on dMAE, Wilcoxon, permutation,
q-values) are left as NaN.

Dependencies: numpy, pandas, matplotlib.  SciPy is optional; if absent,
Wilcoxon falls back to a normal-approximation implemented here, and the
permutation test is always available.

No em dashes are used in output strings (project style).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import blinding as B


def _is_missing(v):
    try:
        return v is None or (isinstance(v, float) and np.isnan(v)) \
            or (isinstance(v, str) and v.strip() == "")
    except TypeError:
        return False


# ===========================================================================
#                       Loading
# ===========================================================================


def load_blinding_run(run_dir):
    """
    Return a dict keyed by (model_slug, property, condition) -> npz dict,
    plus the summary DataFrame.
    """
    run_dir = Path(run_dir)
    summary_csv = run_dir / "summary_blinded.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"No summary_blinded.csv in {run_dir}")
    summary_df = pd.read_csv(summary_csv)

    cells = {}
    for _, row in summary_df.iterrows():
        slug = row["model_slug"]
        prop = row["property"]
        cond = row.get("condition", "")
        if _is_missing(cond) or cond == "ALL":
            continue
        if (slug, prop, cond) in cells:
            continue   # repeated summary row (re-run / append)
        folder = f"{slug}_{prop}_{cond}"
        npz = run_dir / folder / "cycles_blinded.npz"
        if npz.exists():
            # Strings in the NPZ are stored as unicode arrays, so pickle
            # is never needed; keep it disabled for safety.
            with np.load(npz, allow_pickle=False) as z:
                cells[(slug, prop, cond)] = {k: z[k] for k in z.files}
        else:
            print(f"[warn] missing cycles_blinded.npz for {folder}")
    return summary_df, cells


def _paired_arrays(native_npz, blinded_npz):
    """Return (native_mae, blinded_mae) aligned by seed where possible.

    Both arrays are per-cycle testing MAE.  If seed vectors are present
    and equal, pairing is by position (which already corresponds to the
    shared seed).  Returns (arr_n, arr_b, paired_bool)."""
    mae_n = np.asarray(native_npz["MAE_test"], dtype=float)
    mae_b = np.asarray(blinded_npz["MAE_test"], dtype=float)

    seed_n = native_npz.get("seed_per_cycle")
    seed_b = blinded_npz.get("seed_per_cycle")
    paired = False
    if seed_n is not None and seed_b is not None:
        seed_n = np.asarray(seed_n).ravel()
        seed_b = np.asarray(seed_b).ravel()
        if seed_n.shape == seed_b.shape and np.array_equal(seed_n, seed_b):
            paired = True

    n = min(len(mae_n), len(mae_b))
    return mae_n[:n], mae_b[:n], paired


# ===========================================================================
#                       Statistics
# ===========================================================================


def median_ratio_retention(mae_native, mae_blinded):
    """R = median(MAE_native) / median(MAE_blinded).  R ~ 1 means
    performance is retained under blinding."""
    mn = np.nanmedian(mae_native)
    mb = np.nanmedian(mae_blinded)
    if mb == 0 or np.isnan(mb):
        return float("nan")
    return float(mn / mb)


def paired_delta(mae_native, mae_blinded):
    """dMAE_c = blinded - native, dropping cycles where either is NaN."""
    m = ~np.isnan(mae_native) & ~np.isnan(mae_blinded)
    return mae_blinded[m] - mae_native[m]


def bootstrap_median_ci(x, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the median of a 1-D sample."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = np.median(x[idx])
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return (float(np.median(x)), float(lo), float(hi))


def wilcoxon_signed_rank(d):
    """
    Paired Wilcoxon signed-rank test on differences d (already paired).
    Uses SciPy if available; otherwise a normal approximation with tie
    and zero handling (Pratt-style zeros dropped).

    Returns (statistic_W, p_two_sided, method_str).
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return (float("nan"), float("nan"), "empty")

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    if wilcoxon is not None:
        # SciPy renamed `mode` to `method` (1.9+); try the new name first
        # so a TypeError does not silently drop us to the approximation.
        for kw in ({"method": "auto"}, {"mode": "auto"}, {}):
            try:
                W, p = wilcoxon(d, zero_method="wilcox",
                                alternative="two-sided", **kw)
                return (float(W), float(p), "scipy.wilcoxon")
            except TypeError:
                continue
            except ValueError:
                break

    # Normal approximation
    ranks = _rankdata(np.abs(d))
    signs = np.sign(d)
    W_plus = np.sum(ranks[signs > 0])
    W_minus = np.sum(ranks[signs < 0])
    W = min(W_plus, W_minus)
    mean_W = n * (n + 1) / 4.0
    # tie correction
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = np.sum(counts ** 3 - counts)
    var_W = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_W <= 0:
        return (float(W), float("nan"), "normal-approx(degenerate)")
    z = (W - mean_W) / np.sqrt(var_W)
    p = 2.0 * _norm_sf(abs(z))
    return (float(W), float(min(1.0, p)), "normal-approx")


def permutation_sign_flip(d, n_perm=20000, seed=0):
    """
    Paired permutation test: under H0 the sign of each paired difference
    is exchangeable.  Test statistic is the mean difference.  Returns a
    two-sided p-value.
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return float("nan")
    obs = np.abs(np.mean(d))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice((-1.0, 1.0), size=n)
        stat = abs(np.mean(signs * d))
        if stat >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def benjamini_hochberg(pvals):
    """Return BH-FDR adjusted q-values for a list of p-values.

    NaN p-values (e.g. unpaired or empty contrasts) are excluded from the
    family and get NaN q-values, so they neither receive q = 1 nor
    inflate the adjustment of the real tests."""
    p_all = np.asarray(pvals, dtype=float)
    out_all = np.full(len(p_all), np.nan)
    finite = np.isfinite(p_all)
    if not finite.any():
        return out_all
    out_all[finite] = _bh_finite(p_all[finite])
    return out_all


def _bh_finite(p):
    """BH step-up on a vector of finite p-values."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n)
    out[order] = q
    return np.clip(out, 0.0, 1.0)


# ---- small stats helpers (no SciPy needed) --------------------------

def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def _norm_sf(x):
    """Survival function of the standard normal via erfc."""
    import math
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def interpret(retention, q_value, alpha=0.05, tol=0.9):
    """Label a contrast.

    R = median MAE_native / median MAE_blinded, so R > 1 means the blinded
    arm has LOWER error.  For contrasts with q < alpha:
        R >  1/tol          -> improved
        tol <= R <= 1/tol   -> retained (significant but small change)
        R <  tol            -> degraded
    A q-value >= alpha gives "no-significant-change"; a missing q-value
    (unpaired contrast) gives "not-tested".
    """
    if np.isnan(retention):
        return "undefined"
    if q_value is None or np.isnan(q_value):
        return "not-tested"
    if q_value >= alpha:
        return "no-significant-change"
    if retention > 1.0 / tol:
        return "improved"
    if retention >= tol:
        return "retained"
    return "degraded"


# ===========================================================================
#                       Contrast table
# ===========================================================================


def build_contrast_table(cells, n_boot=10000, n_perm=20000, seed=0,
                         alpha=0.05, tol=0.9):
    """
    For each (model, property) and each blinded condition present, pair
    against the native arm and compute the full statistics block.
    """
    # index available conditions per (model, property)
    keyset = {}
    for (slug, prop, cond) in cells:
        keyset.setdefault((slug, prop), set()).add(cond)

    rows = []
    for (slug, prop), conds in sorted(keyset.items()):
        if B.NATIVE not in conds:
            print(f"[warn] no native arm for {slug} x {prop}; skipping")
            continue
        native = cells[(slug, prop, B.NATIVE)]
        for cond in sorted(conds):
            if cond == B.NATIVE:
                continue
            blinded = cells[(slug, prop, cond)]
            mae_n, mae_b, paired = _paired_arrays(native, blinded)
            d = paired_delta(mae_n, mae_b)

            retention = median_ratio_retention(mae_n, mae_b)
            if paired:
                med_d, lo_d, hi_d = bootstrap_median_ci(d, n_boot,
                                                        seed=seed)
                W, p_w, method = wilcoxon_signed_rank(d)
                p_perm = permutation_sign_flip(d, n_perm, seed=seed)
            else:
                print(f"[warn] {slug} x {prop} [{cond}]: seed vectors do "
                      f"not match the native arm (different seed_base or "
                      f"M_cycles?). Paired tests skipped for this contrast.")
                med_d = lo_d = hi_d = W = p_w = p_perm = float("nan")
                method = "unpaired(skipped)"

            rows.append({
                "model_slug": slug,
                "property": prop,
                "condition": cond,
                "paired": bool(paired),
                "n_cycles_paired": int(len(d)),
                "median_MAE_native": float(np.nanmedian(mae_n)),
                "median_MAE_blinded": float(np.nanmedian(mae_b)),
                "retention_ratio": retention,
                "median_dMAE": med_d,
                "dMAE_CI_lo": lo_d,
                "dMAE_CI_hi": hi_d,
                "wilcoxon_W": W,
                "p_wilcoxon": p_w,
                "wilcoxon_method": method,
                "p_permutation": p_perm,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        # BH-FDR across all principal contrasts, using the permutation
        # p-values as primary (robust, assumption-light).
        df["q_permutation_BH"] = benjamini_hochberg(df["p_permutation"].values)
        df["q_wilcoxon_BH"] = benjamini_hochberg(df["p_wilcoxon"].values)
        df["interpretation"] = [
            interpret(r, q, alpha=alpha, tol=tol) for r, q in
            zip(df["retention_ratio"], df["q_permutation_BH"])
        ]
    return df


# ===========================================================================
#                       Figure
# ===========================================================================


CONDITION_ORDER = [B.NATIVE, B.GENERIC, B.AGNOSTIC]


def plot_condition_panels(summary_df, out_dir, metric="median_MAE_test",
                          properties=None):
    """
    One panel per property; x-axis Native/Generic/Agnostic; one line per
    model.  Uses the summary_blinded.csv values (median testing MAE by
    default).  Bold axis labels per project style.
    """
    df = summary_df.copy()
    df = df[df["condition"].isin(CONDITION_ORDER)]
    # Use only successful rows, and the LAST one if a cell was re-run.
    if "backend" in df.columns:
        df = df[df["backend"].isin(["LOCAL", "REMOTE"])]
    df = df.drop_duplicates(subset=["model_slug", "property", "condition"],
                            keep="last")
    if df.empty:
        print("[analysis] no successful cells to plot")
        return
    props = properties or sorted(df["property"].unique())
    n = len(props)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5, nrow * 4),
                             squeeze=False)

    for i, prop in enumerate(props):
        ax = axes[i // ncol][i % ncol]
        sub = df[df["property"] == prop]
        for slug in sorted(sub["model_slug"].unique()):
            s2 = sub[sub["model_slug"] == slug]
            xs, ys = [], []
            for ci, cond in enumerate(CONDITION_ORDER):
                row = s2[s2["condition"] == cond]
                if not row.empty:
                    xs.append(ci)
                    ys.append(float(row.iloc[0][metric]))
            if xs:
                ax.plot(xs, ys, marker="o", label=slug, linewidth=1.8)
        ax.set_xticks(range(len(CONDITION_ORDER)))
        ax.set_xticklabels([c.capitalize() for c in CONDITION_ORDER],
                           fontweight="bold")
        ax.set_title(prop, fontweight="bold")
        ax.set_xlabel("Condition", fontweight="bold")
        ax.set_ylabel(metric.replace("_", " "), fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    plt.suptitle("Information-access blinding: testing MAE by condition",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = out_dir / "blinding_condition_panels.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] panels -> {path}")


# ===========================================================================
#                       Main
# ===========================================================================


def main():
    ap = argparse.ArgumentParser(
        description="Analyze a blinding sweep (retention, paired dMAE, CIs).")
    ap.add_argument("--run-dir", default=None,
                    help="Path to RESULTS/blinding_runs/{run_id}")
    ap.add_argument("--results-root", default=None,
                    help="Results root; use with --run-id instead of "
                         "--run-dir")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--analysis-root", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="FDR level for the interpretation labels")
    ap.add_argument("--retention-tol", type=float, default=0.9,
                    help="Retention band [tol, 1/tol] counted as "
                         "'retained' (default 0.9)")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.results_root and args.run_id:
        run_dir = (Path(args.results_root) / "blinding_runs"
                   / args.run_id).resolve()
    else:
        ap.error("Give either --run-dir, or both --results-root and "
                 "--run-id.")
    if not run_dir.is_dir():
        ap.error(f"Run directory not found: {run_dir}")
    run_id = run_dir.name
    if args.analysis_root:
        out_dir = Path(args.analysis_root) / run_id
    else:
        out_dir = run_dir.parent.parent / "blinding_analysis" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analysis] loading blinding run from {run_dir}")
    summary_df, cells = load_blinding_run(run_dir)
    print(f"[analysis] found {len(cells)} condition-cells")

    contrasts = build_contrast_table(
        cells, n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed,
        alpha=args.alpha, tol=args.retention_tol)
    ctab_path = out_dir / "contrast_table.csv"
    contrasts.to_csv(ctab_path, index=False)
    print(f"[analysis] contrast table -> {ctab_path}")

    # Compact main-text table (native vs generic only)
    if not contrasts.empty:
        gen = contrasts[contrasts["condition"] == B.GENERIC].copy()
        keep = ["model_slug", "property", "median_MAE_native",
                "median_MAE_blinded", "median_dMAE", "dMAE_CI_lo",
                "dMAE_CI_hi", "retention_ratio", "q_permutation_BH",
                "interpretation"]
        gen = gen[[c for c in keep if c in gen.columns]]
        gen_path = out_dir / "main_table_native_vs_generic.csv"
        gen.to_csv(gen_path, index=False)
        print(f"[analysis] main table -> {gen_path}")

    # Figure
    plot_condition_panels(summary_df, out_dir)

    # Console rollup
    print("\n" + "=" * 68)
    print("  BLINDING CONTRAST ROLLUP")
    print("=" * 68)
    if contrasts.empty:
        print("  (no contrasts; is there a native arm per property?)")
    else:
        cols = ["model_slug", "property", "condition", "retention_ratio",
                "median_dMAE", "q_permutation_BH", "interpretation"]
        print(contrasts[cols].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
