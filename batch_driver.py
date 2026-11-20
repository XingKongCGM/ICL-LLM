#!/usr/bin/env python3
"""
batch_driver.py

Created on Tue Aug 26 14:17:09 2025

@author: Carlos Gonzalez
Material Data Division,
National Institute of Standatds and Technology,
Gaithersburg, MD. 20889


================

Systematic sweep over (model, property) cells for the ICL evaluation
framework.  Given a properties config and a list of model choices,
runs one of three phases per cell and writes per-cell outputs into a
run-scoped folder under RESULTS_ROOT.

LOCAL-ONLY build. The remote NIST-RCHAT backend has been removed; all
models (choices 1-7) are local MLX models and are run sequentially.

optimal_samples location
------------------------

The best-cycle demonstration set for each (model, property) cell is
written during Testing and read back during Prediction / Forecast.
Its location is controlled by the per-property `optimal_dir` field in
properties.yaml:

    {optimal_dir}/{model_slug}_optimal_samples.csv

Because the model slug is in the filename, different models predicting
the same property do NOT overwrite one another.  Because optimal_dir
lives in the data tree (not under runs/{run_id}/), a Prediction or
Forecast pass REUSES whatever optimal samples a prior Testing run
wrote there -- it need not share the same --run-id.  This is intended:
a Testing sweep can be run once, then reused by any number of later
Prediction/Forecast passes.

If a property omits `optimal_dir`, the code falls back to the legacy
per-cell location out_dir/optimal_samples.csv (tied to the run_id).

Phases
------

  * TESTING (default): for each cell, run the Mc-cycle testing +
    validation pipeline (run_cell). Writes summary.csv and, per cell,
    cycles.npz + the optimal_samples CSV + summaries.

  * PREDICTION (--predict): for each cell, LOAD the optimal_samples CSV
    (from optimal_dir, or the legacy run folder) and apply those fixed
    demonstrations once to the property's query pool (run_cell_predict).
    Writes predict_summary.csv and, per cell, predictions.csv +
    predict_summary.txt.

  * FORECAST (--forecast): pure prediction with metrological uncertainty.
    Reads ONLY the SMILES column of the query pool and predicts N_exp
    replicates per molecule over a temperature sweep, computing per
    molecule mean/median/std/min/max, uA = s/sqrt(N), and
    U95 = t_{0.975,N-1}*uA (run_cell_forecast). Writes forecast_summary.csv
    and, per cell, forecast_<property>.csv + forecast_summary.txt.
    Reuses the optimal_samples CSV as demonstrations.

Execution model
---------------

All models (choices 1-7) are local MLX models and are memory-bound on
the M5 Max, so we run them sequentially: one property at a time, one
model at a time, all sharing the same MLX weights before moving on to
the next model. The RCHAT module keeps backend state as module-level
globals and MLX's C extension is not thread-safe, so models are never
threaded.

Output layout
-------------

    RESULTS_ROOT/runs/{run_id}/
        manifest.json                 # Testing metadata
        manifest_predict.json         # Prediction metadata (if used)
        manifest_forecast.json        # Forecast metadata (if used)
        summary.csv                   # Testing: one row per cell
        predict_summary.csv           # Prediction: one row per cell
        forecast_summary.csv          # Forecast: one row per cell
        {model_slug}_{property}/
            cycles.npz                # per-cycle metric arrays (Testing)
            testing_summary.txt
            validation_summary.txt
            predictions.csv           # predicted,actual (Predict)
            predict_summary.txt
            forecast_{property}.csv   # per-(T,molecule) stats + raws (Forecast)
            forecast_summary.txt

    {optimal_dir}/                    # from properties.yaml, in data tree
        {model_slug}_optimal_samples.csv   # best-cycle demos (Testing)

Usage
-----

    # Testing pass (writes {optimal_dir}/{slug}_optimal_samples.csv)
    python batch_driver.py --config properties.yaml --models 2 7 \\
        --properties LOGKOW MP BP --M-cycles 50 \\
        --results-root /path/to/RESULTS --run-id myrun \\
        [--permutation-control] [--dry-run] [--notify]

    # Prediction pass (reuses optimal samples from any prior Testing run)
    python batch_driver.py --config properties.yaml --models 2 7 \\
        --properties LOGKOW MP BP \\
        --results-root /path/to/RESULTS --run-id anyrun --predict

    # Forecast pass (SMILES-only, N_exp replicates over a T sweep)
    python batch_driver.py --config properties.yaml --models 7 \\
        --properties PKA \\
        --results-root /path/to/RESULTS --run-id anyrun \\
        --forecast --N-exp 20 --temperatures 0.3 0.7 1.0
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Model slugs for filesystem-safe cell folder names.  These are also
# used as dictionary keys in summary.csv.  Order matches the LOCAL_MODEL_MAP
# in RCHAT_PChem_Properties.py.  All models are local MLX.
MODEL_SLUG = {
    1: "llama3.1-8b-q4",
    2: "llama3.3-70b-q4",
    3: "nemotron-mini-4b-bf16",
    4: "llama3.1-nemotron-70b-q4",
    5: "gemma-4-31b-optiq-q4",
    6: "mistral-small-3.1-24b-q4",
    7: "llama4-scout-17b-16e-q4",
}

# summary.csv columns in the order they appear (TESTING phase)
SUMMARY_COLUMNS = [
    "model_slug", "model_name", "property", "backend", "M_c", "N_shots",
    "permutation",
    "median_MAE_test", "IQR_MAE_test", "mean_MAE_test", "std_MAE_test",
    "best_MAE_test", "best_cycle",
    "MAE_val", "RMSE_val", "R2_val", "Acc_val", "F1_val",
    "NRMSE_range_val",
    "n_rejects_test_total", "n_rejects_val",
    "wall_time_h",
]

# predict_summary.csv columns (PREDICTION phase)
PREDICT_SUMMARY_COLUMNS = [
    "model_slug", "model_name", "property", "backend", "phase",
    "n_demos", "n_queries",
    "MAE_predict", "RMSE_predict", "R2_predict", "Acc_predict",
    "F1_predict", "NRMSE_range_predict",
    "n_rejects_predict", "wall_time_h",
]

# forecast_summary.csv columns (FORECAST phase)
FORECAST_SUMMARY_COLUMNS = [
    "model_slug", "model_name", "property", "backend", "phase",
    "n_molecules", "N_exp", "n_temperatures", "temperatures",
    "n_demos", "n_reject_total", "forecast_csv", "wall_time_h",
]


# ---------------------------------------------------------------------------
# optimal_samples path resolution (single source of truth)
# ---------------------------------------------------------------------------

def _resolve_optimal_samples(prop_cfg, model_slug, out_dir):
    """
    Resolve the path to a cell's optimal_samples CSV.

    If the property config provides `optimal_dir` (from properties.yaml),
    the file is {optimal_dir}/{model_slug}_optimal_samples.csv -- per
    model, so different models predicting the same property do not
    overwrite one another.  This function only BUILDS the path; it does
    not touch the filesystem.  run_cell() creates the directory when it
    writes the file, and Prediction / Forecast only read it, so a bad or
    placeholder optimal_dir can no longer crash the sweep before the
    per-cell input checks run.

    If `optimal_dir` is absent, falls back to the legacy per-cell
    location out_dir/optimal_samples.csv so older configs/runs keep
    working unchanged.
    """
    optimal_dir = prop_cfg.get("optimal_dir")
    if optimal_dir:
        return os.path.join(optimal_dir,
                            f"{model_slug}_optimal_samples.csv")
    return os.path.join(str(out_dir), "optimal_samples.csv")


# ---------------------------------------------------------------------------
# YAML loading with ${VAR} substitution (avoids the pyyaml dependency
# by parsing the tiny subset of YAML the config uses).  If pyyaml IS
# available we use it, otherwise fall back to the built-in parser.
# ---------------------------------------------------------------------------

def _substitute_vars(value, env):
    """Replace ${NAME} references in string values using env dict."""
    if not isinstance(value, str):
        return value

    def repl(m):
        key = m.group(1)
        if key not in env:
            raise KeyError(f"Config references ${{{key}}} but it is "
                           f"not defined in the YAML.")
        return env[key]
    return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", repl, value)


def _load_yaml_config(path):
    """Load the properties YAML.  Uses pyyaml if available, else a
    fallback parser that handles only the config's simple structure:
    top-level scalar keys + one nested `properties:` mapping."""
    try:
        import yaml
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
    except ImportError:
        cfg = _fallback_yaml_load(path)

    # Two-pass substitution: first pass expands DATA_ROOT etc.
    env = {k: v for k, v in cfg.items() if isinstance(v, str)}

    def _walk(node):
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(x) for x in node]
        return _substitute_vars(node, env)

    return _walk(cfg)


def _fallback_yaml_load(path):
    """Minimal YAML loader for our specific config shape.  Handles
    top-level scalars, one 'properties:' block, and per-property
    key: value pairs.  Does NOT handle full YAML; use only for this
    config."""
    cfg = {}
    props = {}
    current = None
    with open(path) as fh:
        for line in fh:
            raw = line.rstrip("\n")
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            # Count leading spaces
            indent = len(raw) - len(raw.lstrip(" "))
            content = raw.strip()

            if indent == 0:
                if content.endswith(":"):
                    section = content[:-1]
                    if section == "properties":
                        current = "PROPS"
                    else:
                        current = section
                        cfg[section] = {}
                else:
                    # top-level scalar key: value
                    key, _, val = content.partition(":")
                    cfg[key.strip()] = val.strip()
            elif indent == 2 and current == "PROPS":
                if content.endswith(":"):
                    prop_key = content[:-1]
                    props[prop_key] = {}
            elif indent >= 4 and current == "PROPS":
                key, _, val = content.partition(":")
                # Strip quotes
                val = val.strip().strip('"').strip("'")
                # Find which property we're inside; the last one seen
                last_prop = list(props.keys())[-1]
                props[last_prop][key.strip()] = val

    cfg["properties"] = props
    return cfg


# ---------------------------------------------------------------------------
# Notification helpers
# ---------------------------------------------------------------------------

def _macos_notify(title, message):
    """Fire a macOS notification via osascript.  Safe no-op on non-Mac."""
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}"'],
            check=False, timeout=5,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cell execution (TESTING)
# ---------------------------------------------------------------------------

def _run_one_cell(rchat_mod, model_choice, model_name, property_key,
                   prop_cfg, run_dir, M_c, N_shots, permutation,
                   summary_lock, summary_csv, seed_base=0, temperature=None):
    """
    Run one (model, property) cell.  Calls rchat_mod.run_cell() and
    appends the returned rollup to summary.csv.  Any exception is
    caught, logged, and reported to summary.csv as a failed row.
    """
    slug = MODEL_SLUG[model_choice]
    if permutation:
        cell_folder = f"{slug}_{property_key}_perm"
    else:
        cell_folder = f"{slug}_{property_key}"
    out_dir = Path(run_dir) / cell_folder

    banner = f"CELL: {slug} x {property_key}"
    print(f"\n{'#' * 72}")
    print(f"#  {banner:<66}  #")
    print(f"#  Permutation control: {str(permutation):<44}  #")
    print(f"{'#' * 72}")

    data_paths = {
        "training":        prop_cfg["training"],
        "validation":      prop_cfg["validation"],
        "var":             prop_cfg["var"],
        "title":           prop_cfg.get("title", property_key),
        "units":           prop_cfg.get("units", "-"),
        "optimal_samples": _resolve_optimal_samples(prop_cfg, slug, out_dir),
    }

    # Sanity-check that the input CSVs exist before starting
    for kind, path in [("training", data_paths["training"]),
                       ("validation", data_paths["validation"])]:
        if not os.path.exists(path):
            msg = (f"MISSING {kind} CSV for {property_key}: {path}. "
                   f"Skipping cell.")
            print(f"  [ERROR] {msg}")
            _append_summary_row(summary_csv, summary_lock, {
                "model_slug": slug, "model_name": model_name,
                "property": property_key, "backend": "SKIPPED",
                "M_c": M_c, "N_shots": N_shots,
                "permutation": bool(permutation),
            })
            return

    try:
        rollup = rchat_mod.run_cell(
            model_name=model_name,
            model_slug=slug,
            property_key=property_key,
            data_paths=data_paths,
            out_dir=str(out_dir),
            M_c=M_c,
            N_shots=N_shots,
            test_split=0.20,
            permutation=permutation,
            seed_base=seed_base,
            temperature=temperature,
        )
        _append_summary_row(summary_csv, summary_lock, rollup)
    except Exception as e:
        print(f"  [CELL FAILED] {type(e).__name__}: {e}")
        traceback.print_exc()
        _append_summary_row(summary_csv, summary_lock, {
            "model_slug": slug, "model_name": model_name,
            "property": property_key, "backend": "FAILED",
            "M_c": M_c, "N_shots": N_shots,
            "permutation": bool(permutation),
        })


def _append_summary_row(summary_csv, lock, row):
    """Thread-safe append to summary.csv.  Missing columns → empty."""
    complete_row = {col: row.get(col, "") for col in SUMMARY_COLUMNS}
    with lock:
        exists = os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(complete_row)


def _run_model_sweep(rchat_mod, model_choice, properties_to_run,
                     properties_config, run_dir, M_c, N_shots,
                     permutation, summary_lock, summary_csv,
                     temperature=None):
    """
    Load one model (via select_llm_model_by_choice) and iterate all
    properties for it.  Called once per model.  Serial loop.
    """
    slug = MODEL_SLUG[model_choice]
    print(f"\n{'*' * 72}")
    print(f"*  MODEL SWEEP  {slug}   ({len(properties_to_run)} properties)")
    print(f"{'*' * 72}\n")

    try:
        model_name = rchat_mod.select_llm_model_by_choice(model_choice)
    except Exception as e:
        print(f"  [MODEL LOAD FAILED] {slug}: "
              f"{type(e).__name__}: {e}")
        return

    for property_key in properties_to_run:
        prop_cfg = properties_config[property_key]
        _run_one_cell(
            rchat_mod, model_choice, model_name, property_key, prop_cfg,
            run_dir, M_c, N_shots, permutation=False,
            summary_lock=summary_lock, summary_csv=summary_csv,
            temperature=temperature,
        )
        if permutation:
            _run_one_cell(
                rchat_mod, model_choice, model_name, property_key,
                prop_cfg, run_dir, M_c, N_shots, permutation=True,
                summary_lock=summary_lock, summary_csv=summary_csv,
                temperature=temperature,
            )


# ---------------------------------------------------------------------------
# Cell execution (PREDICTION)
# ---------------------------------------------------------------------------

def _append_predict_row(predict_csv, lock, row):
    """Thread-safe append to predict_summary.csv.  Missing cols → empty."""
    complete_row = {col: row.get(col, "") for col in PREDICT_SUMMARY_COLUMNS}
    with lock:
        exists = os.path.exists(predict_csv)
        with open(predict_csv, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=PREDICT_SUMMARY_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(complete_row)


def _run_one_predict_cell(rchat_mod, model_choice, model_name, property_key,
                          prop_cfg, run_dir, summary_lock, predict_csv,
                          seed_base=0, temperature=None):
    """
    Run one (model, property) Prediction cell.  Loads the optimal
    demonstrations (from optimal_dir or the legacy run folder), then
    predicts on the property's query pool.  A missing optimal_samples
    CSV is reported as SKIPPED rather than crashing the sweep.
    """
    slug = MODEL_SLUG[model_choice]
    cell_folder = f"{slug}_{property_key}"          # same folder as Testing
    out_dir = Path(run_dir) / cell_folder

    banner = f"PREDICT CELL: {slug} x {property_key}"
    print(f"\n{'#' * 72}")
    print(f"#  {banner:<66}  #")
    print(f"{'#' * 72}")

    data_paths = {
        "var":             prop_cfg["var"],
        "validation":      prop_cfg.get("validation"),
        "predict":         prop_cfg.get("predict", None),   # optional pool
        "feature_col":     prop_cfg.get("feature_col", "Molecule"),
        "optimal_samples": _resolve_optimal_samples(prop_cfg, slug, out_dir),
    }

    try:
        rollup = rchat_mod.run_cell_predict(
            model_name=model_name,
            model_slug=slug,
            property_key=property_key,
            data_paths=data_paths,
            out_dir=str(out_dir),
            seed_base=seed_base,
            temperature=temperature,
        )
        _append_predict_row(predict_csv, summary_lock, rollup)
    except FileNotFoundError as e:
        # Most common cause: no optimal_samples CSV because Testing was
        # not run for this cell.  Report SKIPPED, keep going.
        print(f"  [PREDICT SKIPPED] {type(e).__name__}: {e}")
        _append_predict_row(predict_csv, summary_lock, {
            "model_slug": slug, "model_name": model_name,
            "property": property_key, "backend": "SKIPPED",
            "phase": "PREDICT",
        })
    except Exception as e:
        print(f"  [PREDICT FAILED] {type(e).__name__}: {e}")
        traceback.print_exc()
        _append_predict_row(predict_csv, summary_lock, {
            "model_slug": slug, "model_name": model_name,
            "property": property_key, "backend": "FAILED",
            "phase": "PREDICT",
        })


def _run_model_predict_sweep(rchat_mod, model_choice, properties_to_run,
                             properties_config, run_dir, summary_lock,
                             predict_csv, temperature=None):
    """Load one model and run Prediction for each property."""
    slug = MODEL_SLUG[model_choice]
    print(f"\n{'*' * 72}")
    print(f"*  PREDICT SWEEP  {slug}   ({len(properties_to_run)} properties)")
    print(f"{'*' * 72}\n")

    try:
        model_name = rchat_mod.select_llm_model_by_choice(model_choice)
    except Exception as e:
        print(f"  [MODEL LOAD FAILED] {slug}: {type(e).__name__}: {e}")
        return

    for property_key in properties_to_run:
        _run_one_predict_cell(
            rchat_mod, model_choice, model_name, property_key,
            properties_config[property_key], run_dir,
            summary_lock, predict_csv, temperature=temperature,
        )


# ---------------------------------------------------------------------------
# Cell execution (FORECAST)
# ---------------------------------------------------------------------------

def _append_forecast_row(forecast_csv, lock, row):
    """Thread-safe append to forecast_summary.csv.  Missing cols → empty."""
    complete_row = {col: row.get(col, "") for col in FORECAST_SUMMARY_COLUMNS}
    with lock:
        exists = os.path.exists(forecast_csv)
        with open(forecast_csv, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FORECAST_SUMMARY_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(complete_row)


def _run_one_forecast_cell(rchat_mod, model_choice, model_name, property_key,
                           prop_cfg, run_dir, N_exp, temperatures,
                           summary_lock, forecast_csv, seed_base=0):
    """
    Run one (model, property) Forecast cell.  Loads the optimal_samples
    CSV (from optimal_dir or the legacy run folder) as demonstrations,
    reads ONLY the SMILES column of the query pool, and predicts N_exp
    replicates per molecule over a temperature sweep, computing
    per-molecule mean/std/uA/U95.  A missing optimal_samples CSV is
    reported SKIPPED, not fatal.
    """
    slug = MODEL_SLUG[model_choice]
    cell_folder = f"{slug}_{property_key}"          # same folder as Testing
    out_dir = Path(run_dir) / cell_folder

    banner = f"FORECAST CELL: {slug} x {property_key}"
    print(f"\n{'#' * 72}")
    print(f"#  {banner:<66}  #")
    print(f"{'#' * 72}")

    data_paths = {
        "var":             prop_cfg["var"],
        # Forecast query pool: prefer a dedicated `predict:` set (SMILES
        # only is fine), else fall back to the validation CSV.
        "predict":         prop_cfg.get("predict", prop_cfg.get("validation")),
        "feature_col":     prop_cfg.get("feature_col", "Molecule"),
        "optimal_samples": _resolve_optimal_samples(prop_cfg, slug, out_dir),
    }

    try:
        rollup = rchat_mod.run_cell_forecast(
            model_name=model_name,
            model_slug=slug,
            property_key=property_key,
            data_paths=data_paths,
            out_dir=str(out_dir),
            N_exp=N_exp,
            temperatures=temperatures,
            seed_base=seed_base,
        )
        _append_forecast_row(forecast_csv, summary_lock, rollup)
    except FileNotFoundError as e:
        print(f"  [FORECAST SKIPPED] {type(e).__name__}: {e}")
        _append_forecast_row(forecast_csv, summary_lock, {
            "model_slug": slug, "model_name": model_name,
            "property": property_key, "backend": "SKIPPED",
            "phase": "FORECAST",
        })
    except Exception as e:
        print(f"  [FORECAST FAILED] {type(e).__name__}: {e}")
        traceback.print_exc()
        _append_forecast_row(forecast_csv, summary_lock, {
            "model_slug": slug, "model_name": model_name,
            "property": property_key, "backend": "FAILED",
            "phase": "FORECAST",
        })


def _run_model_forecast_sweep(rchat_mod, model_choice, properties_to_run,
                              properties_config, run_dir, N_exp,
                              temperatures, summary_lock, forecast_csv):
    """Load one model and run Forecast for each property."""
    slug = MODEL_SLUG[model_choice]
    print(f"\n{'*' * 72}")
    print(f"*  FORECAST SWEEP  {slug}   ({len(properties_to_run)} properties)")
    print(f"{'*' * 72}\n")

    try:
        model_name = rchat_mod.select_llm_model_by_choice(model_choice)
    except Exception as e:
        print(f"  [MODEL LOAD FAILED] {slug}: {type(e).__name__}: {e}")
        return

    for property_key in properties_to_run:
        _run_one_forecast_cell(
            rchat_mod, model_choice, model_name, property_key,
            properties_config[property_key], run_dir, N_exp, temperatures,
            summary_lock, forecast_csv,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch driver for the ICL (model, property) sweep."
    )
    parser.add_argument("--config", required=True,
                        help="Path to properties.yaml")
    parser.add_argument("--models", nargs="+", type=int, required=True,
                        help="Model choices (integers 1-7) to include")
    parser.add_argument("--properties", nargs="+", default=None,
                        help="Property keys to run (default: all in config)")
    parser.add_argument("--M-cycles", type=int, default=50,
                        help="Number of Testing cycles per cell (default 50)")
    parser.add_argument("--N-shots", type=int, default=32,
                        help="Number of in-context demos per query (default 32)")
    parser.add_argument("--results-root", required=True,
                        help="Root directory for all outputs")
    parser.add_argument("--run-id", default=None,
                        help="Custom run id; default is timestamp. "
                             "Testing writes per-cell outputs here. "
                             "Prediction/Forecast reuse optimal_samples "
                             "from the config's optimal_dir, so they do NOT "
                             "require the same --run-id as the Testing run "
                             "(unless a property omits optimal_dir, in which "
                             "case the legacy per-run folder is used).")
    parser.add_argument("--permutation-control", action="store_true",
                        help="For each cell, also run a permutation-control "
                             "cell with shuffled training labels")
    parser.add_argument("--predict", action="store_true",
                        help="Run batch-mode Prediction: load each cell's "
                             "optimal_samples CSV and predict once on the "
                             "property's query pool. Writes predict_summary.csv "
                             "and per-cell predictions.csv.")
    parser.add_argument("--forecast", action="store_true",
                        help="Run Forecast mode: read ONLY the SMILES column "
                             "of each property's query pool and predict N_exp "
                             "replicates per molecule over a temperature "
                             "sweep, computing mean/std/uA/U95. Reuses the "
                             "optimal_samples CSV from a prior Testing run.")
    parser.add_argument("--N-exp", type=int, default=20,
                        help="Forecast: replicate queries per molecule per "
                             "temperature (default 20).")
    parser.add_argument("--temperatures", nargs="+", type=float,
                        default=[0.7],
                        help="Forecast: temperatures to sweep, e.g. "
                             "--temperatures 0.3 0.7 1.0 (default 0.7).")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Testing/Predict: sampling temperature for "
                             "every query. Omit to use the module default "
                             "(DEFAULT_TEMPERATURE = 0.9), which reproduces "
                             "the published runs. Ignored by --forecast, "
                             "which uses --temperatures instead.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the execution plan and exit")
    parser.add_argument("--notify", action="store_true",
                        help="Send a macOS notification on completion")
    parser.add_argument("--rchat-script-dir", default=None,
                        help="Directory containing RCHAT_PChem_Properties.py "
                             "(default: same directory as this driver)")
    args = parser.parse_args()

    # Mutually exclusive phase guard.
    if args.predict and args.forecast:
        parser.error("Use only one of --predict or --forecast, not both.")

    # --- Import RCHAT_PChem_Properties from the specified location ----
    script_dir = args.rchat_script_dir or os.path.dirname(
        os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    print(f"[driver] Importing RCHAT_PChem_Properties from {script_dir}")
    import RCHAT_PChem_Properties as rchat_mod

    # --- Load properties config ---------------------------------------
    cfg = _load_yaml_config(args.config)
    all_props = cfg["properties"]
    properties_to_run = args.properties or sorted(all_props.keys())
    for p in properties_to_run:
        if p not in all_props:
            raise KeyError(f"Property {p!r} not in config. "
                           f"Available: {sorted(all_props.keys())}")

    # --- Check the data files BEFORE loading any model ---------------
    # Loading a large model takes time and memory, so fail fast (or at
    # least warn) if the CSVs the requested phase needs are not there.
    # The commonest cause is DATA_ROOT in properties.yaml still holding
    # its placeholder.
    if args.forecast:
        needed = [("predict", "validation")]          # predict, else validation
    elif args.predict:
        needed = [("predict", "validation")]
    else:
        needed = [("training",), ("validation",)]
    missing = {}
    for p in properties_to_run:
        pc = all_props[p]
        for alternatives in needed:
            paths = [pc.get(k) for k in alternatives if pc.get(k)]
            if not any(os.path.exists(x) for x in paths):
                missing.setdefault(p, []).append(
                    f"{'/'.join(alternatives)}: {paths[0] if paths else '(not set)'}")
    if missing:
        data_root = cfg.get("DATA_ROOT", "(not set)")
        print(f"[driver] Data files not found for {len(missing)} of "
              f"{len(properties_to_run)} requested properties:")
        for p, items in missing.items():
            for it in items:
                print(f"           {p:8s} {it}")
        print(f"[driver] DATA_ROOT in {args.config} is: {data_root}")
        if len(missing) == len(properties_to_run):
            print("[driver] Nothing can run. Set DATA_ROOT in the config "
                  "to the folder that holds your datasets (and check the "
                  "file names), then try again.")
            sys.exit(2)
        print("[driver] Those properties will be reported as SKIPPED; "
              "the others will run.")

    # --- Validate model choices ---------------------------------------
    for m in args.models:
        if m not in MODEL_SLUG:
            raise KeyError(f"Model choice {m} not valid. "
                           f"Available: {sorted(MODEL_SLUG.keys())}")

    # --- Determine phase ----------------------------------------------
    if args.forecast:
        phase = "FORECAST"
    elif args.predict:
        phase = "PREDICT"
    else:
        phase = "TESTING"

    # --- Materialize run directory ------------------------------------
    if args.run_id:
        run_id = args.run_id
    else:
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        tag = f"M{args.M_cycles}"
        if args.permutation_control:
            tag += "_perm"
        if args.predict:
            tag += "_predict"
        if args.forecast:
            tag += "_forecast"
        run_id = f"{stamp}_{tag}"

    run_dir = Path(args.results_root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = str(run_dir / "summary.csv")
    predict_csv = str(run_dir / "predict_summary.csv")
    forecast_csv = str(run_dir / "forecast_summary.csv")

    # For predict/forecast, warn early if no optimal_samples exist for
    # the requested models/properties.  Resolve each cell's path exactly
    # as the phase runners will (optimal_dir from config, else the legacy
    # per-cell folder under run_dir).
    if args.predict or args.forecast:
        def _opt_exists(m, p):
            prop_cfg = all_props[p]
            out_dir_cell = run_dir / f"{MODEL_SLUG[m]}_{p}"
            path = _resolve_optimal_samples(prop_cfg, MODEL_SLUG[m],
                                            out_dir_cell)
            return os.path.exists(path)

        any_opt = any(_opt_exists(m, p)
                      for m in args.models for p in properties_to_run)
        if not any_opt:
            print(f"[driver] WARNING: no optimal_samples found for the "
                  f"requested models/properties (checked the config's "
                  f"optimal_dir locations and the legacy per-cell folders "
                  f"under {run_dir}). {phase} needs a prior Testing sweep "
                  f"that wrote them. All cells will be SKIPPED.")

    manifest = {
        "run_id"        : run_id,
        "started_at"    : dt.datetime.now().isoformat(),
        "phase"         : phase,
        "config_path"   : os.path.abspath(args.config),
        "models"        : args.models,
        "model_slugs"   : {str(m): MODEL_SLUG.get(m, "?")
                            for m in args.models},
        "properties"    : properties_to_run,
        "M_cycles"      : args.M_cycles,
        "N_shots"       : args.N_shots,
        "permutation"   : args.permutation_control,
        "predict"       : args.predict,
        "forecast"      : args.forecast,
        "N_exp"         : args.N_exp if args.forecast else None,
        "temperature"   : args.temperature if phase in ("TESTING", "PREDICT") else None,
        "temperatures"  : args.temperatures if args.forecast else None,
        "results_root"  : os.path.abspath(args.results_root),
        "run_dir"       : str(run_dir),
    }
    # Phase-specific manifest so a later pass does not clobber the
    # Testing manifest on the same run_id.
    manifest_name = {
        "TESTING":  "manifest.json",
        "PREDICT":  "manifest_predict.json",
        "FORECAST": "manifest_forecast.json",
    }[phase]
    with open(run_dir / manifest_name, "w") as fh:
        json.dump(manifest, fh, indent=2)

    # --- Plan announcement --------------------------------------------
    if args.forecast:
        n_cells_each_model = len(properties_to_run)
    elif args.predict:
        n_cells_each_model = len(properties_to_run)
    else:
        n_cells_each_model = len(properties_to_run) * (
            2 if args.permutation_control else 1)
    n_cells_total = n_cells_each_model * len(args.models)

    print("=" * 72)
    print(f"  ICL SWEEP RUN  {run_id}   [{phase}]")
    print("=" * 72)
    print(f"  Run directory : {run_dir}")
    if args.forecast:
        print(f"  Forecast CSV  : {forecast_csv}")
    elif args.predict:
        print(f"  Predict CSV   : {predict_csv}")
    else:
        print(f"  Summary CSV   : {summary_csv}")
    print(f"  Models        : {args.models}  ({[MODEL_SLUG[m] for m in args.models]})")
    print(f"  Properties    : {properties_to_run}")
    if phase == "TESTING":
        print(f"  M_cycles      : {args.M_cycles}")
        print(f"  N_shots       : {args.N_shots}")
        print(f"  Permutation   : {args.permutation_control}")
        print(f"  Temperature   : {args.temperature if args.temperature is not None else 'default (0.9)'}")
    if args.forecast:
        print(f"  N_exp         : {args.N_exp}")
        print(f"  Temperatures  : {args.temperatures}")
    print(f"  Cells/model   : {n_cells_each_model}")
    print(f"  Total cells   : {n_cells_total}")
    print("=" * 72)

    if args.dry_run:
        print("\n[dry-run] Not executing.  Manifest written; exiting.")
        return

    # --- Execute ------------------------------------------------------
    # SERIAL EXECUTION: models are processed one at a time in the order
    # given by --models. The RCHAT module keeps backend state as module-
    # level globals that select_llm_model_by_choice() overwrites, and
    # MLX's C extension is not thread-safe, so we never thread across
    # models.
    summary_lock = threading.Lock()

    t_start = time.time()

    if args.forecast:
        # FORECAST PHASE. Requires optimal_samples from a prior Testing
        # sweep (reused via the config's optimal_dir).
        for m in args.models:
            _run_model_forecast_sweep(
                rchat_mod, m, properties_to_run, all_props,
                run_dir, args.N_exp, args.temperatures,
                summary_lock, forecast_csv,
            )
    elif args.predict:
        # PREDICTION PHASE.
        for m in args.models:
            _run_model_predict_sweep(
                rchat_mod, m, properties_to_run, all_props,
                run_dir, summary_lock, predict_csv,
                temperature=args.temperature,
            )
    else:
        # TESTING (+ optional permutation) PHASE.
        for m in args.models:
            _run_model_sweep(
                rchat_mod, m, properties_to_run, all_props,
                run_dir, args.M_cycles, args.N_shots,
                args.permutation_control, summary_lock, summary_csv,
                temperature=args.temperature,
            )

    t_end = time.time()
    wall_h = (t_end - t_start) / 3600.0

    # --- Completion ---------------------------------------------------
    # Count successes/failures from the phase-appropriate CSV.
    if args.forecast:
        active_csv = forecast_csv
    elif args.predict:
        active_csv = predict_csv
    else:
        active_csv = summary_csv

    n_success = n_failed = 0
    if os.path.exists(active_csv):
        with open(active_csv) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("backend") == "LOCAL":
                    n_success += 1
                elif row.get("backend") in ("FAILED", "SKIPPED"):
                    n_failed += 1

    stats = {
        "phase"         : phase,
        "n_cells"       : n_cells_total,
        "n_success"     : n_success,
        "n_failed"      : n_failed,
        "wall_clock_h"  : wall_h,
        "summary_csv"   : active_csv,
    }
    # Phase-specific completion marker so one pass does not overwrite
    # another's RUN_COMPLETE on the same run_id.
    marker_name = {
        "TESTING":  "RUN_COMPLETE.txt",
        "PREDICT":  "RUN_COMPLETE_predict.txt",
        "FORECAST": "RUN_COMPLETE_forecast.txt",
    }[phase]
    marker = Path(run_dir) / marker_name
    with marker.open("w") as fh:
        fh.write(f"Run completed at {dt.datetime.now().isoformat()}\n")
        fh.write(f"Phase              : {phase}\n")
        fh.write(f"Cells run          : {stats['n_cells']}\n")
        fh.write(f"Cells succeeded    : {stats['n_success']}\n")
        fh.write(f"Cells failed       : {stats['n_failed']}\n")
        fh.write(f"Total wall clock   : {stats['wall_clock_h']:.2f} h\n")
        fh.write(f"Summary CSV        : {stats['summary_csv']}\n")

    print()
    print("=" * 72)
    print(f"  SWEEP COMPLETE [{phase}]   wall time {wall_h:.2f} h "
          f"({n_success} ok, {n_failed} failed, {n_cells_total} total)")
    print(f"  Summary          {active_csv}")
    print("=" * 72)

    if args.notify:
        _macos_notify(
            f"ICL {phase.lower()} complete",
            f"{n_success}/{n_cells_total} cells OK, {wall_h:.1f} h",
        )


if __name__ == "__main__":
    main()
