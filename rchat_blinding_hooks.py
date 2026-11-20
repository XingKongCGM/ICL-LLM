#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rchat_blinding_hooks.py

Created on Tue Aug 26 14:17:09 2025

@author: Carlos Gonzalez
Material Data Division,
National Institute of Standatds and Technology,
Gaithersburg, MD. 20889


=======================

Blinded-cell execution hooks for the PFAS LLM-ICL study.

This module adds run_cell_blinded() alongside the existing
RCHAT_PChem_Properties.run_cell().  It reuses every low-level primitive
from RCHAT_PChem_Properties (send_query, extract_numeric_prediction,
read_csv_file, get_template_prompts, get_accuracy, get_f1_stats, the
sklearn metric calls) and every scoring convention, so blinded results
are directly comparable to the native results produced by run_cell().

Why a separate module
---------------------

The core statistical / metric / plotting logic in
RCHAT_PChem_Properties.py stays untouched (its original design goal).
The only NEW behavior lives here:

  * per-cycle label transforms (native -> transformed -> native),
  * optional SMILES recoding,
  * blinded prompt assembly matched to the native prompt structure,
  * paired-cycle bookkeeping so that native and blinded arms of the
    SAME (model, property, cycle) see the SAME split, the SAME
    demonstration identities and ordering, and the SAME query order.

The paired design is what makes the retention-ratio and paired-ΔMAE
statistics valid, so seeding is handled carefully and identically to
run_cell().

Key comparability guarantees vs. run_cell()
-------------------------------------------
  1. Same train/test split: train_test_split(df_train,
     test_size=test_split, random_state=seed_base).
  2. Same per-cycle demonstration subsample: random.Random(seed) with
     seed = seed_base + cycle_index, then rng.sample(...).
  3. Same median (combined train+val) for binary accuracy.
  4. Same NRMSE conventions (range-normalized, native units).
  5. Same numeric parser and rejection accounting.

The ONLY intended difference is the information content of the prompt.

Formatting note
---------------
Demonstration labels are written with spec.value_fmt (default '{:.3f}')
in EVERY condition, including NATIVE.  The native arm of a blinding
sweep therefore matches the generic/agnostic arms in numeric formatting,
but it is NOT byte-identical to the main-study run_cell() prompts, which
write the raw CSV string.  Compare blinded arms against the native arm of
the SAME blinding sweep, not against a main-study run.

Backend
-------
Works with the local-only RCHAT_PChem_Properties core (MLX).  If an older
core that still exposes a `_BACKEND` global is on the path, its value is
used for the summary 'backend' column; otherwise 'LOCAL' is reported.

No em dashes are used in output strings (project style).
"""

from __future__ import annotations

import os
import csv
import time
import math
import random
import warnings

import numpy as np
import pandas as pd

# Import the existing pipeline module.  We deliberately depend on it
# rather than duplicating its primitives.
import RCHAT_PChem_Properties as rchat

# Import the self-contained blinding toolkit.
import blinding as B


def _backend_name():
    """'LOCAL' for the local-only core; the old core's _BACKEND if set."""
    return getattr(rchat, "_BACKEND", None) or "LOCAL"


# ===========================================================================
#           Blinded k-shot evaluation (mirrors k_shot_test)
# ===========================================================================


def k_shot_test_blinded(training_samples, test_samples, median,
                         instruction_text, feature_key, output_key,
                         property_var, spec, Rate, model,
                         struct_col="Molecule", temperature=None):
    """
    Blinded analogue of RCHAT_PChem_Properties.k_shot_test.

    Differences from the native k_shot_test:
      * The prompt is assembled from `instruction_text` +
        transformed/recoded demonstrations, using feature_key /
        output_key that carry NO property identity for blinded
        conditions.

    IMPORTANT distinction between two easily-confused identifiers:
      * struct_col  : the CSV COLUMN name to read the SMILES/structure
                      FROM each record (e.g. "Molecule" or "SMILES").
      * feature_key : the PROMPT-SIDE label to print the structure UNDER
                      (e.g. "Molecule" native, "String" generic,
                      "Code string" agnostic).
    These coincide only for the native arm; conflating them raises
    KeyError on demo[feature_key] for blinded conditions.
      * Demonstration labels are written in TRANSFORMED space via
        spec.format_label().
      * Structure strings are recoded when spec.condition == AGNOSTIC.
      * The raw model output is parsed as a number in TRANSFORMED space,
        then inverse-transformed to native units via
        spec.parse_and_invert() BEFORE any metric is computed.
      * Out-of-range transformed responses are tallied separately.

    Returns a dict (richer than the native tuple) so the caller can
    persist both native-unit and transformed-unit diagnostics:

        {
          "MSE","RMSE","MAE","R2","Acc","F1",        # native units
          "MSE_t","RMSE_t","MAE_t","R2_t",           # transformed units
          "n_rejects","n_out_of_range","n_scored",
        }
    """
    Tau = 60.0 / Rate

    predicted_native = []
    actual_native = []
    predicted_trans = []
    actual_trans = []
    n_rejects = 0
    n_out_of_range = 0

    for nrow in range(len(test_samples)):
        # Build the prompt: instruction block, then demonstrations, then
        # the query stem with an assistant-priming "output_key: " tail
        # (mirrors the native "prop: " prefill so _split_priming in the
        # local backend still works).
        prompt = instruction_text
        for krow in range(len(training_samples)):
            demo = training_samples[krow]
            struct = spec.represent_structure(str(demo[struct_col]))
            label = spec.format_label(float(demo[property_var]))
            prompt += f"{output_key_line(feature_key, struct)}\n"
            prompt += f"{output_key}: {label}\n\n"

        q = test_samples[nrow]
        q_struct = spec.represent_structure(str(q[struct_col]))
        prompt += f"{output_key_line(feature_key, q_struct)}\n"
        prompt += f"{output_key}: "

        raw = rchat.send_query(prompt, model, temperature=temperature)
        val_t = rchat.extract_numeric_prediction(raw)

        if val_t is not None:
            native, in_range = spec.parse_and_invert(float(val_t))
            if not in_range:
                n_out_of_range += 1
            y_native = float(q[property_var])
            predicted_native.append(float(native))
            actual_native.append(y_native)
            predicted_trans.append(float(val_t))
            actual_trans.append(spec.transform.forward(y_native))
        else:
            print(f"Skipping sample (blinded, {spec.condition}) due to "
                  f"invalid prediction (raw: {raw})")
            n_rejects += 1

        time.sleep(Tau)

    out = {
        "MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan,
        "Acc": np.nan, "F1": np.nan,
        "MSE_t": np.nan, "RMSE_t": np.nan, "MAE_t": np.nan, "R2_t": np.nan,
        "n_rejects": n_rejects, "n_out_of_range": n_out_of_range,
        "n_scored": len(predicted_native),
    }

    if len(predicted_native) == 0:
        print(f"\n[ERROR] k_shot_test_blinded ({spec.condition}): all "
              f"{len(test_samples)} predictions rejected "
              f"(n_rejects={n_rejects}). Returning NaN metrics.")
        return out

    pn = np.array(predicted_native)
    an = np.array(actual_native)
    pt = np.array(predicted_trans)
    at = np.array(actual_trans)

    # ---- Native-unit metrics (directly comparable to run_cell) -------
    out["MSE"] = rchat.mean_squared_error(an, pn)
    out["RMSE"] = math.sqrt(out["MSE"])
    out["MAE"] = rchat.mean_absolute_error(an, pn)
    if len(pn) > 3:
        out["R2"] = rchat.r2_score(an, pn)
    out["Acc"] = rchat.get_accuracy(pn, an, median)
    _, out["F1"] = rchat.get_f1_stats(pn, an)

    # ---- Transformed-unit metrics (secondary diagnostic only) --------
    # NOTE: R2_t / NRMSE in [0,100] space are NOT comparable across
    # properties or to the native tables; kept for the anchoring
    # diagnostic described in the project notes.
    out["MSE_t"] = rchat.mean_squared_error(at, pt)
    out["RMSE_t"] = math.sqrt(out["MSE_t"])
    out["MAE_t"] = rchat.mean_absolute_error(at, pt)
    if len(pt) > 3:
        out["R2_t"] = rchat.r2_score(at, pt)

    return out


def output_key_line(feature_key, struct_text):
    """Compose the demonstration/query structure line, e.g.
    'String: Xa...' or 'Molecule: OC(=O)...'.  Kept as a helper so the
    exact spacing matches the native k_shot_test format."""
    return f"{feature_key}: {struct_text}"


# ===========================================================================
#           Blinded cell runner (mirrors run_cell)
# ===========================================================================


def run_cell_blinded(model_name, model_slug, property_key, data_paths,
                     out_dir, spec, M_c=50, N_shots=32, test_split=0.20,
                     seed_base=0, temperature=None):
    """
    Blinded analogue of RCHAT_PChem_Properties.run_cell for ONE
    (model, property, condition) cell.

    Parameters mirror run_cell(), plus:

      spec : blinding.BlindingSpec
          Fully constructed condition/transform/recoder bundle.  For
          spec.condition == NATIVE this reproduces the native pipeline
          (Identity transform, no recoding, native instruction text and
          feature/output labels from get_template_prompts()), which lets
          you regenerate a native arm within the same paired sweep.  Label
          values are formatted with spec.value_fmt, so the native arm is
          format-matched to the blinded arms rather than byte-identical to
          the main-study prompts.

      temperature : float or None
          Sampling temperature; None uses the core's DEFAULT_TEMPERATURE
          (0.9), which reproduces the published runs.

    Writes cycles_blinded.npz + summaries into out_dir and returns a
    rollup dict for the run summary CSV.

    Seeding is identical to run_cell() so a NATIVE blinded run and a
    GENERIC blinded run with the same seed_base are cycle-paired.
    """
    os.makedirs(out_dir, exist_ok=True)
    property_var = data_paths["var"]

    # ---- Data loading (identical to run_cell) -----------------------
    df_train = pd.DataFrame(rchat.read_csv_file(data_paths["training"]))
    df_train[property_var] = pd.to_numeric(df_train[property_var],
                                           errors="coerce")
    df_train = df_train.dropna(subset=[property_var])

    df_val = pd.DataFrame(rchat.read_csv_file(data_paths["validation"]))
    df_val[property_var] = pd.to_numeric(df_val[property_var],
                                         errors="coerce")
    df_val = df_val.dropna(subset=[property_var])

    N_total = len(df_train) + len(df_val)
    if N_total < 20:
        raise ValueError(f"Property {property_key}: too few valid records "
                         f"({N_total}); need at least 20.")

    df_full = pd.concat([df_train, df_val], ignore_index=True)
    median = float(df_full[property_var].median())
    mean_all = float(df_full[property_var].mean())
    y_min = float(df_full[property_var].min())
    y_max = float(df_full[property_var].max())

    # ---- Feature key detection --------------------------------------
    # The native CSVs use a structure column; run_cell hard-codes
    # "Molecule" as the feature key passed to k_shot_test.  We honor an
    # optional override in data_paths, else default to "Molecule".
    feature_col = data_paths.get("feature_col", "Molecule")
    if feature_col not in df_full.columns:
        # Fall back to the first non-property column so a mislabeled
        # config fails soft rather than KeyError deep in the loop.
        candidates = [c for c in df_full.columns if c != property_var]
        if not candidates:
            raise KeyError(f"No structure column found for {property_key}.")
        feature_col = candidates[0]
        print(f"[blinded] feature_col not found; using {feature_col!r}")

    # ---- Instruction text + labels per condition --------------------
    if spec.condition == B.NATIVE:
        # Reuse the native template + native feature/output labels so the
        # native arm is faithful to the main study.
        native_instr, _, native_var, _ = rchat.get_template_prompts(
            property_key)
        instruction_text = native_instr
        feature_key_prompt = "Molecule"      # matches run_cell
        output_key_prompt = native_var       # e.g. "logKow"
    else:
        instruction_text = B.instruction_block(spec.condition, spec.rid)
        feature_key_prompt = B.feature_label(spec.condition)
        output_key_prompt = B.output_label(spec.rid)

    print(f"\n{'='*68}")
    print(f"  BLINDED CELL   {model_slug} x {property_key} "
          f"[{spec.condition}]")
    print(f"  train={len(df_train)} val={len(df_val)} "
          f"median={median:.4f} M_c={M_c} N_shots={N_shots}")
    print(f"  transform={spec.transform.name} "
          f"recoded={'yes' if spec.condition==B.AGNOSTIC else 'no'}")
    print(f"{'='*68}\n")

    # ---- Split (identical seed to run_cell) -------------------------
    train_test_data, test_data = rchat.train_test_split(
        df_train, test_size=test_split, random_state=seed_base
    )
    training_pool = train_test_data.to_dict("records")
    test_pool = test_data.to_dict("records")
    validation_samples = df_val.to_dict("records")

    Rate = 6000 if _backend_name() == "LOCAL" else 30

    metric_keys = ["MSE", "RMSE", "MAE", "R2", "Acc", "F1",
                   "MAE_t", "RMSE_t", "R2_t",
                   "NRMSE_range", "n_rejects", "n_out_of_range"]
    per_cycle = {k: np.full(M_c, np.nan) for k in metric_keys}
    per_cycle["seed"] = np.zeros(M_c, dtype=np.int64)

    best_MAE = np.inf
    best_cycle = -1
    optimal_training_samples = []

    t_cell_start = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for k in range(M_c):
            seed = seed_base + k
            rng = random.Random(seed)
            per_cycle["seed"][k] = seed

            n_sub = min(N_shots, len(training_pool))
            training_samples = rng.sample(training_pool, n_sub)

            # Fit the label transform on THIS cycle's training pool only.
            # (For NATIVE this is Identity and a no-op.)
            y_train_cycle = [float(r[property_var]) for r in training_samples]
            spec.transform.fit(y_train_cycle)

            print(f"    Cycle {k+1:>3}/{M_c} seed={seed} "
                  f"[{spec.condition}]", end="", flush=True)

            try:
                res = k_shot_test_blinded(
                    training_samples, test_pool, median,
                    instruction_text, feature_key_prompt, output_key_prompt,
                    property_var, spec, Rate, model_name,
                    struct_col=feature_col, temperature=temperature,
                )
            except (KeyError, ValueError, AttributeError, TypeError) as e:
                # These are structural errors (bad column name, unfitted
                # transform, malformed record). They will recur on every
                # cycle, so failing fast is far better than silently
                # no-oping 50 times and emitting an all-NaN rollup that
                # looks like a completed run. Re-raise so the cell-level
                # handler in the driver marks this cell FAILED.
                print(f"  FATAL (structural, will recur every cycle): "
                      f"{type(e).__name__}: {e}")
                raise
            except Exception as e:
                # Genuinely per-cycle/transient issues: log and continue.
                print(f"  ERROR (cycle {k+1}): {type(e).__name__}: {e}")
                continue

            per_cycle["MSE"][k] = res["MSE"]
            per_cycle["RMSE"][k] = res["RMSE"]
            per_cycle["MAE"][k] = res["MAE"]
            per_cycle["R2"][k] = res["R2"]
            per_cycle["Acc"][k] = res["Acc"]
            per_cycle["F1"][k] = res["F1"]
            per_cycle["MAE_t"][k] = res["MAE_t"]
            per_cycle["RMSE_t"][k] = res["RMSE_t"]
            per_cycle["R2_t"][k] = res["R2_t"]
            per_cycle["n_rejects"][k] = res["n_rejects"]
            per_cycle["n_out_of_range"][k] = res["n_out_of_range"]
            if not np.isnan(res["RMSE"]) and (y_max - y_min) > 0:
                per_cycle["NRMSE_range"][k] = (
                    res["RMSE"] / (y_max - y_min) * 100.0)

            print(f"  MAE={res['MAE']:.3f} RMSE={res['RMSE']:.3f} "
                  f"R2={res['R2']:.3f} oor={res['n_out_of_range']} "
                  f"n_rej={res['n_rejects']}")

            if not np.isnan(res["MAE"]) and res["MAE"] < best_MAE:
                best_MAE = res["MAE"]
                best_cycle = k
                optimal_training_samples = training_samples

    t_testing = time.time() - t_cell_start

    # ---- Validation on optimal demos (native-unit scoring) ----------
    val_metrics = {k: float("nan") for k in metric_keys}
    if optimal_training_samples:
        # Refit transform on the optimal cycle's training pool so the
        # validation labels use the same map the model saw at selection.
        y_opt = [float(r[property_var]) for r in optimal_training_samples]
        spec.transform.fit(y_opt)
        print(f"\n  Validation using cycle {best_cycle+1} demos "
              f"(MAE={best_MAE:.4f}) [{spec.condition}]")
        try:
            vres = k_shot_test_blinded(
                optimal_training_samples, validation_samples, median,
                instruction_text, feature_key_prompt, output_key_prompt,
                property_var, spec, Rate, model_name,
                struct_col=feature_col, temperature=temperature,
            )
            val_metrics["MSE"] = vres["MSE"]
            val_metrics["RMSE"] = vres["RMSE"]
            val_metrics["MAE"] = vres["MAE"]
            val_metrics["R2"] = vres["R2"]
            val_metrics["Acc"] = vres["Acc"]
            val_metrics["F1"] = vres["F1"]
            val_metrics["MAE_t"] = vres["MAE_t"]
            val_metrics["RMSE_t"] = vres["RMSE_t"]
            val_metrics["R2_t"] = vres["R2_t"]
            val_metrics["n_rejects"] = vres["n_rejects"]
            val_metrics["n_out_of_range"] = vres["n_out_of_range"]
            if not np.isnan(vres["RMSE"]) and (y_max - y_min) > 0:
                val_metrics["NRMSE_range"] = (
                    vres["RMSE"] / (y_max - y_min) * 100.0)
            print(f"  Val: MAE={vres['MAE']:.3f} RMSE={vres['RMSE']:.3f} "
                  f"R2={vres['R2']:.3f} Acc={vres['Acc']:.2f}% "
                  f"oor={vres['n_out_of_range']} n_rej={vres['n_rejects']}")
        except Exception as e:
            print(f"  Validation ERROR: {type(e).__name__}: {e}")

    t_total = time.time() - t_cell_start

    # ---- Persist per-cycle NPZ --------------------------------------
    npz_path = os.path.join(out_dir, "cycles_blinded.npz")
    np.savez_compressed(
        npz_path,
        condition=str(spec.condition),
        transform=str(spec.transform.name),
        MAE_test=per_cycle["MAE"],
        RMSE_test=per_cycle["RMSE"],
        MSE_test=per_cycle["MSE"],
        R2_test=per_cycle["R2"],
        Acc_test=per_cycle["Acc"],
        F1_test=per_cycle["F1"],
        MAE_test_trans=per_cycle["MAE_t"],
        RMSE_test_trans=per_cycle["RMSE_t"],
        R2_test_trans=per_cycle["R2_t"],
        NRMSE_range_test=per_cycle["NRMSE_range"],
        n_rejects_test=per_cycle["n_rejects"],
        n_out_of_range_test=per_cycle["n_out_of_range"],
        seed_per_cycle=per_cycle["seed"],
        best_cycle=np.int64(best_cycle),
        MAE_val=np.float64(val_metrics["MAE"]),
        RMSE_val=np.float64(val_metrics["RMSE"]),
        R2_val=np.float64(val_metrics["R2"]),
        Acc_val=np.float64(val_metrics["Acc"]),
        F1_val=np.float64(val_metrics["F1"]),
        MAE_val_trans=np.float64(val_metrics["MAE_t"]),
        NRMSE_range_val=np.float64(val_metrics["NRMSE_range"]),
        n_rejects_val=np.float64(val_metrics["n_rejects"]),
        n_out_of_range_val=np.float64(val_metrics["n_out_of_range"]),
        median_value=np.float64(median),
        mean_value=np.float64(mean_all),
        y_min=np.float64(y_min),
        y_max=np.float64(y_max),
        wall_time_sec=np.float64(t_total),
        wall_time_testing_sec=np.float64(t_testing),
    )
    print(f"\n  cycles_blinded.npz written to {npz_path}")

    # ---- Dump the transform / recoder provenance for SI -------------
    import json
    with open(os.path.join(out_dir, "blinding_spec.json"), "w") as fh:
        json.dump(spec.describe(), fh, indent=2)

    # ---- Rollup -----------------------------------------------------
    def _nanstat(a, stat):
        arr = np.asarray(a, dtype=float)
        m = ~np.isnan(arr)
        if not m.any():
            return float("nan")
        return float(stat(arr[m]))

    mae_arr = per_cycle["MAE"]
    q1 = _nanstat(mae_arr, lambda a: np.percentile(a, 25))
    q3 = _nanstat(mae_arr, lambda a: np.percentile(a, 75))

    rollup = {
        "model_slug": model_slug,
        "model_name": model_name,
        "property": property_key,
        "condition": spec.condition,
        "transform": spec.transform.name,
        "backend": _backend_name(),
        "M_c": M_c,
        "N_shots": N_shots,
        "median_MAE_test": _nanstat(mae_arr, np.median),
        "IQR_MAE_test": (q3 - q1) if (q3 == q3 and q1 == q1) else float("nan"),
        "mean_MAE_test": _nanstat(mae_arr, np.mean),
        "std_MAE_test": (_nanstat(mae_arr, lambda a: np.std(a, ddof=1))
                         if np.sum(~np.isnan(mae_arr)) > 1 else float("nan")),
        "best_MAE_test": float(best_MAE) if best_MAE != np.inf else float("nan"),
        "best_cycle": int(best_cycle) + 1,
        "MAE_val": val_metrics["MAE"],
        "RMSE_val": val_metrics["RMSE"],
        "R2_val": val_metrics["R2"],
        "Acc_val": val_metrics["Acc"],
        "F1_val": val_metrics["F1"],
        "NRMSE_range_val": val_metrics["NRMSE_range"],
        "MAE_val_trans": val_metrics["MAE_t"],
        "n_rejects_test_total": int(np.nansum(per_cycle["n_rejects"])),
        "n_out_of_range_test_total": int(np.nansum(per_cycle["n_out_of_range"])),
        "n_rejects_val": val_metrics["n_rejects"],
        "n_out_of_range_val": val_metrics["n_out_of_range"],
        "wall_time_h": t_total / 3600.0,
    }

    with open(os.path.join(out_dir, "testing_summary.txt"), "w") as fh:
        fh.write(f"Cell: {model_slug} x {property_key} [{spec.condition}]\n")
        fh.write(f"Transform: {spec.transform.name}\n")
        fh.write(f"Model: {model_name}\n")
        fh.write(f"M_c={M_c}, N_shots={N_shots}, seed_base={seed_base}\n")
        fh.write(f"Temperature: "
                 f"{temperature if temperature is not None else 'default'}\n")
        fh.write(f"Testing dataset: {len(test_pool)} molecules\n")
        fh.write(f"Validation dataset: {len(validation_samples)} molecules\n")
        fh.write(f"Median (binary-class threshold): {median:.4f}\n")
        fh.write("\n--- Per-cycle MAE (native units) ---\n")
        fh.write(f"Median MAE : {rollup['median_MAE_test']:.4f}\n")
        fh.write(f"Mean MAE   : {rollup['mean_MAE_test']:.4f} "
                 f"(std {rollup['std_MAE_test']:.4f})\n")
        fh.write(f"IQR        : {rollup['IQR_MAE_test']:.4f}\n")
        fh.write(f"Best MAE   : {rollup['best_MAE_test']:.4f} "
                 f"(cycle {rollup['best_cycle']})\n")
        fh.write(f"n_rejects  : {rollup['n_rejects_test_total']}\n")
        fh.write(f"n_out_of_range: {rollup['n_out_of_range_test_total']}\n")
        fh.write(f"wall time  : {t_total/60:.1f} min\n")

    return rollup


# ===========================================================================
#           Global recoder construction helper
# ===========================================================================


def build_global_recoder(properties_config, property_keys, seed=20260101):
    """
    Build ONE SmilesRecoder over the union of all SMILES across the
    requested properties (train + validation), per the notes'
    recommendation of a single global mapping for the whole experiment.

    Returns a built SmilesRecoder ready to place in an AGNOSTIC spec.
    """
    all_smiles = []
    for pkey in property_keys:
        cfg = properties_config[pkey]
        var = cfg["var"]
        feature_col = cfg.get("feature_col", "Molecule")
        for path_key in ("training", "validation"):
            path = cfg[path_key]
            if not os.path.exists(path):
                print(f"[recoder] missing {path_key} CSV for {pkey}: {path}")
                continue
            df = pd.DataFrame(rchat.read_csv_file(path))
            col = feature_col if feature_col in df.columns else None
            if col is None:
                cand = [c for c in df.columns if c != var]
                col = cand[0] if cand else None
            if col is None:
                continue
            all_smiles.extend([str(s) for s in df[col].tolist()])

    recoder = B.SmilesRecoder(seed=seed).build(all_smiles)
    print(f"[recoder] built global map over {len(all_smiles)} SMILES; "
          f"{recoder.describe()['n_symbols']} atom symbols")
    return recoder
