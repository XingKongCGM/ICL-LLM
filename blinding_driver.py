#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blinding_driver.py

Created on Tue Aug 26 14:17:09 2025

@author: Carlos Gonzalez
Material Data Division,
National Institute of Standatds and Technology,
Gaithersburg, MD. 20889


==================

Systematic sweep over (model, property, condition) cells for the
controlled information-access (blinding) experiment.

This is the blinded analogue of batch_driver.py.  It reuses the same
config format, the same model slugs, and the same serial-execution
discipline (one model resident at a time; the module-level backend
state in RCHAT_PChem_Properties is not thread-safe, so models are run
one at a time, matching batch_driver.py).

For each (model, property) pair it runs one cell PER requested
condition, all sharing the SAME seed_base so that native and blinded
arms are cycle-paired (same split, same demonstration identities and
ordering, same query order).  That pairing is what the downstream
paired-dMAE / retention-ratio analysis in blinding_analysis.py relies
on.

Output layout
-------------

    RESULTS_ROOT/blinding_runs/{run_id}/
        manifest.json                      # settings of the FIRST sweep
        manifest_<timestamp>.json          # one per later append
        summary_blinded.csv                # one row per (cell, condition)
        recoder_map.json                   # global SMILES map (if agnostic)
        {model_slug}_{property}_{condition}/
            cycles_blinded.npz
            blinding_spec.json
            testing_summary.txt

Usage
-----

    python blinding_driver.py \\
        --config properties.yaml \\
        --models 2 7 \\
        --properties LOGKOW LOGS PKA LOGCMC \\
        --conditions native generic \\
        --transform zscore \\
        --M-cycles 50 \\
        --results-root ./RESULTS \\
        --run-id blind_zscore_native_generic \\
        [--seed-base 0] [--recoder-seed 20260101] [--temperature 0.9] \\
        [--resume] [--dry-run] [--notify]

Notes
-----
  * --conditions accepts any subset of {native, generic, agnostic}.
  * --transform in {zscore, linear, tanh} selects the label map for the
    blinded conditions (native always uses identity).  zscore has a
    whole-real-line inverse and is the recommended choice.
  * agnostic requires a global SMILES recoder, built once over all
    requested properties before the sweep starts.
  * Appending a condition to an existing --run-id is supported.  The
    driver compares the new settings with the run's existing manifest
    and recoder map and refuses to continue if they would break cycle
    pairing (override with --allow-mismatch).
  * Model numbers follow batch_driver.MODEL_SLUG (local-only core):
    2 = Llama-3.3-70B, 7 = Llama-4-Scout-17B-16E.  Older runs that used
    "8" for Scout have the same folder names (folders use slugs, not
    numbers), so they can be appended to with "--models 7".

No em dashes are used in output strings (project style).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

# Reuse the config loader, slugs, and notify helpers from batch_driver.
import batch_driver as bd

import blinding as B


# summary_blinded.csv columns
SUMMARY_COLUMNS = [
    "model_slug", "model_name", "property", "condition", "transform",
    "backend", "M_c", "N_shots",
    "median_MAE_test", "IQR_MAE_test", "mean_MAE_test", "std_MAE_test",
    "best_MAE_test", "best_cycle",
    "MAE_val", "RMSE_val", "R2_val", "Acc_val", "F1_val",
    "NRMSE_range_val", "MAE_val_trans",
    "n_rejects_test_total", "n_out_of_range_test_total",
    "n_rejects_val", "n_out_of_range_val",
    "wall_time_h",
]

# Manifest keys that must agree between an existing run and an append,
# otherwise the arms are not validly cycle-paired or not comparable.
PAIRING_KEYS = [
    "seed_base", "M_cycles", "N_shots", "transform", "value_fmt",
    "out_max", "tanh_center", "tanh_amp", "std_center", "std_scale",
    "temperature",
]


def _append_summary_row(summary_csv, lock, row):
    complete = {c: row.get(c, "") for c in SUMMARY_COLUMNS}
    with lock:
        exists = os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
            if not exists:
                w.writeheader()
            w.writerow(complete)


def _spec_for(condition, transform_kind, recoder, args):
    """Construct a BlindingSpec for one condition."""
    return B.make_spec(
        condition=condition,
        transform_kind=transform_kind,
        recoder=recoder if condition == B.AGNOSTIC else None,
        out_max=args.out_max,
        tanh_center=args.tanh_center,
        tanh_amp=args.tanh_amp,
        std_center=args.std_center,
        std_scale=args.std_scale,
        value_fmt=args.value_fmt,
    )


def _run_one_cell(hooks, model_choice, model_name, property_key, prop_cfg,
                  conditions, transform_kind, recoder, run_dir, args,
                  summary_lock, summary_csv, progress):
    """Run every requested condition for one (model, property) pair."""
    slug = bd.MODEL_SLUG[model_choice]

    data_paths = {
        "training": prop_cfg["training"],
        "validation": prop_cfg["validation"],
        "var": prop_cfg["var"],
        "title": prop_cfg.get("title", property_key),
        "units": prop_cfg.get("units", "-"),
        "feature_col": prop_cfg.get("feature_col", "Molecule"),
    }

    for kind, path in [("training", data_paths["training"]),
                       ("validation", data_paths["validation"])]:
        if not os.path.exists(path):
            print(f"  [ERROR] MISSING {kind} CSV for {property_key}: {path}")
            _append_summary_row(summary_csv, summary_lock, {
                "model_slug": slug, "model_name": model_name,
                "property": property_key, "condition": "ALL",
                "backend": "SKIPPED", "M_c": args.M_cycles,
                "N_shots": args.N_shots,
            })
            progress["done"] += len(conditions)
            return

    for condition in conditions:
        progress["done"] += 1
        cell_folder = f"{slug}_{property_key}_{condition}"
        out_dir = Path(run_dir) / cell_folder

        print(f"\n{'#'*72}")
        print(f"#  BLINDED CELL {progress['done']}/{progress['total']}: "
              f"{slug} x {property_key} [{condition}]")
        print(f"{'#'*72}")

        if args.resume and (out_dir / "cycles_blinded.npz").exists():
            print(f"  [resume] {cell_folder}/cycles_blinded.npz exists; "
                  f"skipping (delete the folder to re-run it).")
            continue

        spec = _spec_for(condition, transform_kind, recoder, args)
        try:
            rollup = hooks.run_cell_blinded(
                model_name=model_name,
                model_slug=slug,
                property_key=property_key,
                data_paths=data_paths,
                out_dir=str(out_dir),
                spec=spec,
                M_c=args.M_cycles,
                N_shots=args.N_shots,
                test_split=0.20,
                seed_base=args.seed_base,
                temperature=args.temperature,
            )
            _append_summary_row(summary_csv, summary_lock, rollup)
        except Exception as e:
            print(f"  [CELL FAILED] {type(e).__name__}: {e}")
            traceback.print_exc()
            _append_summary_row(summary_csv, summary_lock, {
                "model_slug": slug, "model_name": model_name,
                "property": property_key, "condition": condition,
                "transform": transform_kind, "backend": "FAILED",
                "M_c": args.M_cycles, "N_shots": args.N_shots,
            })


def _run_model_sweep(rchat_mod, hooks, model_choice, properties_to_run,
                     properties_config, conditions, transform_kind, recoder,
                     run_dir, args, summary_lock, summary_csv, progress):
    slug = bd.MODEL_SLUG[model_choice]
    print(f"\n{'*'*72}")
    print(f"*  BLINDED MODEL SWEEP {slug} "
          f"({len(properties_to_run)} properties x "
          f"{len(conditions)} conditions)")
    print(f"{'*'*72}\n")

    if args.resume:
        # Skip the (slow) model load if every cell for this model is done.
        pending = [
            (p, c) for p in properties_to_run for c in conditions
            if not (Path(run_dir) / f"{slug}_{p}_{c}" /
                    "cycles_blinded.npz").exists()
        ]
        if not pending:
            print(f"  [resume] all cells for {slug} already exist; "
                  f"model not loaded.")
            progress["done"] += len(properties_to_run) * len(conditions)
            return

    try:
        model_name = rchat_mod.select_llm_model_by_choice(model_choice)
    except Exception as e:
        print(f"  [MODEL LOAD FAILED] {slug}: {type(e).__name__}: {e}")
        progress["done"] += len(properties_to_run) * len(conditions)
        return

    for pkey in properties_to_run:
        _run_one_cell(hooks, model_choice, model_name, pkey,
                      properties_config[pkey], conditions, transform_kind,
                      recoder, run_dir, args, summary_lock, summary_csv,
                      progress)


def _check_pairing(existing, new, allow_mismatch):
    """Compare PAIRING_KEYS between an existing manifest and the new one.
    Keys absent from the older manifest are reported but not enforced."""
    mismatches, unknown = [], []
    for k in PAIRING_KEYS:
        if k not in existing:
            unknown.append(k)
            continue
        if existing.get(k) != new.get(k):
            mismatches.append((k, existing.get(k), new.get(k)))
    if unknown:
        print(f"[driver] NOTE: existing manifest does not record "
              f"{unknown}; cannot verify those settings. Make sure they "
              f"match the original sweep.")
    if mismatches:
        print("[driver] Settings differ from the existing run in this "
              "folder; arms would NOT be validly cycle-paired:")
        for k, old, new_v in mismatches:
            print(f"           {k}: existing={old!r}  new={new_v!r}")
        if not allow_mismatch:
            print("[driver] Refusing to append. Use a new --run-id, match "
                  "the settings, or pass --allow-mismatch.")
            sys.exit(2)
        print("[driver] --allow-mismatch given; continuing anyway.")


def main():
    parser = argparse.ArgumentParser(
        description="Blinding sweep for the ICL information-access study.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", type=int, required=True,
                        help="Model choices from batch_driver.MODEL_SLUG "
                             "(the study uses 2 = Llama-3.3-70B and "
                             "7 = Llama-4-Scout)")
    parser.add_argument("--properties", nargs="+", default=None)
    parser.add_argument("--conditions", nargs="+", default=["native", "generic"],
                        choices=list(B.VALID_CONDITIONS),
                        help="Subset of native generic agnostic")
    parser.add_argument("--transform", default="zscore",
                        choices=["zscore", "linear", "tanh"],
                        help="Label transform for blinded conditions "
                             "(default zscore). 'zscore' "
                             "(LinearStandardize) has a whole-real-line "
                             "inverse and cannot produce out-of-range "
                             "outputs. 'linear' (invert + rescale to "
                             "[0, out_max]) and 'tanh' have bounded "
                             "windows and can yield out-of-range "
                             "predictions.")
    parser.add_argument("--M-cycles", type=int, default=50)
    parser.add_argument("--N-shots", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=0,
                        help="Base seed; MUST match across conditions for "
                             "cycle pairing (default 0)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature for every query. Omit "
                             "for the core default (0.9), which "
                             "reproduces the published runs.")
    parser.add_argument("--out-max", type=float, default=100.0,
                        help="Upper bound of the linear rescale window")
    parser.add_argument("--tanh-center", type=float, default=50.0)
    parser.add_argument("--tanh-amp", type=float, default=40.0)
    parser.add_argument("--std-center", type=float, default=500.0,
                        help="Center of the zscore (LinearStandardize) map")
    parser.add_argument("--std-scale", type=float, default=100.0,
                        help="Scale of the zscore map; use a negative "
                             "value to also flip label ordering")
    parser.add_argument("--value-fmt", default="{:.3f}",
                        help="Format string for demonstration labels "
                             "(all conditions, including native)")
    parser.add_argument("--recoder-seed", type=int, default=20260101)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells whose cycles_blinded.npz already "
                             "exists (restart an interrupted sweep)")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="Append to an existing run even if settings "
                             "differ from its manifest (breaks pairing)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--rchat-script-dir", default=None)
    args = parser.parse_args()

    # Validate model numbers early, with a hint for the old numbering.
    for m in args.models:
        if m not in bd.MODEL_SLUG:
            hint = (" (Llama-4-Scout was model 8 in the older core; it is "
                    "now 7.)" if m == 8 else "")
            parser.error(f"model choice {m} not valid; available: "
                         f"{sorted(bd.MODEL_SLUG)}{hint}")

    script_dir = args.rchat_script_dir or os.path.dirname(
        os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    print(f"[driver] Importing modules from {script_dir}")
    import RCHAT_PChem_Properties as rchat_mod
    import rchat_blinding_hooks as hooks

    cfg = bd._load_yaml_config(args.config)
    all_props = cfg["properties"]
    properties_to_run = args.properties or sorted(all_props.keys())
    for p in properties_to_run:
        if p not in all_props:
            raise KeyError(f"Property {p!r} not in config. "
                           f"Available: {sorted(all_props.keys())}")

    conditions = [c.lower() for c in args.conditions]

    # Run id
    if args.run_id:
        run_id = args.run_id
    else:
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        tag = f"M{args.M_cycles}_{args.transform}_" + "-".join(conditions)
        run_id = f"{stamp}_blind_{tag}"

    run_dir = Path(args.results_root) / "blinding_runs" / run_id
    existing_manifest_path = run_dir / "manifest.json"
    appending = existing_manifest_path.exists()
    summary_csv = str(run_dir / "summary_blinded.csv")

    manifest = {
        "run_id": run_id,
        "started_at": dt.datetime.now().isoformat(),
        "config_path": os.path.abspath(args.config),
        "models": args.models,
        "model_slugs": {str(m): bd.MODEL_SLUG.get(m, "?") for m in args.models},
        "properties": properties_to_run,
        "conditions": conditions,
        "transform": args.transform,
        "M_cycles": args.M_cycles,
        "N_shots": args.N_shots,
        "seed_base": args.seed_base,
        "temperature": args.temperature,
        "out_max": args.out_max,
        "tanh_center": args.tanh_center,
        "tanh_amp": args.tanh_amp,
        "std_center": args.std_center,
        "std_scale": args.std_scale,
        "value_fmt": args.value_fmt,
        "recoder_seed": args.recoder_seed if B.AGNOSTIC in conditions else None,
        "results_root": os.path.abspath(args.results_root),
        "run_dir": str(run_dir),
    }

    # Pairing guard when appending to an existing run.
    if appending:
        try:
            existing = json.loads(existing_manifest_path.read_text())
        except Exception as e:
            print(f"[driver] WARNING: could not read existing manifest "
                  f"({e}); pairing not verified.")
            existing = {}
        print(f"[driver] Run folder exists; appending to {run_dir}")
        _check_pairing(existing, manifest, args.allow_mismatch)

    n_cells = len(properties_to_run) * len(conditions) * len(args.models)
    print("=" * 72)
    print(f"  BLINDING SWEEP  {run_id}"
          f"{'   [APPEND]' if appending else ''}")
    print("=" * 72)
    print(f"  Run directory : {run_dir}")
    print(f"  Models        : {args.models} "
          f"({[bd.MODEL_SLUG[m] for m in args.models]})")
    print(f"  Properties    : {properties_to_run}")
    print(f"  Conditions    : {conditions}")
    print(f"  Transform     : {args.transform}")
    print(f"  M_cycles      : {args.M_cycles}   N_shots: {args.N_shots}")
    print(f"  seed_base     : {args.seed_base}  (paired across conditions)")
    print(f"  Temperature   : "
          f"{args.temperature if args.temperature is not None else 'default (0.9)'}")
    print(f"  Resume        : {args.resume}")
    print(f"  Total cells   : {n_cells}")
    print("=" * 72)

    if args.dry_run:
        print("\n[dry-run] Plan printed; nothing written, not executing.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    # Build the global SMILES recoder ONCE if any agnostic arm requested.
    recoder = None
    if B.AGNOSTIC in conditions:
        recoder = hooks.build_global_recoder(
            all_props, properties_to_run, seed=args.recoder_seed)
        recoder_path = run_dir / "recoder_map.json"
        new_map = recoder.describe()
        if recoder_path.exists():
            try:
                old_map = json.loads(recoder_path.read_text())
            except Exception:
                old_map = None
            if old_map is not None and old_map != new_map:
                print("[driver] Existing recoder_map.json differs from the "
                      "map built now (different seed or property set). "
                      "Agnostic cells in this folder would use "
                      "inconsistent codes.")
                if not args.allow_mismatch:
                    print("[driver] Refusing to continue. Use the same "
                          "--properties and --recoder-seed as before, a new "
                          "--run-id, or --allow-mismatch.")
                    sys.exit(2)
                alt = run_dir / f"recoder_map_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
                alt.write_text(json.dumps(new_map, indent=2))
                print(f"[driver] new map written to {alt}")
            else:
                print(f"[driver] recoder map matches {recoder_path}")
        else:
            recoder_path.write_text(json.dumps(new_map, indent=2))
            print(f"[driver] recoder map dumped to {recoder_path}")

    # Never overwrite the original manifest: later appends get their own.
    if appending:
        mpath = run_dir / f"manifest_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    else:
        mpath = existing_manifest_path
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[driver] manifest -> {mpath}")

    summary_lock = threading.Lock()
    progress = {"done": 0, "total": n_cells}
    t_start = time.time()
    try:
        for m in args.models:
            _run_model_sweep(
                rchat_mod, hooks, m, properties_to_run, all_props,
                conditions, args.transform, recoder, run_dir, args,
                summary_lock, summary_csv, progress)
    except KeyboardInterrupt:
        print("\n[driver] Interrupted. Completed cells are saved; rerun the "
              "same command with --resume to continue.")
        sys.exit(130)
    wall_h = (time.time() - t_start) / 3600.0

    n_ok = n_fail = 0
    if os.path.exists(summary_csv):
        with open(summary_csv) as fh:
            for row in csv.DictReader(fh):
                if row.get("backend") in ("LOCAL", "REMOTE"):
                    n_ok += 1
                elif row.get("backend") in ("FAILED", "SKIPPED"):
                    n_fail += 1

    print()
    print("=" * 72)
    print(f"  BLINDING SWEEP COMPLETE  wall {wall_h:.2f} h "
          f"(summary rows: {n_ok} ok, {n_fail} failed; "
          f"{n_cells} cells planned this invocation)")
    print(f"  Summary  {summary_csv}")
    print("=" * 72)

    if args.notify:
        bd._macos_notify("Blinding sweep complete",
                         f"{n_ok} ok, {n_fail} failed, {wall_h:.1f} h")


if __name__ == "__main__":
    main()
