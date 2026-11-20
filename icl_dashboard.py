#!/usr/bin/env python3
"""
icl_dashboard.py  --  Streamlit control panel for the ICL evaluation pipeline
and the blinding (information-access) study.

Created on Tue Aug 26 14:17:09 2025

@author: Carlos Gonzalez
Material Data Division,
National Institute of Standatds and Technology,
Gaithersburg, MD. 20889



Runs LOCALLY on your Mac: launches the drivers as a background subprocess,
streams the log live, monitors / stops the run, browses finished results,
and RE-ADOPTS a still-running job after a Streamlit restart or
browser-session reset (orphan-reattach).

SETUP (one time):
    pip install streamlit pyyaml pandas numpy matplotlib

RUN (from the repository folder):
    streamlit run icl_dashboard.py
    # opens http://localhost:8501

Phases (Run mode)
-----------------
    Testing / Predict / Forecast   batch_driver.py
    Analyze                        analysis.py
    Blinding sweep                 blinding_driver.py
    Blinding analysis              blinding_analysis.py (+ oor_extract.py)

Results mode browses RESULTS/runs/* (ICL sweeps) and
RESULTS/blinding_runs/* (blinding study), including per-cycle
diagnostics and, for blinding runs, a paired native-vs-blinded view.

LAUNCHER + MONITOR, not a scheduler. One job at a time (the pipeline is
serial and memory-bound; MLX is not thread-safe).

Memory note: the shell wrappers raise the GPU wired-memory limit with
`sudo sysctl iogpu.wired_limit_mb=...`; the dashboard does not (it cannot
prompt for a password). For large models, run that command once per boot
before launching from here.

Orphan-reattach (no psutil):
    On launch, a runlock JSON (PID, log path, command, start time) is
    written to .dashboard_logs/runlock.json. On page load, if the current
    session has no live process but the runlock points at a PID that is
    (a) alive and (b) whose `ps` command line still names one of this
    project's scripts, the dashboard re-adopts that run (shows its log,
    status, and a working Stop). Stale locks are cleared automatically.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# EDIT THESE TO MATCH YOUR ENVIRONMENT
# =============================================================================
PYTHON = sys.executable
DRIVER = "batch_driver.py"
ANALYSIS = "analysis.py"
BLIND_DRIVER = "blinding_driver.py"
BLIND_ANALYSIS = "blinding_analysis.py"
OOR_EXTRACT = "oor_extract.py"
CONFIG_DEFAULT = "properties.yaml"
RESULTS_ROOT_DEFAULT = "./RESULTS"

# Any process whose command line names one of these is "ours" (reattach).
OUR_SCRIPTS = (DRIVER, ANALYSIS, BLIND_DRIVER, BLIND_ANALYSIS, OOR_EXTRACT)

MODEL_LABELS = {
    1: "1 · Llama-3.1-8B (Q4)",
    2: "2 · Llama-3.3-70B (Q4)",
    3: "3 · Nemotron-Mini-4B (bf16)",
    4: "4 · Llama-3.1-Nemotron-70B (Q4)",
    5: "5 · Gemma-4-31B OptiQ (Q4)",
    6: "6 · Mistral-Small-3.1-24B (Q4)",
    7: "7 · Llama-4-Scout-17B-16E (Q4)",
}

# Blinding study defaults (match run_blinding.sh).
BLIND_DEFAULT_MODELS = [2, 7]
BLIND_DEFAULT_PROPS = ["LOGKOW", "LOGS", "PKA", "LOGCMC"]
BLIND_CONDITIONS = ["native", "generic", "agnostic"]
BLIND_TRANSFORMS = ["zscore", "linear", "tanh"]

# Settings that must match for cycle pairing when appending to a run.
# Kept in sync with blinding_driver.PAIRING_KEYS.
try:
    from blinding_driver import PAIRING_KEYS as BLIND_PAIRING_KEYS
except Exception:
    BLIND_PAIRING_KEYS = [
        "seed_base", "M_cycles", "N_shots", "transform", "value_fmt",
        "out_max", "tanh_center", "tanh_amp", "std_center", "std_scale",
        "temperature",
    ]

LOG_DIR = Path(".dashboard_logs")
RUNLOCK = LOG_DIR / "runlock.json"

SUMMARY_FILES = {
    "summary.csv":          "Testing",
    "predict_summary.csv":  "Predict",
    "forecast_summary.csv": "Forecast",
}

# Per-cycle array keys in cycles.npz (Testing) with friendly labels.
CYCLE_METRICS = {
    "MAE_test":        "MAE",
    "RMSE_test":       "RMSE",
    "MSE_test":        "MSE",
    "R2_test":         "R²",
    "Acc_test":        "Accuracy (%)",
    "F1_test":         "F1 (macro)",
    "NRMSE_mean_test": "NRMSE (mean) %",
    "NRMSE_range_test":"NRMSE (range) %",
    "n_rejects_test":  "n_rejects",
}

# Per-cycle array keys in cycles_blinded.npz (blinding study).
BLIND_CYCLE_METRICS = {
    "MAE_test":            "MAE (native units)",
    "RMSE_test":           "RMSE (native units)",
    "MSE_test":            "MSE (native units)",
    "R2_test":             "R²",
    "Acc_test":            "Accuracy (%)",
    "F1_test":             "F1 (macro)",
    "NRMSE_range_test":    "NRMSE (range) %",
    "MAE_test_trans":      "MAE (transformed units)",
    "n_rejects_test":      "n_rejects",
    "n_out_of_range_test": "n_out_of_range",
}
# =============================================================================


st.set_page_config(page_title="ICL Sweep Dashboard", layout="wide")


# -----------------------------------------------------------------------------
# Streamlit version compatibility (use_container_width -> width="stretch")
# -----------------------------------------------------------------------------
def _code_wrapped(text: str, language: str = "bash"):
    """st.code with line wrapping (Streamlit >= 1.39), so long commands
    stay fully visible; plain st.code on older versions."""
    try:
        st.code(text, language=language, wrap_lines=True)
    except TypeError:
        st.code(text, language=language)


def _stretch(fn, *args, **kwargs):
    """Call a Streamlit element full-width on both old and new versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


# -----------------------------------------------------------------------------
# properties.yaml parsing
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_property_keys(config_path: str) -> list[str]:
    try:
        import yaml
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        return sorted((cfg.get("properties", {}) or {}).keys())
    except ImportError:
        keys, in_props = [], False
        with open(config_path) as fh:
            for line in fh:
                raw = line.rstrip("\n")
                if not raw.strip() or raw.lstrip().startswith("#"):
                    continue
                indent = len(raw) - len(raw.lstrip(" "))
                content = raw.strip()
                if indent == 0:
                    in_props = (content.rstrip() == "properties:")
                elif indent == 2 and in_props and content.endswith(":"):
                    keys.append(content[:-1])
        return sorted(keys)
    except FileNotFoundError:
        return []


# -----------------------------------------------------------------------------
# Low-level PID helpers (no psutil)
# -----------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    """True if a process with this PID exists. os.kill(pid, 0) raises
    ProcessLookupError if not, PermissionError if it exists but isn't ours."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_cmdline(pid: int) -> str:
    """Return the process command line via `ps`, or '' if unavailable."""
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _is_our_driver(pid: int) -> bool:
    """Alive AND its command line names one of this project's scripts.
    Guards against PID reuse by unrelated processes."""
    if not isinstance(pid, int) or not _pid_alive(pid):
        return False
    cmd = _pid_cmdline(pid)
    return any(s in cmd for s in OUR_SCRIPTS)


# -----------------------------------------------------------------------------
# Runlock persistence
# -----------------------------------------------------------------------------
def _write_runlock(pid: int, log_path: str, cmd: list[str], started_iso: str):
    LOG_DIR.mkdir(exist_ok=True)
    RUNLOCK.write_text(json.dumps({
        "pid": pid,
        "log_path": log_path,
        "cmd": cmd,
        "started": started_iso,
    }, indent=2))


def _read_runlock() -> dict | None:
    if not RUNLOCK.exists():
        return None
    try:
        return json.loads(RUNLOCK.read_text())
    except Exception:
        return None


def _clear_runlock():
    try:
        RUNLOCK.unlink(missing_ok=True)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Subprocess state (survives reruns via session_state; reattach via runlock)
# -----------------------------------------------------------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("proc", None)         # Popen (this session launched it)
    ss.setdefault("log_path", None)
    ss.setdefault("run_cmd", None)
    ss.setdefault("run_started", None)
    ss.setdefault("adopted_pid", None)  # int if re-adopted from runlock
    ss.setdefault("reattach_checked", False)


def _try_reattach():
    """Adopt a still-living job from the runlock if this session has none.
    Clears stale locks. Runs once per session."""
    ss = st.session_state
    if ss.reattach_checked:
        return
    ss.reattach_checked = True

    if ss.proc is not None and ss.proc.poll() is None:
        return

    lock = _read_runlock()
    if not lock:
        return
    pid = lock.get("pid")
    if not isinstance(pid, int):
        _clear_runlock()
        return

    if _is_our_driver(pid):
        ss.adopted_pid = pid
        ss.log_path = Path(lock["log_path"]) if lock.get("log_path") else None
        ss.run_cmd = lock.get("cmd")
        try:
            ss.run_started = datetime.fromisoformat(lock["started"])
        except Exception:
            ss.run_started = None
    else:
        _clear_runlock()


def _is_running() -> bool:
    """Running if we own a live Popen OR we adopted a live job PID."""
    ss = st.session_state
    if ss.proc is not None and ss.proc.poll() is None:
        return True
    if ss.adopted_pid is not None and _is_our_driver(ss.adopted_pid):
        return True
    if ss.adopted_pid is not None and not _is_our_driver(ss.adopted_pid):
        ss.adopted_pid = None
        _clear_runlock()
    return False


def _current_pid() -> int | None:
    ss = st.session_state
    if ss.proc is not None and ss.proc.poll() is None:
        return ss.proc.pid
    if ss.adopted_pid is not None:
        return ss.adopted_pid
    return None


def _launch(cmd: list[str]):
    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{stamp}.log"
    log_fh = open(log_path, "w", buffering=1)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )
    ss = st.session_state
    ss.proc = proc
    ss.adopted_pid = None
    ss.log_path = log_path
    ss.run_cmd = cmd
    ss.run_started = datetime.now()
    _write_runlock(proc.pid, str(log_path), cmd, ss.run_started.isoformat())


def _stop():
    """Terminate the running process group (owned OR adopted) by PID."""
    pid = _current_pid()
    if pid is None:
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    time.sleep(1.0)
    if not _is_our_driver(pid):
        _clear_runlock()
        st.session_state.adopted_pid = None


def _read_log(tail_lines: int = 400) -> str:
    lp = st.session_state.get("log_path")
    if not lp or not Path(lp).exists():
        return ""
    with open(lp, errors="replace") as fh:
        lines = fh.readlines()
    return "".join(lines[-tail_lines:])


# -----------------------------------------------------------------------------
# Results helpers: ICL sweeps
# -----------------------------------------------------------------------------
def _list_dirs_newest_first(parent: Path) -> list[str]:
    if not parent.is_dir():
        return []
    runs = [p.name for p in parent.iterdir() if p.is_dir()]
    runs.sort(key=lambda r: (parent / r).stat().st_mtime, reverse=True)
    return runs


def list_runs(results_root: str) -> list[str]:
    return _list_dirs_newest_first(Path(results_root) / "runs")


def find_summaries(results_root: str, run_id: str) -> dict[str, Path]:
    run_dir = Path(results_root) / "runs" / run_id
    found = {}
    for fname, label in SUMMARY_FILES.items():
        p = run_dir / fname
        if p.exists():
            found[label] = p
    return found


def run_markers(results_root: str, run_id: str) -> list[str]:
    run_dir = Path(results_root) / "runs" / run_id
    if not run_dir.is_dir():
        return []
    return sorted(p.name for p in run_dir.glob("RUN_COMPLETE*.txt"))


def list_cells_with_npz(run_dir: Path, npz_name: str) -> list[str]:
    """Cell-folder names under run_dir that contain npz_name, sorted."""
    if not run_dir.is_dir():
        return []
    return sorted(p.name for p in run_dir.iterdir()
                  if p.is_dir() and (p / npz_name).exists())


def load_npz(path: Path) -> dict | None:
    """Load an NPZ into a plain dict (no pickle). None if unreadable."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Results helpers: blinding runs
# -----------------------------------------------------------------------------
def list_blinding_runs(results_root: str) -> list[str]:
    return _list_dirs_newest_first(Path(results_root) / "blinding_runs")


def load_blinding_manifests(run_dir: Path) -> list[dict]:
    """manifest.json first, then manifest_<timestamp>.json in order."""
    out = []
    first = run_dir / "manifest.json"
    paths = ([first] if first.exists() else []) + sorted(
        run_dir.glob("manifest_*.json"))
    for p in paths:
        try:
            d = json.loads(p.read_text())
            d["_file"] = p.name
            out.append(d)
        except Exception:
            pass
    return out


def parse_blind_cell(folder: str):
    """'<slug>_<PROP>_<condition>' -> (slug, prop, condition), or None.
    Slugs and property keys contain no underscores."""
    parts = folder.rsplit("_", 2)
    if len(parts) != 3 or parts[2] not in BLIND_CONDITIONS:
        return None
    return tuple(parts)


def blinding_cells(run_dir: Path) -> dict:
    """(slug, prop, condition) -> cell folder Path, for cells with an NPZ."""
    cells = {}
    for name in list_cells_with_npz(run_dir, "cycles_blinded.npz"):
        key = parse_blind_cell(name)
        if key:
            cells[key] = run_dir / name
    return cells


def blinding_expected_cells(manifests: list[dict]) -> set:
    """All (slug, prop, condition) cells the manifests planned."""
    exp = set()
    for m in manifests:
        slugs = list((m.get("model_slugs") or {}).values())
        for s in slugs:
            for p in m.get("properties", []):
                for c in m.get("conditions", []):
                    exp.add((s, p, c))
    return exp


def auto_blind_run_id(transform: str, conditions: list[str]) -> str:
    """Same default names as run_blinding.sh."""
    conds = set(conditions)
    if conds == {"native", "generic"}:
        tail = "native_generic"
    elif conds == set(BLIND_CONDITIONS):
        tail = "three_condition"
    else:
        tail = "_".join(c for c in BLIND_CONDITIONS if c in conds) or "none"
    return f"blind_{transform}_{tail}"


def blinding_pairing_mismatches(manifest: dict, new: dict) -> list:
    """[(key, existing, new)] for pairing keys that differ. Keys missing
    from an older manifest are skipped (the driver reports them)."""
    out = []
    for k in BLIND_PAIRING_KEYS:
        if k in manifest and manifest.get(k) != new.get(k):
            out.append((k, manifest.get(k), new.get(k)))
    return out


def paired_view_stats(native: dict, blinded: dict) -> dict:
    """Retention ratio and paired dMAE summary from two NPZ dicts."""
    mn = np.asarray(native["MAE_test"], dtype=float)
    mb = np.asarray(blinded["MAE_test"], dtype=float)
    sn = np.asarray(native.get("seed_per_cycle", []))
    sb = np.asarray(blinded.get("seed_per_cycle", []))
    paired = sn.shape == sb.shape and sn.size > 0 and np.array_equal(sn, sb)
    n = min(len(mn), len(mb))
    mn, mb = mn[:n], mb[:n]
    ok = ~np.isnan(mn) & ~np.isnan(mb)
    d = mb[ok] - mn[ok]
    med_n, med_b = np.nanmedian(mn), np.nanmedian(mb)
    return {
        "paired": bool(paired),
        "n_pairs": int(ok.sum()),
        "median_native": float(med_n),
        "median_blinded": float(med_b),
        "retention": float(med_n / med_b) if med_b and not np.isnan(med_b) else float("nan"),
        "median_dMAE": float(np.median(d)) if d.size else float("nan"),
        "frac_blinded_worse": float(np.mean(d > 0)) if d.size else float("nan"),
        "d": d,
        "mae_native": mn,
        "mae_blinded": mb,
    }


# -----------------------------------------------------------------------------
# Shared plotting: per-cell cycle diagnostics
# -----------------------------------------------------------------------------
def render_cycle_diagnostics(data: dict, cell: str, metric_map: dict,
                             key_prefix: str):
    """Metric picker, stats row, histogram + box plot, validation caption."""
    import matplotlib.pyplot as plt

    present = [k for k in metric_map if k in data
               and np.asarray(data[k]).ndim == 1]
    if not present:
        st.warning("No per-cycle arrays in this NPZ.")
        return
    labels = [metric_map[k] for k in present]
    default_idx = present.index("MAE_test") if "MAE_test" in present else 0
    pick_label = st.selectbox("Metric", options=labels, index=default_idx,
                              key=f"{key_prefix}_metric")
    key = present[labels.index(pick_label)]
    arr = np.asarray(data[key], dtype=float)
    finite = arr[np.isfinite(arr)]

    best_cycle = None
    if "best_cycle" in data:
        try:
            best_cycle = int(np.asarray(data["best_cycle"]))
        except Exception:
            best_cycle = None

    if finite.size == 0:
        st.warning(f"All {pick_label} values are NaN.")
        return

    q1 = float(np.percentile(finite, 25))
    q3 = float(np.percentile(finite, 75))
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Median", f"{np.median(finite):.4f}")
    s2.metric("Mean", f"{np.mean(finite):.4f}")
    s3.metric("Std", f"{np.std(finite, ddof=1):.4f}" if finite.size > 1 else "n/a")
    s4.metric("IQR", f"{q3 - q1:.4f}")
    s5.metric("Cycles (finite)", f"{finite.size}/{arr.size}")

    fig, (ax_h, ax_b) = plt.subplots(
        1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [3, 1]})
    ax_h.hist(finite, bins=min(20, max(5, finite.size // 2)),
              edgecolor="black", alpha=0.75)
    ax_h.axvline(np.median(finite), color="C1", linestyle="--", linewidth=1.5,
                 label=f"median={np.median(finite):.3f}")
    if best_cycle is not None and 0 <= best_cycle < arr.size:
        bv = float(arr[best_cycle])
        if np.isfinite(bv):
            ax_h.axvline(bv, color="C2", linestyle=":", linewidth=1.5,
                         label=f"best cyc {best_cycle+1}={bv:.3f}")
    ax_h.set_xlabel(pick_label, fontweight="bold")
    ax_h.set_ylabel("count", fontweight="bold")
    ax_h.set_title(f"{cell}  —  {pick_label} per-cycle distribution")
    ax_h.legend(fontsize=8)
    ax_b.boxplot(finite, widths=0.6)
    ax_b.set_xticks([])
    ax_b.set_ylabel(pick_label, fontweight="bold")
    ax_b.set_title("box")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    val_bits = []
    for vk in ("MAE_val", "RMSE_val", "R2_val", "Acc_val",
               "n_rejects_val", "n_out_of_range_val"):
        if vk in data:
            try:
                vv = float(np.asarray(data[vk]))
                if np.isfinite(vv):
                    val_bits.append(f"{vk}={vv:.4f}")
            except Exception:
                pass
    if val_bits:
        st.caption("Validation (best-cycle demos):  " + "   ".join(val_bits))


def render_csv_tab(path: Path, key: str):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not read {path.name}: {e}")
        return None
    if "backend" in df.columns:
        n_ok = int(df["backend"].isin(["LOCAL", "REMOTE"]).sum())
        n_bad = int(df["backend"].isin(["FAILED", "SKIPPED"]).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("OK", n_ok)
        c3.metric("Failed / skipped", n_bad)
    _stretch(st.dataframe, df, height=400)
    st.download_button(f"Download {path.name}", data=path.read_bytes(),
                       file_name=path.name, mime="text/csv", key=key)
    st.caption(f"Source: {path}")
    return df


# =============================================================================
# Page
# =============================================================================
def main():
    _init_state()
    _try_reattach()   # adopt an orphaned run, if any, before rendering

    st.title("ICL Sweep Dashboard")
    st.caption("Local launcher + monitor + results viewer for the ICL "
               "sweeps and the blinding study")

    if st.session_state.get("adopted_pid") is not None and _is_running():
        st.info(f"Re-adopted a running job (PID "
                f"{st.session_state.adopted_pid}) from a prior session. "
                f"Live log and Stop are available below.")

    with st.sidebar:
        st.header("Environment")
        config_path = st.text_input("properties.yaml", value=CONFIG_DEFAULT)
        results_root = st.text_input("Results root", value=RESULTS_ROOT_DEFAULT)
        st.caption(f"= `{Path(results_root).resolve()}`")
        python_bin = st.text_input("Python interpreter", value=PYTHON)
        st.divider()
        for script in (DRIVER, ANALYSIS, BLIND_DRIVER, BLIND_ANALYSIS,
                       OOR_EXTRACT):
            mark = "" if Path(script).exists() else "  (not found)"
            st.caption(f"`{script}`{mark}")
        if not Path(config_path).exists():
            st.error(f"Config not found: {config_path}")
        st.divider()
        st.caption("Large models: run `sudo sysctl iogpu.wired_limit_mb=...` "
                   "once per boot before launching (the shell wrappers do "
                   "this; the dashboard cannot).")
        lock = _read_runlock()
        if lock:
            alive = _is_our_driver(lock.get("pid", -1))
            st.caption(f"Runlock PID {lock.get('pid')}: "
                       f"{'alive' if alive else 'stale'}")

    mode = st.radio("Mode", ["Run", "Results"], horizontal=True)
    available_props = load_property_keys(config_path)

    if mode == "Run":
        render_run_mode(config_path, results_root, python_bin,
                        available_props)
    else:
        render_results_mode(results_root)


# =============================================================================
# MODE: RUN
# =============================================================================
def render_run_mode(config_path, results_root, python_bin, available_props):
    phase = st.radio(
        "Phase",
        ["Testing", "Predict", "Forecast", "Analyze",
         "Blinding sweep", "Blinding analysis"],
        horizontal=True)
    is_blind = phase.startswith("Blinding")
    needs_models = phase in ("Testing", "Predict", "Forecast",
                             "Blinding sweep")

    # ---- Models / properties / run id -----------------------------------
    col_left, col_right = st.columns(2)
    chosen_models, chosen_props = [], []
    with col_left:
        if needs_models:
            default_models = (BLIND_DEFAULT_MODELS if is_blind else [1, 2, 7])
            chosen_labels = st.multiselect(
                "Models", options=list(MODEL_LABELS.values()),
                default=[MODEL_LABELS[m] for m in default_models],
                key=f"models_{phase}",
            )
            chosen_models = [k for k, v in MODEL_LABELS.items()
                             if v in chosen_labels]
            if available_props:
                if is_blind:
                    dprops = [p for p in BLIND_DEFAULT_PROPS
                              if p in available_props] or available_props
                else:
                    dprops = available_props
                chosen_props = st.multiselect(
                    "Properties", options=available_props, default=dprops,
                    key=f"props_{phase}",
                    help="Empty => driver default (all in config).",
                )
            else:
                st.warning("No properties parsed; enter manually.")
                chosen_props = st.text_input("Properties (manual)",
                                             key=f"propsman_{phase}").split()

    with col_right:
        run_id = _render_run_id(phase, results_root)

    # ---- Phase options --------------------------------------------------
    st.subheader("Phase options")
    opts = _render_phase_options(phase)

    # Blinding sweep: show the run id that will be used and the pairing
    # check against an existing run folder.
    if phase == "Blinding sweep":
        if not run_id:
            run_id = auto_blind_run_id(opts["transform"], opts["conditions"])
            st.caption(f"Run id (auto): `{run_id}`")
        _render_blind_append_check(results_root, run_id, opts)

    cmd, err = build_command(phase, python_bin, config_path, results_root,
                             chosen_models, chosen_props, run_id, opts)

    st.subheader("Command preview")
    if err:
        st.warning(err)
    else:
        _code_wrapped(shlex.join(cmd))

    # ---- Launch / stop / status ------------------------------------------
    run_col, stop_col, status_col = st.columns([1, 1, 3])
    with run_col:
        if _stretch(st.button, "▶  Launch", type="primary",
                    disabled=_is_running() or err is not None):
            _launch(cmd)
            st.rerun()
    with stop_col:
        if _stretch(st.button, "■  Stop", disabled=not _is_running()):
            _stop()
            time.sleep(1.0)
            st.rerun()
    with status_col:
        if _is_running():
            started = st.session_state.run_started
            if started:
                elapsed = str(datetime.now() - started).split(".")[0]
                st.success(f"RUNNING — started {started:%H:%M:%S}, "
                           f"elapsed {elapsed}")
            else:
                st.success("RUNNING (adopted; start time unknown).")
        elif st.session_state.get("proc") is not None:
            rc = st.session_state.proc.poll()
            if rc == 0:
                st.success("Last run finished (exit 0).")
            else:
                st.error(f"Last run exited with code {rc}.")
            _clear_runlock()

    st.subheader("Live log")
    auto = st.checkbox("Auto-refresh (2 s) while running", value=True)
    st.text_area("driver output", value=_read_log(), height=420,
                 label_visibility="collapsed")
    if st.session_state.get("log_path"):
        st.caption(f"Log file: {st.session_state.log_path}")
    if _is_running() and auto:
        time.sleep(2.0)
        st.rerun()


def _render_run_id(phase, results_root):
    """Run-id widget per phase. Analysis phases pick an existing run."""
    if phase == "Analyze":
        runs = list_runs(results_root)
        if runs:
            return st.selectbox("Run to analyze", options=runs,
                                key="rid_analyze",
                                help="Newest first. Testing runs only "
                                     "(needs summary.csv).")
        _no_runs_note(Path(results_root) / "runs")
        return st.text_input("Run id (none found; type one)",
                             key="rid_analyze_txt").strip()
    if phase == "Blinding analysis":
        runs = list_blinding_runs(results_root)
        if runs:
            return st.selectbox("Blinding run to analyze", options=runs,
                                key="rid_banalyze")
        _no_runs_note(Path(results_root) / "blinding_runs")
        return st.text_input("Blinding run id (none found; type one)",
                             key="rid_banalyze_txt").strip()
    if phase == "Blinding sweep":
        existing = list_blinding_runs(results_root)
        rid = st.text_input(
            "Run id",
            key="rid_bsweep",
            help="Blank => blind_<transform>_<conditions>. Use an EXISTING "
                 "run id to append a condition (for example agnostic) to it.")
        if existing:
            st.caption("Existing blinding runs: "
                       + ", ".join(f"`{r}`" for r in existing[:8])
                       + (" ..." if len(existing) > 8 else ""))
        return rid.strip()
    return st.text_input(
        "Run id", key=f"rid_{phase}",
        help="Testing: names the run. Predict/Forecast: optional.").strip()


def _no_runs_note(folder: Path):
    """Say exactly where we looked, so a wrong Results root is obvious."""
    st.caption(f"No runs found in `{folder.resolve()}`. If your results "
               f"are elsewhere, set **Results root** in the sidebar to the "
               f"folder that contains `{folder.name}/`.")


def _render_phase_options(phase):
    """Return a dict of phase options."""
    o = {"dry_run": False, "notify": False}
    c = st.columns(4)

    if phase == "Testing":
        with c[0]:
            o["m_cycles"] = st.number_input("M-cycles", 1, 500, 50)
        with c[1]:
            o["n_shots"] = st.number_input("N-shots", 1, 256, 32)
        with c[2]:
            o["temperature"] = st.text_input("Temperature", value="",
                                             help="Blank => default 0.9.")
        with c[3]:
            o["permutation"] = st.checkbox("Permutation control")
            o["dry_run"] = st.checkbox("Dry run")
            o["notify"] = st.checkbox("Notify (macOS)")

    elif phase == "Predict":
        with c[0]:
            o["temperature"] = st.text_input("Temperature", value="",
                                             help="Blank => default 0.9.")
        with c[1]:
            o["dry_run"] = st.checkbox("Dry run")
        with c[2]:
            o["notify"] = st.checkbox("Notify (macOS)")

    elif phase == "Forecast":
        with c[0]:
            o["n_exp"] = st.number_input("N-exp (replicates)", 1, 200, 20)
        with c[1]:
            o["temperatures"] = st.text_input("Temperatures (space-sep)",
                                              value="0.3 0.7 1.0")
        with c[2]:
            o["dry_run"] = st.checkbox("Dry run")
        with c[3]:
            o["notify"] = st.checkbox("Notify (macOS)")
        st.info("Forecast sweeps --temperatures; ignores single "
                "--temperature. T=0 gives zero uncertainty.")

    elif phase == "Analyze":
        st.info("Runs analysis.py on an existing Testing run.")

    elif phase == "Blinding sweep":
        with c[0]:
            o["conditions"] = st.multiselect(
                "Conditions", BLIND_CONDITIONS,
                default=["native", "generic"],
                help="All share the same seed base, so they are "
                     "cycle-paired.")
        with c[1]:
            o["transform"] = st.selectbox(
                "Label transform", BLIND_TRANSFORMS, index=0,
                help="zscore: whole-real-line inverse, never out of range "
                     "(recommended). linear / tanh: bounded windows.")
        with c[2]:
            o["m_cycles"] = st.number_input("M-cycles", 1, 500, 50,
                                            key="b_mc")
            o["n_shots"] = st.number_input("N-shots", 1, 256, 32,
                                           key="b_ns")
        with c[3]:
            o["seed_base"] = st.number_input(
                "Seed base", 0, 10**9, 0, key="b_seed",
                help="Must be the same for every arm of a run.")
            o["temperature"] = st.text_input(
                "Temperature", value="", key="b_temp",
                help="Blank => default 0.9.")

        with st.expander("Transform, label format and recoder settings"):
            e = st.columns(4)
            with e[0]:
                o["std_center"] = st.number_input("zscore center", value=500.0)
                o["std_scale"] = st.number_input(
                    "zscore scale", value=100.0,
                    help="Negative also reverses the ordering.")
            with e[1]:
                o["out_max"] = st.number_input("linear out-max", value=100.0)
            with e[2]:
                o["tanh_center"] = st.number_input("tanh center", value=50.0)
                o["tanh_amp"] = st.number_input("tanh amplitude", value=40.0)
            with e[3]:
                o["value_fmt"] = st.text_input("Label format", value="{:.3f}")
                o["recoder_seed"] = st.number_input(
                    "Recoder seed (agnostic)", 0, 2**31 - 1, 20260101)

        f = st.columns(4)
        with f[0]:
            o["resume"] = st.checkbox(
                "Resume", help="Skip cells whose cycles_blinded.npz exists.")
        with f[1]:
            o["allow_mismatch"] = st.checkbox(
                "Allow mismatch",
                help="Append even if settings differ from the run's "
                     "manifest (breaks cycle pairing).")
        with f[2]:
            o["dry_run"] = st.checkbox("Dry run", key="b_dry")
        with f[3]:
            o["notify"] = st.checkbox("Notify (macOS)", key="b_notify")

    elif phase == "Blinding analysis":
        with c[0]:
            o["n_boot"] = st.number_input("Bootstrap resamples", 100,
                                          10**6, 10000, step=1000)
        with c[1]:
            o["n_perm"] = st.number_input("Permutation draws", 100,
                                          10**6, 20000, step=1000)
        with c[2]:
            o["stat_seed"] = st.number_input("Statistics seed", 0, 10**9, 0)
            o["alpha"] = st.number_input("FDR alpha", 0.001, 0.5, 0.05,
                                         step=0.01, format="%.3f")
        with c[3]:
            o["retention_tol"] = st.number_input(
                "Retention tolerance", 0.5, 0.999, 0.9, step=0.01,
                help="Band [tol, 1/tol] counted as 'retained'.")
            o["oor"] = st.checkbox("Also run out-of-range summary",
                                   value=True)
    return o


def _blind_settings_from_opts(o):
    """Settings dict comparable to the driver's manifest PAIRING_KEYS."""
    t = (o.get("temperature") or "").strip()
    try:
        temp = float(t) if t else None
    except ValueError:
        temp = t   # left as text; build_command reports the error
    return {
        "seed_base": int(o["seed_base"]),
        "M_cycles": int(o["m_cycles"]),
        "N_shots": int(o["n_shots"]),
        "transform": o["transform"],
        "value_fmt": o["value_fmt"],
        "out_max": float(o["out_max"]),
        "tanh_center": float(o["tanh_center"]),
        "tanh_amp": float(o["tanh_amp"]),
        "std_center": float(o["std_center"]),
        "std_scale": float(o["std_scale"]),
        "temperature": temp,
    }


def _render_blind_append_check(results_root, run_id, o):
    """If the run folder exists, preview the driver's pairing guard."""
    run_dir = Path(results_root) / "blinding_runs" / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        st.warning(f"`{run_id}` exists but its manifest.json is unreadable.")
        return
    have = sorted({c for (_, _, c) in blinding_cells(run_dir)})
    st.info(f"`{run_id}` already exists (conditions with results: "
            f"{', '.join(have) or 'none'}). Launching will APPEND to it.")
    mism = blinding_pairing_mismatches(manifest,
                                       _blind_settings_from_opts(o))
    if mism:
        rows = "\n".join(f"- `{k}`: run has `{a!r}`, you set `{b!r}`"
                         for k, a, b in mism)
        st.warning("These settings differ from the existing run, so the "
                   "new arm would NOT be cycle-paired. The driver will "
                   "refuse unless 'Allow mismatch' is ticked.\n\n" + rows)
    else:
        st.success("Pairing settings match the existing run.")


def build_command(phase, python_bin, config_path, results_root,
                  chosen_models, chosen_props, run_id, o):
    """Return (cmd_list, error_message_or_None)."""
    if phase == "Analyze":
        if not run_id:
            return None, "Analyze requires a run id."
        if not Path(ANALYSIS).exists():
            return None, f"Analysis script not found: {ANALYSIS}"
        return ([python_bin, ANALYSIS, "--results-root", results_root,
                 "--run-id", run_id], None)

    if phase == "Blinding analysis":
        if not run_id:
            return None, "Blinding analysis requires a run id."
        if not Path(BLIND_ANALYSIS).exists():
            return None, f"Script not found: {BLIND_ANALYSIS}"
        a = [python_bin, BLIND_ANALYSIS, "--results-root", results_root,
             "--run-id", run_id,
             "--n-boot", str(int(o["n_boot"])),
             "--n-perm", str(int(o["n_perm"])),
             "--seed", str(int(o["stat_seed"])),
             "--alpha", str(o["alpha"]),
             "--retention-tol", str(o["retention_tol"])]
        if not o.get("oor"):
            return a, None
        if not Path(OOR_EXTRACT).exists():
            return None, f"Script not found: {OOR_EXTRACT}"
        b = [python_bin, OOR_EXTRACT, "--results-root", results_root,
             "--run-id", run_id]
        # Two scripts in sequence, one process group (Stop ends both).
        return (["/bin/sh", "-c", f"{shlex.join(a)} && {shlex.join(b)}"],
                None)

    if not chosen_models:
        return None, "Select at least one model."

    t = (o.get("temperature") or "").strip()
    if t:
        try:
            float(t)
        except ValueError:
            return None, f"Temperature must be a number; got {t!r}."

    if phase == "Blinding sweep":
        if not Path(BLIND_DRIVER).exists():
            return None, f"Script not found: {BLIND_DRIVER}"
        if not o["conditions"]:
            return None, "Select at least one condition."
        cmd = [python_bin, BLIND_DRIVER, "--config", config_path,
               "--models", *[str(m) for m in chosen_models],
               "--results-root", results_root, "--run-id", run_id,
               "--conditions", *o["conditions"],
               "--transform", o["transform"],
               "--M-cycles", str(int(o["m_cycles"])),
               "--N-shots", str(int(o["n_shots"])),
               "--seed-base", str(int(o["seed_base"])),
               "--recoder-seed", str(int(o["recoder_seed"])),
               "--std-center", str(o["std_center"]),
               "--std-scale", str(o["std_scale"]),
               "--out-max", str(o["out_max"]),
               "--tanh-center", str(o["tanh_center"]),
               "--tanh-amp", str(o["tanh_amp"]),
               "--value-fmt", o["value_fmt"]]
        if chosen_props:
            cmd += ["--properties", *chosen_props]
        if t:
            cmd += ["--temperature", t]
        if o.get("resume"):
            cmd += ["--resume"]
        if o.get("allow_mismatch"):
            cmd += ["--allow-mismatch"]
        if o.get("dry_run"):
            cmd += ["--dry-run"]
        if o.get("notify"):
            cmd += ["--notify"]
        return cmd, None

    # ICL phases (batch_driver.py)
    if not Path(DRIVER).exists():
        return None, f"Driver not found: {DRIVER}"
    cmd = [python_bin, DRIVER, "--config", config_path,
           "--models", *[str(m) for m in chosen_models],
           "--results-root", results_root]
    if chosen_props:
        cmd += ["--properties", *chosen_props]
    if run_id:
        cmd += ["--run-id", run_id]
    if phase == "Testing":
        cmd += ["--M-cycles", str(int(o["m_cycles"])),
                "--N-shots", str(int(o["n_shots"]))]
        if o.get("permutation"):
            cmd += ["--permutation-control"]
        if t:
            cmd += ["--temperature", t]
    elif phase == "Predict":
        cmd += ["--predict"]
        if t:
            cmd += ["--temperature", t]
    elif phase == "Forecast":
        temps = (o.get("temperatures") or "").split()
        for x in temps:
            try:
                float(x)
            except ValueError:
                return None, f"Forecast temperature {x!r} is not a number."
        cmd += ["--forecast", "--N-exp", str(int(o["n_exp"]))]
        if temps:
            cmd += ["--temperatures", *temps]
    if o.get("dry_run"):
        cmd += ["--dry-run"]
    if o.get("notify"):
        cmd += ["--notify"]
    return cmd, None


# =============================================================================
# MODE: RESULTS
# =============================================================================
def render_results_mode(results_root):
    st.subheader("Results viewer")
    kind = st.radio("Run type", ["ICL sweeps", "Blinding runs"],
                    horizontal=True)
    if kind == "ICL sweeps":
        render_icl_results(results_root)
    else:
        render_blinding_results(results_root)
    if st.button("↻ Rescan runs"):
        st.rerun()


def render_icl_results(results_root):
    runs = list_runs(results_root)
    if not runs:
        st.info(f"No runs found in {(Path(results_root) / 'runs').resolve()}. "
                f"Launch a Testing sweep first, or set the Results root in "
                f"the sidebar.")
        return
    sel_run = st.selectbox("Run", options=runs,
                           help="Newest first (by folder mtime).")
    markers = run_markers(results_root, sel_run)
    st.caption("Completion markers: " + ", ".join(markers) if markers
               else "No RUN_COMPLETE marker yet (in progress or stopped).")

    summaries = find_summaries(results_root, sel_run)
    if not summaries:
        st.warning("No summary CSVs in this run folder yet.")
    else:
        tabs = st.tabs(list(summaries.keys()))
        for tab, (label, path) in zip(tabs, summaries.items()):
            with tab:
                render_csv_tab(path, key=f"dl_icl_{label}")

    st.divider()
    st.markdown("#### Per-cell cycle diagnostics")
    run_dir = Path(results_root) / "runs" / sel_run
    cells = list_cells_with_npz(run_dir, "cycles.npz")
    if not cells:
        st.caption("No cycles.npz found in this run "
                   "(Testing phase produces them).")
        return
    cell = st.selectbox("Cell", options=cells,
                        help="One (model x property) cell; "
                             "_perm = permutation control.")
    data = load_npz(run_dir / cell / "cycles.npz")
    if data is None:
        st.warning("Could not read cycles.npz for this cell.")
    else:
        render_cycle_diagnostics(data, cell, CYCLE_METRICS, "icl")


def render_blinding_results(results_root):
    import matplotlib.pyplot as plt

    runs = list_blinding_runs(results_root)
    if not runs:
        st.info(f"No blinding runs in "
                f"{(Path(results_root) / 'blinding_runs').resolve()}. Launch "
                f"a Blinding sweep first, or set the Results root in the "
                f"sidebar to the folder that contains blinding_runs/.")
        return
    sel_run = st.selectbox("Blinding run", options=runs,
                           help="Newest first (by folder mtime).")
    run_dir = Path(results_root) / "blinding_runs" / sel_run
    ana_dir = Path(results_root) / "blinding_analysis" / sel_run

    # ---- Settings and progress -------------------------------------------
    manifests = load_blinding_manifests(run_dir)
    cells = blinding_cells(run_dir)
    expected = blinding_expected_cells(manifests)
    if manifests:
        m0 = manifests[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Cells with results",
                  f"{len(cells)}/{len(expected) or '?'}")
        c2.metric("Transform", str(m0.get("transform", "?")))
        c3.metric("M-cycles", str(m0.get("M_cycles", "?")))
        c4.metric("N-shots", str(m0.get("N_shots", "?")))
        c5.metric("Seed base", str(m0.get("seed_base", "?")))
        missing = sorted(expected - set(cells))
        if missing:
            st.caption(f"{len(missing)} planned cell(s) without results "
                       f"(running, failed, or interrupted; Resume continues): "
                       + ", ".join(f"{s}_{p}_{c}" for s, p, c in missing[:6])
                       + (" ..." if len(missing) > 6 else ""))
        with st.expander(f"Manifests ({len(manifests)})"):
            for m in manifests:
                st.markdown(f"**{m.get('_file')}**  started "
                            f"{m.get('started_at', '?')}, conditions "
                            f"{m.get('conditions')}")
                st.json({k: v for k, v in m.items() if k != "_file"},
                        expanded=False)
    else:
        st.warning("No manifest.json in this run folder.")

    # ---- Tables and figure -------------------------------------------------
    files = {}
    if (run_dir / "summary_blinded.csv").exists():
        files["Summary"] = run_dir / "summary_blinded.csv"
    if (ana_dir / "contrast_table.csv").exists():
        files["Contrasts"] = ana_dir / "contrast_table.csv"
    if (ana_dir / "main_table_native_vs_generic.csv").exists():
        files["Main table"] = ana_dir / "main_table_native_vs_generic.csv"
    if (run_dir / "oor_summary.csv").exists():
        files["Out-of-range"] = run_dir / "oor_summary.csv"
    fig_path = ana_dir / "blinding_condition_panels.png"

    tab_names = list(files) + (["Figure"] if fig_path.exists() else [])
    if not tab_names:
        st.warning("No summary or analysis files yet.")
    else:
        tabs = st.tabs(tab_names)
        for tab, name in zip(tabs, tab_names):
            with tab:
                if name == "Figure":
                    _stretch(st.image, str(fig_path))
                else:
                    render_csv_tab(files[name], key=f"dl_bl_{name}")
    if not (ana_dir / "contrast_table.csv").exists():
        st.caption("No statistics yet: run the 'Blinding analysis' phase "
                   "for this run to create the contrast table and figure.")

    if not cells:
        return

    # ---- Paired native vs blinded view -------------------------------------
    st.divider()
    st.markdown("#### Paired comparison (per cycle)")
    pairs = sorted({(s, p) for (s, p, _) in cells})
    labels = [f"{s} x {p}" for s, p in pairs]
    pick = st.selectbox("Model x property", options=labels,
                        key="bl_pair_pick")
    slug, prop = pairs[labels.index(pick)]
    present = [c for c in BLIND_CONDITIONS if (slug, prop, c) in cells]
    data = {c: load_npz(cells[(slug, prop, c)] / "cycles_blinded.npz")
            for c in present}
    data = {c: d for c, d in data.items() if d is not None}

    if "native" not in data or len(data) < 2:
        st.caption("Needs the native arm and at least one blinded arm for "
                   "this model and property.")
    else:
        blinded = [c for c in ("generic", "agnostic") if c in data]
        stats = {c: paired_view_stats(data["native"], data[c])
                 for c in blinded}
        cols = st.columns(len(blinded))
        for col, c in zip(cols, blinded):
            s = stats[c]
            with col:
                st.markdown(f"**native vs {c}**"
                            + ("" if s["paired"] else "  (NOT paired)"))
                m1, m2, m3 = st.columns(3)
                m1.metric("Retention R", f"{s['retention']:.3f}")
                m2.metric("median ΔMAE", f"{s['median_dMAE']:+.4f}")
                m3.metric("cycles blinded worse",
                          f"{100 * s['frac_blinded_worse']:.0f}%")
                st.caption(f"median MAE native {s['median_native']:.4f}, "
                           f"{c} {s['median_blinded']:.4f}; "
                           f"{s['n_pairs']} paired cycles")
        if any(not stats[c]["paired"] for c in blinded):
            st.warning("Seed vectors differ between arms, so cycles are "
                       "not paired; per-cycle differences are not "
                       "meaningful.")

        # Distinct marker + line style per condition so arms that overlap
        # (identical per-cycle MAEs) stay visible.
        style = {
            "native":   dict(color="C0", marker="o", linestyle="-",
                             markersize=7, markerfacecolor="none",
                             linewidth=1.6, zorder=3),
            "generic":  dict(color="C1", marker="s", linestyle="--",
                             markersize=4, linewidth=1.2, zorder=2),
            "agnostic": dict(color="C2", marker="^", linestyle=":",
                             markersize=4, linewidth=1.2, zorder=2),
        }
        fig, (ax_l, ax_d) = plt.subplots(
            1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [3, 2]})
        for c in present:
            if c in data:
                y = np.asarray(data[c]["MAE_test"], dtype=float)
                ax_l.plot(np.arange(1, len(y) + 1), y, label=c, **style[c])
        ax_l.set_xlabel("Cycle", fontweight="bold")
        ax_l.set_ylabel("Test MAE (native units)", fontweight="bold")
        ax_l.set_title(f"{slug} x {prop}")
        ax_l.grid(True, linestyle="--", alpha=0.4)
        ax_l.legend(fontsize=8)

        # Strip plot of the paired differences: one dot per cycle, one row
        # per blinded condition, median as a vertical bar.
        jit = np.random.default_rng(0)
        for row, c in enumerate(blinded):
            d = stats[c]["d"]
            if not d.size:
                continue
            yy = row + jit.uniform(-0.15, 0.15, d.size)
            ax_d.scatter(d, yy, s=16, alpha=0.7,
                         color=style[c]["color"], marker=style[c]["marker"])
            ax_d.plot([np.median(d)] * 2, [row - 0.3, row + 0.3],
                      color="black", linewidth=2.2)
        ax_d.axvline(0.0, color="grey", linewidth=1, linestyle="--")
        ax_d.set_yticks(range(len(blinded)))
        ax_d.set_yticklabels(blinded, fontweight="bold")
        ax_d.set_ylim(-0.6, len(blinded) - 0.4)
        ax_d.set_xlabel("ΔMAE = blinded − native",
                        fontweight="bold")
        ax_d.set_ylabel("Condition", fontweight="bold")
        ax_d.set_title("Paired per-cycle differences (bar = median)")
        ax_d.grid(True, axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Quick look only. Use the 'Blinding analysis' phase for "
                   "bootstrap CIs, Wilcoxon / permutation tests and "
                   "BH-FDR q-values.")

    # ---- Per-cell diagnostics ----------------------------------------------
    st.divider()
    st.markdown("#### Per-cell cycle diagnostics")
    names = sorted(p.name for p in cells.values())
    cell = st.selectbox("Cell", options=names, key="bl_cell_pick")
    d = load_npz(run_dir / cell / "cycles_blinded.npz")
    if d is None:
        st.warning("Could not read cycles_blinded.npz for this cell.")
    else:
        spec_path = run_dir / cell / "blinding_spec.json"
        if spec_path.exists():
            with st.expander("blinding_spec.json"):
                try:
                    st.json(json.loads(spec_path.read_text()))
                except Exception:
                    st.caption("unreadable")
        render_cycle_diagnostics(d, cell, BLIND_CYCLE_METRICS, "blind")


main()
