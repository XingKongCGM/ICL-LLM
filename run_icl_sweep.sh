#!/usr/bin/env bash
# =============================================================================
# run_icl_sweep.sh
# =============================================================================
# Convenience wrapper around batch_driver.py (and analysis.py) for the
# LOCAL-only ICL evaluation framework.
#
# Models are local MLX, choices 1-7 (contiguous; the remote backend was
# removed). Temperature control:
#   * Testing / Predict : single --temperature (env: TEMPERATURE)
#   * Forecast          : --temperatures sweep  (env: TEMPERATURES)
# Omitting the temperature reproduces the published runs (module default 0.9).
#
# Any tokens after the subcommand are forwarded verbatim to batch_driver.py,
# so you can add or override any driver flag without editing this script,
# e.g.:   ./run_icl_sweep.sh testing --M-cycles 10 --permutation-control
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# GPU wired-memory limit (macOS / Apple Silicon)
# -----------------------------------------------------------------------------
# Large models (e.g. 70B Q4, ~40 GB) can exceed macOS's default GPU wired
# memory limit. WIRED_LIMIT_MB raises it via `sudo sysctl` (asks for your
# password; resets on reboot). 61440 MB (60 GB) suits a 64 GB machine;
# leave headroom for macOS on yours. Set WIRED_LIMIT_MB=0 (or empty) to
# skip this step entirely.
WIRED_LIMIT_MB="${WIRED_LIMIT_MB-61440}"

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
PYTHON="${PYTHON:-python3}"
DRIVER="${DRIVER:-batch_driver.py}"
ANALYSIS="${ANALYSIS:-analysis.py}"
CONFIG="${CONFIG:-properties.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-./RESULTS}"

# Default sweep scope
MODELS="${MODELS:-1 2 7}"
PROPERTIES="${PROPERTIES:-}"          # empty => all properties in the config
M_CYCLES="${M_CYCLES:-50}"
N_SHOTS="${N_SHOTS:-32}"

# Temperature knobs
TEMPERATURE="${TEMPERATURE:-}"        # Testing/Predict; empty => module default 0.9
TEMPERATURES="${TEMPERATURES:-0.3 0.7 1.0}"   # Forecast sweep
N_EXP="${N_EXP:-20}"                  # Forecast replicates per molecule per T

# Run id (empty => driver assigns a timestamped id)
RUN_ID="${RUN_ID:-}"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_die() { echo "ERROR: $*" >&2; exit 1; }

_check_python_deps() {
    # Fail fast (before any sudo prompt) if required packages are missing.
    local missing
    missing="$("$PYTHON" - <<'PY'
import importlib.util as u
need = ["numpy", "pandas", "sklearn", "matplotlib", "mlx_lm",
        "huggingface_hub"]
print(" ".join(m for m in need if u.find_spec(m) is None))
PY
)"
    if [[ -n "$missing" ]]; then
        _die "missing Python packages for $PYTHON: $missing
       Install with:  $PYTHON -m pip install -r requirements.txt"
    fi
}

_set_wired_limit() {
    # Runs at most once per invocation, and only for subcommands that
    # actually load models.
    [[ -n "${_WIRED_DONE:-}" ]] && return 0
    _WIRED_DONE=1
    if [[ -z "$WIRED_LIMIT_MB" || "$WIRED_LIMIT_MB" == "0" ]]; then
        return 0
    fi
    if [[ "$(uname -s)" != "Darwin" ]]; then
        echo "[wired] not macOS; skipping iogpu.wired_limit_mb"
        return 0
    fi
    echo "[wired] setting iogpu.wired_limit_mb=$WIRED_LIMIT_MB (sudo)"
    sudo sysctl "iogpu.wired_limit_mb=$WIRED_LIMIT_MB" \
        || echo "[wired] WARNING: sysctl failed; continuing with the current limit" >&2
}

_preflight() {
    [[ -f "$DRIVER" ]]   || _die "driver not found: $DRIVER (set DRIVER=...)"
    [[ -f "$CONFIG" ]]   || _die "config not found: $CONFIG (set CONFIG=...)"
    command -v "$PYTHON" >/dev/null 2>&1 \
        || _die "python interpreter not found: $PYTHON (set PYTHON=...)"
    mkdir -p "$RESULTS_ROOT"
}

# Build the shared leading args common to every phase.
_base_args() {
    local a=(--config "$CONFIG"
             --models $MODELS
             --results-root "$RESULTS_ROOT")
    [[ -n "$PROPERTIES" ]] && a+=(--properties $PROPERTIES)
    [[ -n "$RUN_ID"     ]] && a+=(--run-id "$RUN_ID")
    printf '%s\n' "${a[@]}"
}

# Read the base args back into an array (whitespace-safe line split).
_read_base() {
    BASE=()
    while IFS= read -r line; do BASE+=("$line"); done < <(_base_args)
}

# -----------------------------------------------------------------------------
# Subcommands
# -----------------------------------------------------------------------------

cmd_smoke() {
    # Fast end-to-end check: one model, one property, Mc=2.
    _preflight
    _check_python_deps
    _set_wired_limit
    local model="${SMOKE_MODEL:-7}"
    local prop="${SMOKE_PROP:-PKA}"
    echo "[smoke] one cell: model=$model property=$prop M_cycles=2"
    local args=(--config "$CONFIG" --models "$model"
                --properties "$prop" --results-root "$RESULTS_ROOT"
                --M-cycles 2 --run-id "smoke_test")
    [[ -n "$TEMPERATURE" ]] && args+=(--temperature "$TEMPERATURE")
    "$PYTHON" "$DRIVER" "${args[@]}" "$@"
    echo
    echo "[smoke] done. Inspect:"
    echo "        $RESULTS_ROOT/runs/smoke_test/summary.csv"
}

cmd_dry() {
    # Print the Testing execution plan without running anything.
    _preflight
    _read_base
    local args=("${BASE[@]}" --M-cycles "$M_CYCLES" --N-shots "$N_SHOTS" --dry-run)
    [[ -n "$TEMPERATURE" ]] && args+=(--temperature "$TEMPERATURE")
    "$PYTHON" "$DRIVER" "${args[@]}" "$@"
}

cmd_testing() {
    _preflight
    _check_python_deps
    _set_wired_limit
    _read_base
    local args=("${BASE[@]}" --M-cycles "$M_CYCLES" --N-shots "$N_SHOTS")
    [[ -n "$TEMPERATURE" ]] && args+=(--temperature "$TEMPERATURE")
    "$PYTHON" "$DRIVER" "${args[@]}" "$@"
}

cmd_permute() {
    # Testing sweep plus the permutation-null control.
    _preflight
    _check_python_deps
    _set_wired_limit
    _read_base
    local args=("${BASE[@]}" --M-cycles "$M_CYCLES" --N-shots "$N_SHOTS"
                --permutation-control)
    [[ -n "$TEMPERATURE" ]] && args+=(--temperature "$TEMPERATURE")
    "$PYTHON" "$DRIVER" "${args[@]}" "$@"
}

cmd_predict() {
    # Prediction pass; reuses optimal_samples from a prior Testing run.
    _preflight
    _check_python_deps
    _set_wired_limit
    _read_base
    local args=("${BASE[@]}" --predict)
    [[ -n "$TEMPERATURE" ]] && args+=(--temperature "$TEMPERATURE")
    "$PYTHON" "$DRIVER" "${args[@]}" "$@"
}

cmd_forecast() {
    # SMILES-only replicate forecasting over a temperature sweep.
    _preflight
    _check_python_deps
    _set_wired_limit
    _read_base
    local args=("${BASE[@]}" --forecast --N-exp "$N_EXP"
                --temperatures $TEMPERATURES)
    "$PYTHON" "$DRIVER" "${args[@]}" "$@"
}

cmd_analyze() {
    # Run analysis.py on a finished run. Requires RUN_ID.
    _preflight
    [[ -n "$RUN_ID" ]] || _die "analyze requires RUN_ID=... (the run to analyze)"
    [[ -f "$ANALYSIS" ]] || _die "analysis script not found: $ANALYSIS"
    echo "[analyze] run_id=$RUN_ID"
    "$PYTHON" "$ANALYSIS" \
        --results-root "$RESULTS_ROOT" --run-id "$RUN_ID" "$@"
}

cmd_full() {
    # Testing then analysis, sharing one run id.
    _preflight
    if [[ -z "$RUN_ID" ]]; then
        RUN_ID="full_$(date +%Y%m%d_%H%M%S)"
        echo "[full] no RUN_ID set; using $RUN_ID"
    fi
    cmd_testing "$@"
    RUN_ID="$RUN_ID" cmd_analyze
}

cmd_help() {
    cat <<EOF
run_icl_sweep.sh  --  wrapper for batch_driver.py (LOCAL MLX, models 1-7)

USAGE
  ./run_icl_sweep.sh <subcommand> [extra driver args...]

SUBCOMMANDS
  smoke       One cell (model 7, PKA, Mc=2) end-to-end sanity check
  dry         Print the Testing execution plan (--dry-run), no execution
  testing     Full Testing sweep
  permute     Testing sweep + permutation-null control
  predict     Prediction pass (reuses optimal_samples from a prior Testing run)
  forecast    SMILES-only replicate forecasting over a temperature sweep
  analyze     Run analysis.py on a finished run (requires RUN_ID)
  full        testing -> analyze, sharing one run id
  help        This message

ENVIRONMENT VARIABLES (all optional)
  PYTHON        Python interpreter           (default: python3)
  DRIVER        Driver script path           (default: batch_driver.py)
  ANALYSIS      Analysis script path         (default: analysis.py)
  CONFIG        Properties YAML              (default: properties.yaml)
  RESULTS_ROOT  Output root                  (default: ./RESULTS)
  MODELS        Model choices, space-sep     (default: "1 2 7")
  PROPERTIES    Property keys, space-sep     (default: all in config)
  M_CYCLES      Testing cycles per cell      (default: 50)
  N_SHOTS       In-context demos per query   (default: 32)
  TEMPERATURE   Testing/Predict temperature  (default: empty => module 0.9)
  TEMPERATURES  Forecast sweep               (default: "0.3 0.7 1.0")
  N_EXP         Forecast replicates per mol  (default: 20)
  RUN_ID        Run id                       (default: driver timestamp)
  WIRED_LIMIT_MB  GPU wired-memory limit, MB   (default: 61440; 0 => skip sudo)

EXAMPLES
  # Sanity check first:
  ./run_icl_sweep.sh smoke

  # See the plan without running:
  ./run_icl_sweep.sh dry

  # Full testing, default models (1 2 7), all properties, T=0.9 (default):
  ./run_icl_sweep.sh testing

  # Testing at a chosen temperature:
  TEMPERATURE=0.3 ./run_icl_sweep.sh testing
  # ...or inline (forwarded to the driver):
  ./run_icl_sweep.sh testing --temperature 0.3

  # Narrow the scope:
  MODELS="2 7" PROPERTIES="LOGKOW PKA" ./run_icl_sweep.sh testing

  # Prediction reusing a prior run's optimal samples:
  ./run_icl_sweep.sh predict --run-id my_testing_run

  # Forecast, 30 replicates, custom T sweep:
  N_EXP=30 TEMPERATURES="0.5 1.0" ./run_icl_sweep.sh forecast \\
      --models 7 --properties PKA

  # Testing + analysis in one shot with a named run:
  RUN_ID=paper_v1 ./run_icl_sweep.sh full
EOF
}

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
main() {
    local sub="${1:-help}"
    shift || true
    case "$sub" in
        smoke)    cmd_smoke    "$@" ;;
        dry)      cmd_dry      "$@" ;;
        testing)  cmd_testing  "$@" ;;
        permute)  cmd_permute  "$@" ;;
        predict)  cmd_predict  "$@" ;;
        forecast) cmd_forecast "$@" ;;
        analyze)  cmd_analyze  "$@" ;;
        full)     cmd_full     "$@" ;;
        help|-h|--help) cmd_help ;;
        *) echo "Unknown subcommand: $sub" >&2; echo; cmd_help; exit 1 ;;
    esac
}

main "$@"
