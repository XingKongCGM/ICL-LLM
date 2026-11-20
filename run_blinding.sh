#!/usr/bin/env bash
# =============================================================================
# run_blinding.sh
# =============================================================================
# Convenience wrapper around blinding_driver.py, blinding_analysis.py and
# oor_extract.py for the controlled information-access (blinding) study.
#
# Default study design: two local models (2 = Llama-3.3-70B dense,
# 7 = Llama-4-Scout-17B-16E MoE), four properties (LOGKOW LOGS PKA LOGCMC),
# 50 cycle-paired replicates, z-score label map.
#
# Any tokens after the subcommand are forwarded verbatim to the Python
# script, e.g.:  ./run_blinding.sh native-generic --resume
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-properties.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-./RESULTS}"

MODELS="${MODELS:-2 7}"
PROPERTIES="${PROPERTIES:-LOGKOW LOGS PKA LOGCMC}"
M_CYCLES="${M_CYCLES:-50}"
N_SHOTS="${N_SHOTS:-32}"
SEED_BASE="${SEED_BASE:-0}"            # MUST be identical across arms
TRANSFORM="${TRANSFORM:-zscore}"       # zscore | linear | tanh
RECODER_SEED="${RECODER_SEED:-20260101}"
TEMPERATURE="${TEMPERATURE:-}"         # empty => core default 0.9
RUN_ID="${RUN_ID:-}"

N_BOOT="${N_BOOT:-10000}"
N_PERM="${N_PERM:-20000}"
STAT_SEED="${STAT_SEED:-0}"

# GPU wired-memory limit (see run_icl_sweep.sh). 0 or empty => skip sudo.
WIRED_LIMIT_MB="${WIRED_LIMIT_MB-61440}"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_die() { echo "ERROR: $*" >&2; exit 1; }

_preflight() {
    for f in blinding_driver.py blinding.py rchat_blinding_hooks.py \
             batch_driver.py RCHAT_PChem_Properties.py; do
        [[ -f "$f" ]] || _die "$f not found; run from the repository folder"
    done
    [[ -f "$CONFIG" ]] || _die "config not found: $CONFIG (set CONFIG=...)"
    command -v "$PYTHON" >/dev/null 2>&1 \
        || _die "python interpreter not found: $PYTHON (set PYTHON=...)"
    mkdir -p "$RESULTS_ROOT"
}

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

_require_run_id() {
    [[ -n "$RUN_ID" ]] || _die "$1 requires RUN_ID=... (the blinding run folder name)"
}

# Run the sweep driver with the shared arguments.
#   $1 = conditions (space-separated), $2 = default run id; rest forwarded.
_sweep() {
    local conds="$1" default_id="$2"; shift 2
    local rid="${RUN_ID:-$default_id}"
    local args=(--config "$CONFIG"
                --models $MODELS
                --properties $PROPERTIES
                --conditions $conds
                --transform "$TRANSFORM"
                --M-cycles "$M_CYCLES"
                --N-shots "$N_SHOTS"
                --seed-base "$SEED_BASE"
                --recoder-seed "$RECODER_SEED"
                --results-root "$RESULTS_ROOT"
                --run-id "$rid")
    [[ -n "$TEMPERATURE" ]] && args+=(--temperature "$TEMPERATURE")
    "$PYTHON" blinding_driver.py "${args[@]}" "$@"
}

# -----------------------------------------------------------------------------
# Subcommands
# -----------------------------------------------------------------------------
cmd_dry() {
    _preflight
    _sweep "native generic agnostic" "${RUN_ID:-blind_dryrun_check}" --dry-run "$@"
}

cmd_native_generic() {
    _preflight; _check_python_deps; _set_wired_limit
    _sweep "native generic" "blind_${TRANSFORM}_native_generic" "$@"
}

cmd_full() {
    _preflight; _check_python_deps; _set_wired_limit
    _sweep "native generic agnostic" "blind_${TRANSFORM}_three_condition" "$@"
}

cmd_agnostic() {
    # Append the agnostic arm to an EXISTING native/generic run. The driver
    # verifies that seed_base, M_cycles, N_shots, transform, etc. match the
    # run's manifest before writing anything.
    _require_run_id "agnostic"
    _preflight; _check_python_deps; _set_wired_limit
    _sweep "agnostic" "$RUN_ID" "$@"
}

cmd_tanh() {
    # Nonlinear label stress test (Supporting Information level).
    _preflight; _check_python_deps; _set_wired_limit
    TRANSFORM=tanh _sweep "native generic" "blind_tanh_stress" "$@"
}

cmd_analyze() {
    _require_run_id "analyze"
    [[ -f blinding_analysis.py ]] || _die "blinding_analysis.py not found"
    "$PYTHON" blinding_analysis.py --results-root "$RESULTS_ROOT" \
        --run-id "$RUN_ID" --n-boot "$N_BOOT" --n-perm "$N_PERM" \
        --seed "$STAT_SEED" "$@"
}

cmd_oor() {
    _require_run_id "oor"
    [[ -f oor_extract.py ]] || _die "oor_extract.py not found"
    "$PYTHON" oor_extract.py --results-root "$RESULTS_ROOT" --run-id "$RUN_ID" "$@"
}

cmd_help() {
    cat <<EOF
run_blinding.sh  --  wrapper for the blinding (information-access) study

USAGE
  ./run_blinding.sh <subcommand> [extra args forwarded to the Python script]

SUBCOMMANDS
  dry             Print the plan for native+generic+agnostic (nothing written)
  native-generic  Native vs generic sweep         (run id: blind_<transform>_native_generic)
  full            Native + generic + agnostic     (run id: blind_<transform>_three_condition)
  agnostic        Append the agnostic arm to an existing run (requires RUN_ID)
  tanh            Native vs generic with the tanh label map (run id: blind_tanh_stress)
  analyze         Retention, paired dMAE, CIs, BH-FDR (requires RUN_ID)
  oor             Out-of-range / reject summary   (requires RUN_ID)
  help            This message

ENVIRONMENT VARIABLES (all optional)
  PYTHON          Python interpreter          (default: python3)
  CONFIG          Properties YAML             (default: properties.yaml)
  RESULTS_ROOT    Output root                 (default: ./RESULTS)
  MODELS          Model choices               (default: "2 7")
  PROPERTIES      Property keys               (default: "LOGKOW LOGS PKA LOGCMC")
  M_CYCLES        Paired cycles per cell      (default: 50)
  N_SHOTS         Demonstrations per prompt   (default: 32)
  SEED_BASE       Base seed, same for all arms (default: 0)
  TRANSFORM       zscore | linear | tanh      (default: zscore)
  RECODER_SEED    Agnostic SMILES recoder seed (default: 20260101)
  TEMPERATURE     Sampling temperature        (default: empty => 0.9)
  RUN_ID          Run folder name             (default: per subcommand)
  N_BOOT, N_PERM  Bootstrap / permutation draws (default: 10000 / 20000)
  STAT_SEED       Seed for the statistics     (default: 0)
  WIRED_LIMIT_MB  GPU wired-memory limit, MB  (default: 61440; 0 => skip sudo)

EXAMPLES
  ./run_blinding.sh dry
  ./run_blinding.sh native-generic
  RUN_ID=blind_zscore_native_generic ./run_blinding.sh agnostic
  RUN_ID=blind_zscore_native_generic ./run_blinding.sh analyze
  RUN_ID=blind_zscore_native_generic ./run_blinding.sh oor

  # Resume an interrupted sweep (completed cells are skipped):
  ./run_blinding.sh native-generic --resume
EOF
}

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
main() {
    local sub="${1:-help}"
    shift || true
    case "$sub" in
        dry)             cmd_dry            "$@" ;;
        native-generic)  cmd_native_generic "$@" ;;
        full)            cmd_full           "$@" ;;
        agnostic)        cmd_agnostic       "$@" ;;
        tanh)            cmd_tanh           "$@" ;;
        analyze)         cmd_analyze        "$@" ;;
        oor)             cmd_oor            "$@" ;;
        help|-h|--help)  cmd_help ;;
        *) echo "Unknown subcommand: $sub" >&2; echo; cmd_help; exit 1 ;;
    esac
}

main "$@"
