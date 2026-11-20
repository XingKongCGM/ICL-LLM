# ICL-PChem: In-Context Learning Evaluation Framework for PFAS Physicochemical Properties

A local, reproducible framework for evaluating how well large language models (LLMs) predict physicochemical properties of per- and polyfluoroalkyl substances (PFAS) through **in-context learning (ICL)**: the model sees a set of `(SMILES, value)` demonstrations in its prompt and is asked to predict the value for a new molecule. No fine-tuning is involved.

The framework is built around metrological principles: repeated stochastic cycles instead of single runs, robust statistics (median, IQR), an independent validation set, a permutation-null control, and Type A uncertainty estimates for pure predictions.

All inference runs **locally on Apple Silicon** through [MLX](https://github.com/ml-explore/mlx) and `mlx-lm`, using quantized models from the `mlx-community` organization on Hugging Face.

---

## Contents

- [Features](#features)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Supported models](#supported-models)
- [Supported properties](#supported-properties)
- [Data format](#data-format)
- [Configuration (`properties.yaml`)](#configuration-propertiesyaml)
- [Quick start](#quick-start)
- [The three phases](#the-three-phases)
- [Running sweeps](#running-sweeps)
- [Streamlit dashboard](#streamlit-dashboard)
- [Post-hoc analysis](#post-hoc-analysis)
- [Output layout](#output-layout)
- [Metrics](#metrics)
- [Blinding study (information access)](#blinding-study-information-access)
- [Legacy interactive mode](#legacy-interactive-mode)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License and disclaimer](#license-and-disclaimer)

---

## Features

- **Systematic (model × property) sweeps** with one command, run serially to respect MLX memory limits.
- **Three phases**: Testing (with optional permutation control), Prediction (fixed demonstrations), and Forecast (replicate sampling with uncertainty, SMILES only).
- **Per-cycle metric distributions** saved as NPZ files, so results are reported as distributions rather than single runs.
- **Reusable optimal demonstration sets**: the best Testing cycle's demos are saved per (model, property) and reused by later Prediction and Forecast passes.
- **Metrological uncertainty** in Forecast mode: mean, standard deviation, $u_A$, and $U_{95}$ per molecule per temperature.
- **Robustness fixes** for known mlx-community conversion issues (Llama-4 config type mismatches, Mistral tokenizer regex, missing chat templates, assistant-priming placement).
- **Streamlit dashboard** to launch, monitor, stop, and re-attach to ICL sweeps and blinding runs, and to browse their results.
- **Analysis script** for box plots, scaling-law fits, property correlation matrices, and a framework summary table.
- **Blinding study add-on** that hides the property name, units, value scale and (optionally) SMILES semantics, to test whether models learn from the demonstrations or recall from pretraining. See [BLINDING.md](BLINDING.md).

---

## Repository layout

```
.
├── RCHAT_PChem_Properties.py   # Core engine: MLX backend, prompts, metrics, run_cell* APIs
├── batch_driver.py             # CLI sweep driver over (model, property) cells
├── run_icl_sweep.sh            # Convenience shell wrapper around batch_driver.py
├── analysis.py                 # Post-hoc analysis of a finished Testing run
├── icl_dashboard.py            # Streamlit launcher / monitor / results viewer (ICL + blinding)
├── properties.yaml             # Property-to-dataset configuration
├── requirements.txt            # Python dependencies
│
├── blinding.py                 # Blinding toolkit: label transforms, SMILES recoder, prompts
├── rchat_blinding_hooks.py     # run_cell_blinded(): blinded analogue of run_cell()
├── blinding_driver.py          # Sweep over (model, property, condition) cells
├── blinding_analysis.py        # Retention, paired dMAE, bootstrap CIs, tests, BH-FDR
├── oor_extract.py              # Out-of-range / reject summary for blinded cells
├── run_blinding.sh             # Shell wrapper for the blinding study
├── BLINDING.md                 # Blinding study documentation
└── README.md
```

All scripts are expected to live in the same directory. `batch_driver.py` imports `RCHAT_PChem_Properties.py` from its own directory unless `--rchat-script-dir` is given.

---

## Requirements

### Hardware

- A Mac with **Apple Silicon** (M-series). MLX does not run on Intel Macs, Linux, or Windows.
- Enough unified memory for the models you select. As a rough guide, a 4-bit 70B model needs about 40 GB; 8B models need well under 10 GB. The default wrapper settings were developed on a 64 GB machine.

### Software

- macOS with Python 3.10 or newer
- Required Python packages:
  - `mlx-lm` (pulls in `mlx`)
  - `huggingface_hub`
  - `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- Optional:
  - `pyyaml` (a built-in fallback parser handles the shipped config shape if absent)
  - `scipy` (for exact Student's $t$ critical values in Forecast; a lookup table is used otherwise)
  - `streamlit` (only for the dashboard)

---

## Installation

```bash
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` includes the optional packages (`pyyaml`, `scipy`, `streamlit`); remove any you do not want.

Model weights are downloaded automatically from Hugging Face on first use and cached in `~/.cache/huggingface`. If a repository is gated, log in first:

```bash
huggingface-cli login
```

---

## Supported models

Models are selected by integer choice. The same integers are used by `batch_driver.py`, `run_icl_sweep.sh`, and the dashboard.

| Choice | Slug (used in folder names) | Hugging Face repository |
|:-:|---|---|
| 1 | `llama3.1-8b-q4` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| 2 | `llama3.3-70b-q4` | `mlx-community/Llama-3.3-70B-Instruct-4bit` |
| 3 | `nemotron-mini-4b-bf16` | `mlx-community/Nemotron-Mini-4B-Instruct-bf16-mlx` |
| 4 | `llama3.1-nemotron-70b-q4` | `mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-4bit` |
| 5 | `gemma-4-31b-optiq-q4` | `mlx-community/gemma-4-31B-it-OptiQ-4bit` |
| 6 | `mistral-small-3.1-24b-q4` | `mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit` |
| 7 | `llama4-scout-17b-16e-q4` | `mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit` |

To add a model, extend `LOCAL_MODEL_MAP` in `RCHAT_PChem_Properties.py`, `MODEL_SLUG` in `batch_driver.py`, `MODEL_LABELS` in `icl_dashboard.py`, and (for scaling plots) the parameter-count tables in `analysis.py`.

Per-query generation budget is 256 tokens (512 for the Gemma-4 OptiQ reasoning model).

---

## Supported properties

Property keys used on the command line, and the prompt variable each one expects:

| Key | Property | CSV column (`var`) |
|---|---|---|
| `LOGCMC` | Log critical micelle concentration | `logCMC` |
| `MP` | Melting point (°C) | `mpC` |
| `BP` | Boiling point (°C) | `bpC` |
| `LOGKOW` | Log octanol-water partition coefficient | `logKow` |
| `LOGS` | Log aqueous solubility | `logS` |
| `PKA` | Acid dissociation constant | `pka` |
| `VP` | Negative log vapor pressure, $-\log_{10}(\text{VP}/\text{mmHg})$ | `logVP` |
| `HLC` | Negative log Henry's law constant, $-\log_{10}(\text{HLC}/(\text{atm m}^3\,\text{mol}^{-1}))$ | `logHLC` |

The units above are the ones stated to the model in the prompt templates, so your data must use them. The `units` field in `properties.yaml` is informational and is kept consistent with the prompts.

Prompt templates for each key are defined in `get_template_prompts()` in `RCHAT_PChem_Properties.py`. The module also contains templates for `GIBBS` (solvation free energy) and `LOGKWA` (water-air partition coefficient), which can be added to `properties.yaml` if you have data for them.

To add a new property, add a prompt template branch in `get_template_prompts()` and a matching block in `properties.yaml`.

---

## Data format

Each property needs CSV files with a header row. Two columns are required:

| Column | Content |
|---|---|
| `Molecule` | SMILES string of the molecule |
| `<var>` | Property value, with the column name matching `var` in `properties.yaml` |

Extra columns are allowed and are carried through into the saved demonstration file.

Example (`PFAS_pKa_Training.csv`):

```csv
Molecule,pka
OC(=O)C(F)(F)F,0.52
OC(=O)C(F)(F)C(F)(F)F,0.38
...
```

Three files per property:

| File | Used by | Property column required? |
|---|---|---|
| `training` | Testing (split internally into a demo pool and a test pool) | Yes |
| `validation` | Testing (independent validation) | Yes |
| `predict` | Prediction and Forecast | **Prediction: yes** (rows without a value are dropped, since the phase scores predictions against known values). **Forecast: no** (only `Molecule` is read). |

Rows with non-numeric or missing property values are dropped. A Testing cell needs at least 20 valid records across training and validation combined.

---

## Configuration (`properties.yaml`)

Set `DATA_ROOT` to the directory that holds your datasets, then check that each path matches your file names.

```yaml
DATA_ROOT: /Users/you/data/PFAS

properties:
  PKA:
    var:         pka
    title:       "Acid Dissociation Constant"
    units:       "-"
    training:    ${DATA_ROOT}/PKa/PFAS_pKa_Training.csv
    validation:  ${DATA_ROOT}/PKa/PFAS_pKa_Validation.csv
    predict:     ${DATA_ROOT}/PKa/PFAS_pKa_Predict.csv
    optimal_dir: ${DATA_ROOT}/PKa/Optimal_Training_Set
```

| Key | Meaning |
|---|---|
| `var` | Column name of the property in the CSVs; also used in the prompt |
| `title`, `units` | Human-readable labels for reports |
| `training`, `validation`, `predict` | CSV paths (see [Data format](#data-format)) |
| `optimal_dir` | Where the best-cycle demonstration set is written during Testing and read during Prediction / Forecast, as `{optimal_dir}/{model_slug}_optimal_samples.csv`. Created if missing. If omitted, the per-run cell folder is used instead. |

Notes:

- `${NAME}` substitution uses **top-level scalar keys in the YAML only** (such as `DATA_ROOT`), not shell environment variables.
- Because `optimal_dir` lives in the data tree and the file name includes the model slug, one Testing sweep can serve any number of later Prediction or Forecast passes, with any `--run-id`, and different models never overwrite each other.

---

## Quick start

```bash
# 1. Edit DATA_ROOT in properties.yaml.

# 2. Sanity check: one model (7), one property (PKA), 2 cycles.
./run_icl_sweep.sh smoke

# 3. See what a full sweep would do, without running it.
./run_icl_sweep.sh dry

# 4. Full Testing sweep (default models 1 2 7, all properties, 50 cycles).
RUN_ID=paper_v1 ./run_icl_sweep.sh testing

# 5. Analyze it.
RUN_ID=paper_v1 ./run_icl_sweep.sh analyze
```

Make the wrappers executable once with `chmod +x run_icl_sweep.sh run_blinding.sh`.

Before loading any model, both wrappers check that the required Python packages are importable by `$PYTHON` and stop with an install hint if not.

> **GPU memory limit and `sudo`:** before subcommands that load models (`smoke`, `testing`, `permute`, `predict`, `forecast`), the wrapper runs `sudo sysctl iogpu.wired_limit_mb=$WIRED_LIMIT_MB` to let large models use more unified memory. This asks for your password, and the setting resets on reboot. The default, 61440 MB (60 GB), suits a 64 GB machine; set a value that leaves headroom for macOS on yours, or skip the step entirely with `WIRED_LIMIT_MB=0`:
>
> ```bash
> WIRED_LIMIT_MB=0 ./run_icl_sweep.sh smoke        # no sudo
> WIRED_LIMIT_MB=110000 ./run_icl_sweep.sh testing  # e.g. a 128 GB machine
> ```

---

## The three phases

### 1. Testing (default)

For each (model, property) cell:

1. The `training` CSV is split 80/20 into a **demonstration pool** and a **test pool** (fixed `random_state`, so the split is reproducible).
2. For each cycle $k = 1, \dots, M_c$, a fresh random subset of $N_\text{shots}$ demonstrations is drawn from the demo pool (seed $= \text{seed\_base} + k$). The model predicts every molecule in the test pool with those demonstrations in context.
3. Per-cycle metrics (MAE, RMSE, $R^2$, accuracy, F1, NRMSE, rejects) are stored.
4. The cycle with the **lowest test MAE** defines the optimal demonstration set, which is saved to `optimal_dir`.
5. The optimal set is evaluated once on the **independent validation** CSV.

The number of model calls per cell is

$$
N_\text{calls} = M_c \cdot N_\text{test} + N_\text{val},
$$

so with $M_c = 50$ and a test pool of 40 molecules, expect about 2,000 inferences per cell. Plan run times accordingly, especially for 70B models.

**Permutation control** (`--permutation-control`): for each real cell, a second cell is run in which the property labels of the demonstrations are randomly shuffled among themselves (SMILES unchanged). This gives a null baseline: the error a model achieves when the demonstrations carry no structure-property information. A model that has genuinely learned from context should beat this null.

### 2. Prediction (`--predict`)

Loads `{optimal_dir}/{model_slug}_optimal_samples.csv` and applies those fixed demonstrations once to every molecule in the `predict` CSV, scoring against the known values. Writes `predictions.csv` (predicted, actual) per cell. Cells without a saved optimal set are reported as `SKIPPED`.

### 3. Forecast (`--forecast`)

Pure prediction with uncertainty. Reads only the `Molecule` column of the `predict` CSV (falling back to `validation` if `predict` is not configured). For each temperature $T$ in `--temperatures` and each molecule, the model is queried $N_\text{exp}$ times with the same fixed demonstrations. Per molecule and temperature it reports the mean, median, sample standard deviation, min, max, and

$$
u_A = \frac{s}{\sqrt{N_\text{valid}}}, \qquad
U_{95} = t_{0.975,\,N_\text{valid}-1}\; u_A ,
$$

where $s$ is the sample standard deviation of the valid replicates. All raw replicate values are kept in the output CSV.

These quantities describe the **sampling dispersion of the model's own predictions** at a given temperature. They do not include model bias relative to the true value; the Testing and Prediction phases characterize that. At $T = 0$ (greedy decoding) all replicates are identical, so $s = u_A = U_{95} = 0$.

### Temperature

| Phase | Flag | Default |
|---|---|---|
| Testing, Prediction | `--temperature T` | `DEFAULT_TEMPERATURE = 0.9` in `RCHAT_PChem_Properties.py` (reproduces published runs) |
| Forecast | `--temperatures T1 T2 ...` | `0.7` (driver); `0.3 0.7 1.0` (wrapper) |

---

## Running sweeps

### With the shell wrapper

```
./run_icl_sweep.sh <subcommand> [extra batch_driver.py args...]
```

| Subcommand | Action |
|---|---|
| `smoke` | One cell (model 7, PKA, 2 cycles) end-to-end check |
| `dry` | Print the Testing plan without running |
| `testing` | Full Testing sweep |
| `permute` | Testing sweep plus permutation-null control |
| `predict` | Prediction pass using saved optimal sets |
| `forecast` | Forecast pass over a temperature sweep |
| `analyze` | Run `analysis.py` on a finished run (requires `RUN_ID`) |
| `full` | `testing` then `analyze`, sharing one run id |
| `help` | Usage |

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON` | `python3` | Interpreter |
| `CONFIG` | `properties.yaml` | Property config |
| `RESULTS_ROOT` | `./RESULTS` | Output root |
| `MODELS` | `"1 2 7"` | Model choices |
| `PROPERTIES` | all in config | Property keys |
| `M_CYCLES` | `50` | Testing cycles per cell |
| `N_SHOTS` | `32` | Demonstrations per prompt |
| `TEMPERATURE` | module default (0.9) | Testing / Prediction temperature |
| `TEMPERATURES` | `"0.3 0.7 1.0"` | Forecast sweep |
| `N_EXP` | `20` | Forecast replicates per molecule per temperature |
| `RUN_ID` | timestamp | Run identifier |
| `SMOKE_MODEL`, `SMOKE_PROP` | `7`, `PKA` | Smoke-test cell |
| `WIRED_LIMIT_MB` | `61440` | GPU wired-memory limit in MB; `0` skips the `sudo sysctl` step |

Examples:

```bash
# Two models, two properties, T = 0.3
MODELS="2 7" PROPERTIES="LOGKOW PKA" TEMPERATURE=0.3 ./run_icl_sweep.sh testing

# Testing with permutation control and 10 cycles (extra args are forwarded)
./run_icl_sweep.sh permute --M-cycles 10

# Prediction reusing optimal sets from any earlier Testing run
./run_icl_sweep.sh predict --run-id predict_v1

# Forecast, 30 replicates, two temperatures, one cell
N_EXP=30 TEMPERATURES="0.5 1.0" ./run_icl_sweep.sh forecast --models 7 --properties PKA
```

### With `batch_driver.py` directly

```bash
python batch_driver.py --config properties.yaml --models 2 7 \
    --properties LOGKOW MP BP --M-cycles 50 --N-shots 32 \
    --results-root ./RESULTS --run-id myrun \
    [--permutation-control] [--temperature 0.9] [--dry-run] [--notify]
```

| Flag | Default | Description |
|---|---|---|
| `--config` | required | Path to `properties.yaml` |
| `--models` | required | Model choices (1 to 7) |
| `--properties` | all | Property keys |
| `--M-cycles` | 50 | Testing cycles per cell |
| `--N-shots` | 32 | Demonstrations per prompt |
| `--results-root` | required | Output root |
| `--run-id` | timestamp | Run folder name |
| `--permutation-control` | off | Add a shuffled-label null cell for each cell |
| `--predict` | off | Prediction phase |
| `--forecast` | off | Forecast phase (mutually exclusive with `--predict`) |
| `--N-exp` | 20 | Forecast replicates |
| `--temperatures` | 0.7 | Forecast temperature sweep |
| `--temperature` | module default | Testing / Prediction temperature |
| `--dry-run` | off | Print plan and exit |
| `--notify` | off | macOS notification when finished |
| `--rchat-script-dir` | driver's directory | Location of `RCHAT_PChem_Properties.py` |

**Execution model.** Models run one at a time in the order given. Each model is loaded once and used for all requested properties, then unloaded (and the MLX cache cleared) before the next model loads, so peak memory is that of the largest single model rather than their sum. Models are never run in parallel because MLX is not thread-safe and the backend keeps state in module globals.

**Fault tolerance.** A failing cell is logged and recorded as `FAILED` (or `SKIPPED` for missing inputs) in the summary CSV, and the sweep continues.

---

## Streamlit dashboard

```bash
pip install -r requirements.txt    # includes streamlit
streamlit run icl_dashboard.py     # opens http://localhost:8501
```

Run it from the repository directory (script paths are relative).

**Run mode.** Pick a phase, set its options, check the exact command in the preview, then launch, watch the live log (auto-refresh) and stop the job if needed.

| Phase | Script | Notes |
|---|---|---|
| Testing, Predict, Forecast | `batch_driver.py` | Same options as the CLI |
| Analyze | `analysis.py` | Pick an existing Testing run from a list |
| Blinding sweep | `blinding_driver.py` | Conditions, label transform, cycles, shots, seed base, temperature; transform and recoder parameters in an expander; Resume, Allow mismatch, Dry run |
| Blinding analysis | `blinding_analysis.py`, optionally followed by `oor_extract.py` | Bootstrap and permutation counts, statistics seed, FDR alpha, retention tolerance |

For a **Blinding sweep**, leaving the run id blank uses the same default names as `run_blinding.sh` (for example `blind_zscore_native_generic`). Typing the id of an existing blinding run appends to it. The dashboard then lists the conditions already present and previews the driver's pairing check: it shows any setting (seed base, cycles, shots, transform parameters, temperature) that differs from the run's manifest, so you can fix it before launching.

**Results mode** has two views:

- **ICL sweeps:** summary CSVs (Testing, Predict, Forecast) with download buttons, completion markers, and per-cell cycle diagnostics from `cycles.npz` (histogram, box plot, median, mean, standard deviation, IQR, best cycle, validation metrics).
- **Blinding runs:**
  - **Settings and progress:** settings from the manifests, and how many planned cells have results.
  - **Tables:** summary, contrast table, main native vs generic table, out-of-range summary, and the condition-panel figure, when present.
  - **Paired comparison:** for one model and property, per-cycle test MAE for every condition, the paired differences $\Delta_k$ as a strip plot, and quick retention and median $\Delta$MAE values.
  - **Per-cell diagnostics:** from `cycles_blinded.npz`, including transformed-unit MAE and out-of-range counts.

**Re-attach.** Launching writes `.dashboard_logs/runlock.json`. If Streamlit restarts or the browser session resets while a job is still running, the dashboard re-adopts it, so the live log and Stop keep working. This works for every phase, blinding included. Stale locks are cleared automatically.

**GPU memory limit.** The dashboard cannot prompt for a password, so it does not run `sudo sysctl iogpu.wired_limit_mb=...`. For large models, run that command (or any wrapper subcommand that loads a model) once after each reboot before launching from the dashboard.

The dashboard is a launcher and monitor, not a scheduler: one job at a time. Jobs are started in their own process session, so they keep running if the dashboard is closed.

---

## Post-hoc analysis

```bash
# Either point at the run directory...
python analysis.py --run-dir ./RESULTS/runs/<run_id>

# ...or give the results root and run id (what the wrapper and dashboard use)
python analysis.py --results-root ./RESULTS --run-id <run_id>

# Optional: --analysis-root <dir> to write outputs elsewhere
```

The same analysis is available as `RUN_ID=<run_id> ./run_icl_sweep.sh analyze` and as the **Analyze** phase in the dashboard.

Outputs go to `RESULTS/analysis/<run_id>/` by default:

| File | Content |
|---|---|
| `framework_table.csv` | One row per real cell: median MAE, IQR, Bowley skewness, validation metrics, information estimate, null MAE, MAE skill |
| `boxplot_<PROPERTY>.png` | Per-cycle MAE distributions across models |
| `scaling_fits_N_active_B.png` | $\log$ MAE vs. $\log$ active parameters, per property |
| `scaling_fits_N_total_B.png` | Same, using total parameters (differs for mixture-of-experts models) |
| `property_correlation_matrix.png` | Spearman correlation of median MAE across models, property by property |

A rollup table is also printed to the terminal.

---

## Output layout

```
RESULTS/
├── runs/<run_id>/
│   ├── manifest.json / manifest_predict.json / manifest_forecast.json
│   ├── summary.csv                  # Testing: one row per cell
│   ├── predict_summary.csv          # Prediction
│   ├── forecast_summary.csv         # Forecast
│   ├── RUN_COMPLETE*.txt            # Phase-specific completion markers
│   └── <model_slug>_<PROPERTY>[_perm]/
│       ├── cycles.npz               # Per-cycle metric arrays + validation scalars
│       ├── testing_summary.txt
│       ├── validation_summary.txt
│       ├── predictions.csv          # Prediction phase
│       ├── predict_summary.txt
│       ├── forecast_<PROPERTY>.csv  # Forecast phase, one row per (T, molecule)
│       └── forecast_summary.txt
└── analysis/<run_id>/               # analysis.py outputs

<optimal_dir>/
└── <model_slug>_optimal_samples.csv # Best-cycle demonstrations (Testing)
```

Phase-specific manifests and markers mean Testing, Prediction, and Forecast can share a run id without overwriting one another.

**`cycles.npz` keys.** Per-cycle arrays of length $M_c$: `MAE_test`, `RMSE_test`, `MSE_test`, `R2_test`, `Acc_test`, `F1_test`, `NRMSE_mean_test`, `NRMSE_range_test`, `n_rejects_test`, `seed_per_cycle`. Scalars: `best_cycle` (0-based), `MAE_val`, `RMSE_val`, `MSE_val`, `R2_val`, `Acc_val`, `F1_val`, `NRMSE_mean_val`, `NRMSE_range_val`, `n_rejects_val`, `median_value`, `mean_value`, `y_min`, `y_max`, `wall_time_sec`, `wall_time_testing_sec`.

Load one with:

```python
import numpy as np
d = dict(np.load("RESULTS/runs/myrun/llama4-scout-17b-16e-q4_PKA/cycles.npz"))
print(np.median(d["MAE_test"]), d["R2_val"])
```

---

## Metrics

For $n$ valid predictions $\hat y_i$ with reference values $y_i$:

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |\hat y_i - y_i|, \qquad
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (\hat y_i - y_i)^2}, \qquad
R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}
$$

**Normalized RMSE** (in percent), using the mean and range of the combined training and validation data:

$$
\text{NRMSE}_\text{mean} = 100\,\frac{\text{RMSE}}{|\bar y|}, \qquad
\text{NRMSE}_\text{range} = 100\,\frac{\text{RMSE}}{y_\text{max} - y_\text{min}}
$$

**Binary accuracy**: both $\hat y_i$ and $y_i$ are classified as above or below the dataset median $\tilde y$, and accuracy is the percentage of matching classes. This measures ordinal ranking skill.

**Macro F1**: predictions and references are each binned into quintiles, and the F1 score is averaged over the five classes.

**Distribution statistics over cycles** (Testing): median MAE, IQR $= Q_3 - Q_1$, and Bowley (quartile) skewness

$$
B = \frac{(Q_3 - Q_2) - (Q_2 - Q_1)}{Q_3 - Q_1}.
$$

**Permutation-normalized skill** (`analysis.py`):

$$
\text{skill} = 1 - \frac{\widetilde{\text{MAE}}_\text{real}}{\widetilde{\text{MAE}}_\text{null}},
$$

where $\widetilde{\text{MAE}}$ is the median over cycles. Skill $> 0$ means the model does better than with shuffled demonstrations.

**Information estimate** (`analysis.py`):

$$
I \approx \tfrac{1}{2}\ln\!\left(\frac{\sigma_Y^2}{\text{MSE}_\text{val}}\right), \qquad
\sigma_Y \approx \frac{y_\text{max} - y_\text{min}}{4},
$$

in nats. For Gaussian $Y$ this is the rate-distortion **lower bound** on $I(Y;\hat Y)$ at distortion $\text{MSE}_\text{val}$; because $\sigma_Y$ is approximated from the range, treat it as an estimate rather than a strict bound.

**Scaling fits** (`analysis.py`): per property, $\log_{10}\widetilde{\text{MAE}} = a + b\,\log_{10} N$, reported as $\text{MAE} \propto N^{-\alpha}$ with $\alpha = -b$, where $N$ is the active or total parameter count in billions.

**Rejects**: a response from which no number can be parsed is counted as a reject and excluded from the metrics. The parser takes the last number in the response. Reject counts are reported for every cell and should be checked alongside the error metrics.

---

## Blinding study (information access)

A controlled experiment that removes, step by step, the information a model could take from pretraining rather than from the demonstrations. Each cell is run in up to three cycle-paired conditions:

- **native**: the main-study prompt;
- **generic**: anonymous property label and invertibly transformed values;
- **agnostic**: generic, plus SMILES with every atom symbol recoded.

All predictions are mapped back to native units before scoring. The retention ratio

$$
R = \frac{\operatorname{median}_k \text{MAE}_{\text{native},k}}{\operatorname{median}_k \text{MAE}_{\text{blinded},k}}
$$

and the paired per-cycle differences $\Delta_k = \text{MAE}_{\text{blinded},k} - \text{MAE}_{\text{native},k}$ (bootstrap CI, Wilcoxon and permutation tests, BH-FDR) quantify how much accuracy survives.

```bash
./run_blinding.sh native-generic                                  # models 2 and 7, four properties
RUN_ID=blind_zscore_native_generic ./run_blinding.sh agnostic     # append the agnostic arm
RUN_ID=blind_zscore_native_generic ./run_blinding.sh analyze
```

Full documentation, including the transforms, recoder, pairing guard and statistics, is in [BLINDING.md](BLINDING.md).

---

## Legacy interactive mode

`RCHAT_PChem_Properties.py` can also be run on its own:

```bash
python RCHAT_PChem_Properties.py
```

It prompts for a model, a property, and a mode (Testing, K-Fold, LOO, or Predict), and supports `GIBBS` and `LOGKWA` in addition to the properties above. Dataset paths in this mode are hard-coded under `JSONL_DIR` in the `__main__` block, so you must edit them before use. For systematic work, `batch_driver.py` is the recommended entry point.

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `mlx_lm is not installed` / `ImportError` | Install with `pip install mlx-lm`, on an Apple Silicon Mac. |
| All predictions rejected, metrics are `NaN` | Look at the `[DIAG 1/3]` lines printed at the start of a run: they show the raw model output and the parsed number. Common causes are a broken tokenizer, a model answering in prose, or too small a token budget. |
| Llama-4 load fails with a dataclass type error | The code patches the known `attn_temperature_tuning` and `rope_theta` type mismatches automatically. If it still fails, upgrade with `pip install --upgrade mlx-lm`. |
| Garbled Mistral output | `fix_mistral_regex=True` is applied automatically; upgrade `mlx-lm` if your version does not accept `tokenizer_config`. |
| `chat_template is not set` note | Some mlx-community conversions drop the chat template; a family-specific fallback template is used automatically. |
| Prediction / Forecast cells all `SKIPPED` | No `{optimal_dir}/{model_slug}_optimal_samples.csv` exists for that cell. Run Testing for the same model and property first. |
| `Config references ${X} but it is not defined` | Define `X` as a top-level key in `properties.yaml`. |
| `MISSING training CSV`, or `Data files not found ... Nothing can run` | Check `DATA_ROOT` and file names in `properties.yaml`. The shipped value `/path/to/your/data` is a placeholder and must be replaced with your data folder. |
| `missing Python packages for python3: ...` | The interpreter the wrapper uses lacks a package. Run `python3 -m pip install -r requirements.txt` (or point `PYTHON=` at the environment that has them). |
| Password prompt when starting a sweep | That is the `sudo sysctl` memory-limit step. Use `WIRED_LIMIT_MB=0` to skip it. |
| Swapping or very slow 70B runs | Reduce the number of large models per sweep, close other applications, and check `WIRED_LIMIT_MB`. |
| Dashboard shows a stale runlock | It is cleared automatically on the next page load; you can also delete `.dashboard_logs/runlock.json`. |

---

## Citation

If you use this framework, please cite:

```
Carlos Gonzalez, "Towards a Metrological Validation of PFAS Property Predictions by In-Context LLMs Across Four Architectures", 2026, NIST.
```


---

## License and disclaimer


Certain commercial equipment and software are identified in this license statement to specify the computational procedure adequately. Such identification is not intended to imply recommendation or endorsement by the National Institute of Standards and Technology, nor is it intended to imply that the equipment and software identified are necessarily the best available for the purpose.


Model weights are downloaded from Hugging Face and are governed by their own licenses (Llama, Gemma, Mistral, and Nemotron licenses respectively). Users are responsible for complying with them.

LLM predictions produced by this framework are research outputs. They should not be used as reference values without independent validation.
