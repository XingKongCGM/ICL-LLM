# Blinding study: controlled information access

This add-on tests **what an LLM actually uses** when it predicts a PFAS property in context. If a model's accuracy comes from learning the structure-value mapping in the demonstrations, it should survive removing the property's name, units and familiar numeric scale. If it relies on what it memorized in pretraining (for example, typical pKa values of perfluorinated acids), accuracy should drop once those cues are hidden.

Each (model, property) cell is run under up to three prompt **conditions**, all **cycle-paired** so they can be compared one cycle at a time:

| Condition | Structure shown | Property identity | Label values |
|---|---|---|---|
| `native` | SMILES | Property name and units (main-study prompt) | Original values |
| `generic` | SMILES | Anonymous label `Response R-17`, no name or units | Transformed by an invertible map |
| `agnostic` | Recoded SMILES (atom symbols replaced) | Anonymous label | Transformed by an invertible map |

Predictions in the blinded conditions are mapped back to native units **before** any metric is computed, so every MAE in this study is in the property's own units and directly comparable across conditions.

---

## Contents

- [Files](#files)
- [Quick start](#quick-start)
- [How the conditions are built](#how-the-conditions-are-built)
- [Cycle pairing](#cycle-pairing)
- [Running the study](#running-the-study)
- [Statistics](#statistics)
- [Out-of-range and reject summary](#out-of-range-and-reject-summary)
- [Output layout](#output-layout)
- [Reproducibility notes](#reproducibility-notes)

---

## Files

| File | Role |
|---|---|
| `blinding.py` | Self-contained toolkit: label transforms, SMILES recoder, blinded instruction text, `BlindingSpec`. No model calls. Run it directly for a self-test. |
| `rchat_blinding_hooks.py` | `run_cell_blinded()`: the blinded analogue of `run_cell()`, reusing the core's query, parsing and metric functions. |
| `blinding_driver.py` | Sweep over (model, property, condition) cells. |
| `blinding_analysis.py` | Retention ratios, paired ΔMAE, bootstrap CIs, Wilcoxon and permutation tests, BH-FDR, figure. |
| `oor_extract.py` | Out-of-range and reject counts and rates for blinded cells. |
| `run_blinding.sh` | Shell wrapper with subcommands for all of the above. |

The study uses the same `properties.yaml`, `batch_driver.py` (model numbers, config loader) and `RCHAT_PChem_Properties.py` (local MLX backend) as the main framework. Installation is the same; see the main [README](README.md).

---

## Quick start

```bash
# Plan only (nothing written)
./run_blinding.sh dry

# Native vs generic, models 2 and 7, LOGKOW LOGS PKA LOGCMC, 50 paired cycles
./run_blinding.sh native-generic            # run id: blind_zscore_native_generic

# Add the agnostic arm to the same run folder
RUN_ID=blind_zscore_native_generic ./run_blinding.sh agnostic

# Statistics, figure and out-of-range summary
RUN_ID=blind_zscore_native_generic ./run_blinding.sh analyze
RUN_ID=blind_zscore_native_generic ./run_blinding.sh oor
```

An interrupted sweep can be restarted with `--resume`, which skips every cell that already has its `cycles_blinded.npz`:

```bash
./run_blinding.sh native-generic --resume
```

---

## How the conditions are built

### Prompts

The **native** arm uses the main-study instruction text from `get_template_prompts()` and the line format `Molecule: <SMILES>` / `<var>: <value>`.

The **generic** arm uses a neutral instruction and anonymous keys:

```
Examples contain a string and its associated response R-17.
Infer the response R-17 for the final string from the examples.
Return only one numeric value.
String: OC(=O)C(F)(F)F
Response R-17: 407.283

String: OC(=O)CC(F)(F)F
Response R-17: 605.963

String: OC(=O)CCC(F)(F)F
Response R-17:
```

The **agnostic** arm is identical except that it says "coded string", uses the key `Code string`, and shows recoded structures.

The blinded instruction text is checked against a list of banned words (property names, units, "molecule", "SMILES", "PFAS", and so on). If any appears, `blinding.py` raises an error instead of running a contaminated control.

### Label transforms

Each transform is **fitted on the demonstrations of the current cycle only** (never on test or validation values) and applied to every demonstration label. The model's answer $\hat z$ is inverted to native units, $\hat y = f^{-1}(\hat z)$, before scoring.

| `--transform` | Forward map $z = f(y)$ | Inverse | Out of range possible? |
|---|---|---|---|
| `zscore` (default, recommended) | $z = c + s\,\dfrac{y - \mu}{\sigma}$ | $y = \mu + \dfrac{\sigma}{s}(z - c)$ | No: defined for every real $z$ |
| `linear` | $z = z_\text{max}\left(1 - \dfrac{y - y_\text{min}}{y_\text{max} - y_\text{min}}\right)$ | $y = y_\text{min} + \left(1 - \dfrac{z}{z_\text{max}}\right)(y_\text{max} - y_\text{min})$ | Yes: outside $[0, z_\text{max}]$ |
| `tanh` | $z = c + a \tanh\!\left(\dfrac{y - \mu}{\sigma}\right)$ | $y = \mu + \sigma\,\operatorname{artanh}\!\left(\dfrac{z - c}{a}\right)$ | Yes: outside $(c - a,\, c + a)$; the argument is clamped to keep $\hat y$ finite |

Here $\mu$, $\sigma$, $y_\text{min}$, $y_\text{max}$ come from the cycle's demonstrations. Defaults: $c = 500$, $s = 100$ for `zscore` (`--std-center`, `--std-scale`; a negative $s$ also reverses the ordering); $z_\text{max} = 100$ for `linear` (`--out-max`); $c = 50$, $a = 40$ for `tanh` (`--tanh-center`, `--tanh-amp`).

`zscore` is recommended because its inverse has no boundary: every parsed answer maps to a finite native value, so a higher blinded MAE reflects how well the model learned the in-context mapping rather than whether it stayed inside a window. With `linear` or `tanh`, out-of-range answers are kept (not clipped) and counted separately; see [Out-of-range](#out-of-range-and-reject-summary).

### SMILES recoder (agnostic only)

One global, seeded, one-to-one substitution is built over every SMILES in the training and validation files of the requested properties. Each atom symbol (`C`, `F`, `O`, `S`, aromatic lowercase, `Cl`, `Br`, and the leading element inside brackets) is replaced by a two-letter code such as `Xw`. Bonds, branches, ring-closure digits, charges, hydrogen counts and stereo marks are kept, so the formal pattern is intact but the string no longer reads as chemistry:

```
OC(=O)C(F)(F)C(F)(F)F         ->  XgXw(=Xg)Xw(Ye)(Ye)Xw(Ye)(Ye)Ye
FC(F)(F)C(F)(F)S(=O)(=O)[O-]  ->  YeXw(Ye)(Ye)Xw(Ye)(Ye)Xk(=Xg)(=Xg)[Yc-]
```

An element written inside brackets gets its own code (here `[O-]` becomes `[Yc-]` while a plain `O` is `Xg`). The mapping is written to `recoder_map.json` for the Supporting Information. It depends on `--recoder-seed` and on the set of properties requested.

---

## Cycle pairing

Every condition of a cell uses the same `--seed-base`. Cycle $k$ then has, in every arm:

- the same train/test split (`random_state = seed_base`),
- the same demonstration molecules in the same order (`random.Random(seed_base + k)`),
- the same test queries in the same order.

The only difference between arms is the information in the prompt, so the per-cycle difference

$$
\Delta_k = \text{MAE}_{\text{blinded},k} - \text{MAE}_{\text{native},k}
$$

is a paired quantity.

**Appending to an existing run.** Running a new condition with an existing `--run-id` adds new cell folders and summary rows without touching existing cells. Before writing anything, the driver compares the new settings with the run's `manifest.json` (`seed_base`, `M_cycles`, `N_shots`, `transform`, `value_fmt`, transform parameters, `temperature`). It also compares the recoder map if one exists. If anything differs it stops, because the arms would no longer be paired. `--allow-mismatch` overrides this. Each append writes its own `manifest_<timestamp>.json`; the original `manifest.json` is never overwritten. Manifests from older versions of the driver do not record `std_center`, `std_scale` or `temperature`. For such runs the driver says it cannot verify those settings and continues, so make sure they match.

---

## Running the study

### `run_blinding.sh`

| Subcommand | Conditions | Default run id |
|---|---|---|
| `dry` | native, generic, agnostic (plan only) | `blind_dryrun_check` |
| `native-generic` | native, generic | `blind_<transform>_native_generic` |
| `full` | native, generic, agnostic | `blind_<transform>_three_condition` |
| `agnostic` | agnostic, appended to `RUN_ID` | `RUN_ID` (required) |
| `tanh` | native, generic with `--transform tanh` | `blind_tanh_stress` |
| `analyze` | statistics on `RUN_ID` | |
| `oor` | out-of-range summary on `RUN_ID` | |

Environment variables: `MODELS` (default `"2 7"`), `PROPERTIES` (`"LOGKOW LOGS PKA LOGCMC"`), `M_CYCLES` (50), `N_SHOTS` (32), `SEED_BASE` (0), `TRANSFORM` (`zscore`), `RECODER_SEED` (20260101), `TEMPERATURE` (empty means 0.9), `RUN_ID`, `N_BOOT` (10000), `N_PERM` (20000), `STAT_SEED` (0), `RESULTS_ROOT`, `CONFIG`, `PYTHON`, `WIRED_LIMIT_MB`. Run `./run_blinding.sh help` for the full list. Extra arguments are forwarded to the Python script.

Model-loading subcommands check the Python dependencies first, then set the GPU memory limit (asks for a password unless `WIRED_LIMIT_MB=0`), exactly like `run_icl_sweep.sh`.

### From the dashboard

`icl_dashboard.py` has **Blinding sweep** and **Blinding analysis** phases with the same options, and a **Blinding runs** results view with a paired per-cycle comparison. See the [dashboard section](README.md#streamlit-dashboard) of the README.

### `blinding_driver.py` directly

```bash
python blinding_driver.py --config properties.yaml --models 2 7 \
    --properties LOGKOW LOGS PKA LOGCMC --conditions native generic \
    --transform zscore --M-cycles 50 --N-shots 32 --seed-base 0 \
    --results-root ./RESULTS --run-id blind_zscore_native_generic
```

Main flags: `--conditions`, `--transform`, `--M-cycles`, `--N-shots`, `--seed-base`, `--temperature`, `--recoder-seed`, `--std-center`, `--std-scale`, `--out-max`, `--tanh-center`, `--tanh-amp`, `--value-fmt`, `--resume`, `--allow-mismatch`, `--dry-run`, `--notify`.

The number of model calls per cell is $M_c \cdot N_\text{test} + N_\text{val}$, as in the main Testing phase, multiplied by the number of conditions.

**Model numbers.** Model choices follow `batch_driver.py`: 2 = Llama-3.3-70B, 7 = Llama-4-Scout-17B-16E. Earlier internal versions numbered Scout as 8. Run folders are named by model slug, not number, so older runs can be appended to and analyzed with `--models 7`.

---

## Statistics

`blinding_analysis.py` (or `./run_blinding.sh analyze`) pairs each blinded arm with the native arm of the same (model, property) and reports, from the $M_c$ per-cycle test MAEs:

**Retention ratio**

$$
R = \frac{\operatorname{median}_k \text{MAE}_{\text{native},k}}{\operatorname{median}_k \text{MAE}_{\text{blinded},k}}
$$

$R \approx 1$: performance retained under blinding. $R < 1$: the blinded arm has larger error. $R > 1$: the blinded arm has smaller error.

**Paired difference.** The median of $\Delta_k$ with a percentile bootstrap 95% confidence interval (`--n-boot` resamples).

**Tests on $\Delta_k$**

- Wilcoxon signed-rank test, two-sided (SciPy if installed; otherwise a normal approximation with tie correction).
- Paired sign-flip permutation test on the mean of $\Delta_k$ (`--n-perm` draws), two-sided, with $p = (b + 1)/(B + 1)$, where $b$ is the number of draws at least as extreme as observed out of $B$.

**Multiple comparisons.** Benjamini-Hochberg FDR adjustment across all contrasts in the run, separately for the permutation and Wilcoxon p-values. Contrasts without a p-value are left out of the family.

**Interpretation label** (from $R$ and the permutation q-value, with `--alpha` 0.05 and `--retention-tol` $\tau = 0.9$):

| Condition | Label |
|---|---|
| $q \ge \alpha$ | `no-significant-change` |
| $q < \alpha$ and $R > 1/\tau$ | `improved` |
| $q < \alpha$ and $\tau \le R \le 1/\tau$ | `retained` |
| $q < \alpha$ and $R < \tau$ | `degraded` |
| no paired test available | `not-tested` |

If the seed vectors of two arms do not match (different `seed_base` or `M_cycles`), the contrast is marked `paired = False`, a warning is printed, and the paired statistics are left blank. Use the [pairing guard](#cycle-pairing) to avoid this.

Outputs (in `RESULTS/blinding_analysis/<run_id>/`):

| File | Content |
|---|---|
| `contrast_table.csv` | Every contrast with all statistics above |
| `main_table_native_vs_generic.csv` | Compact native vs generic table for the main text |
| `blinding_condition_panels.png` | Median test MAE by condition, one panel per property, one line per model |

---

## Out-of-range and reject summary

`oor_extract.py` (or `./run_blinding.sh oor`) reads every blinded cell and writes `oor_summary.csv` into the run folder:

$$
\text{OOR rate} = \frac{\sum_k n_{\text{OOR},k}}{N_\text{test}\, n_\text{cycles}}, \qquad
\text{reject rate} = \frac{\sum_k n_{\text{rej},k}}{N_\text{test}\, n_\text{cycles}}
$$

where $N_\text{test}$ is read from each cell's `testing_summary.txt`. An **out-of-range** answer was parsed but lies outside the transform's window (only possible with `linear` or `tanh`). A **reject** is an answer with no parsable number. With `zscore`, OOR counts are zero by construction.

---

## Output layout

```
RESULTS/
├── blinding_runs/<run_id>/
│   ├── manifest.json                  # settings of the first sweep
│   ├── manifest_<timestamp>.json      # one per later append
│   ├── summary_blinded.csv            # one row per (model, property, condition)
│   ├── recoder_map.json               # agnostic only
│   ├── oor_summary.csv                # from oor_extract.py
│   └── <model_slug>_<PROPERTY>_<condition>/
│       ├── cycles_blinded.npz
│       ├── blinding_spec.json         # transform, recoder seed, label format
│       └── testing_summary.txt
└── blinding_analysis/<run_id>/        # from blinding_analysis.py
```

`cycles_blinded.npz` holds per-cycle arrays of length $M_c$ (`MAE_test`, `RMSE_test`, `MSE_test`, `R2_test`, `Acc_test`, `F1_test`, `NRMSE_range_test`, `MAE_test_trans`, `RMSE_test_trans`, `R2_test_trans`, `n_rejects_test`, `n_out_of_range_test`, `seed_per_cycle`). It also holds validation scalars (`MAE_val`, `RMSE_val`, `R2_val`, `Acc_val`, `F1_val`, `MAE_val_trans`, `NRMSE_range_val`, `n_rejects_val`, `n_out_of_range_val`), dataset statistics, wall times, and the `condition` and `transform` names. The `*_trans` metrics are in transformed units and are a diagnostic only; they are not comparable across properties.

---

## Reproducibility notes

- **Label formatting.** Demonstration labels are written with `--value-fmt` (default three decimals) in every condition, including native. The blinding native arm is therefore format-matched to the blinded arms, but not byte-identical to main-study prompts, which use the raw CSV text. Compare blinded arms with the native arm of the **same** blinding run.
- **Validation.** As in the main study, the best test cycle's demonstrations are evaluated on the independent validation set. The transform is refitted on those demonstrations first.
- **Temperature.** The default 0.9 matches the main study and the published runs. Use `--temperature` (or `TEMPERATURE=`) to change it; the value is recorded in the manifest and in each `testing_summary.txt`.
- **Seeds.** Recorded in the manifest: `seed_base` (data and demonstrations), `recoder_seed` (agnostic codes). The statistics have their own `--seed`.
