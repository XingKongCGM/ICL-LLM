#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 14:17:09 2025

@author: Carlos Gonzalez
Material Data Division,
National Institute of Standatds and Technology,
Gaithersburg, MD. 20889


LOCAL-ONLY build. The remote NIST-RCHAT (OpenAI-compatible) backend has
been removed; all inference now runs through the local MLX backend only.
Model choices are a single contiguous 1..7 set (see LOCAL_MODEL_MAP).

optimal_samples location: run_cell writes, and run_cell_predict /
run_cell_forecast read, the best-cycle demonstration CSV at the path
supplied in data_paths["optimal_samples"] (resolved by the batch driver
from each property's `optimal_dir` in properties.yaml). If that key is
absent, the legacy per-cell location out_dir/optimal_samples.csv is used.
"""

# =============================================================================
#                 Import necessary libraries
# =============================================================================
import json
import os
import sys
import time
import datetime
import numpy as np
import math
import pandas as pd
import csv
import re
import random
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

# =============================================================================
#              Backend Configuration  (LOCAL MLX only)
# =============================================================================
# This build runs a set of mlx-community quantized models directly on
# Apple Silicon through mlx_lm. Every downstream function
# (Cross_Validation, k_shot_test, k_shot_predict, and the metric/
# plotting/summary machinery) calls exactly one primitive:
# send_query(prompt, model, temperature=temperature). That primitive dispatches to the local
# MLX generator.
#
# Memory-and-load discipline for local models: weights are loaded exactly
# once, at startup, in select_llm_model() / select_llm_model_by_choice().
# A run with M_c cycles and N_t test molecules issues N = M_c * N_t
# inferences; at ~30 s per model load, avoiding the reload saves many
# hours per run compared to a naive per-query load.
# =============================================================================

# Local mlx-community model paths.  Keep these in sync with
# PFAS_MP_Fine-tune_MLX_CV.py:select_model() so an ICL run and a fine-tune
# run of the same option k target byte-identical weights, giving direct
# A/B comparability across the two workflows.
LOCAL_MODEL_MAP = {
    1: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    2: "mlx-community/Llama-3.3-70B-Instruct-4bit",
    3: "mlx-community/Nemotron-Mini-4B-Instruct-bf16-mlx",
    4: "mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-4bit",
    5: "mlx-community/gemma-4-31B-it-OptiQ-4bit",
    6: "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
    7: "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
}

# ----------------------------------------------------------------------
# Sampling temperature: single editable default for ALL phases.
# Testing / K-Fold / LOO / Predict use this value when no explicit
# temperature is passed. Set to 0.9 to reproduce the published runs.
# Forecast sweeps its own --temperatures list and is unaffected.
# ----------------------------------------------------------------------
DEFAULT_TEMPERATURE = 0.9



# Module-level backend state.  Set by select_llm_model(); read by
# send_query() and _run_local_generate() during the CV/Testing/Predict
# loops.  Grouped here so a debugger session shows the full picture at
# a glance.
_LOCAL_MODEL     = None    # loaded mlx_lm model object
_LOCAL_TOKENIZER = None    # matching tokenizer with chat_template
_LOCAL_MAX_TOKENS = 256    # generation budget per query (bumped to 512
                           # for reasoning models like Gemma-4 OptiQ)

# Sentinel used by verify-mlx probe.  When set, select_llm_model()
# imports mlx_lm eagerly BEFORE prompting for a model choice, so that a
# missing dependency fails fast instead of after the user has typed
# through several prompts.
_EAGER_MLX_CHECK = True


def _mlx_available():
    """
    Cheap import probe.  Returns (True, load_fn) if mlx_lm is importable
    on this machine, (False, exception) otherwise.  Used both by the
    startup diagnostic and inside select_llm_model() so the error path
    is uniform.
    """
    try:
        from mlx_lm import load as _mlx_load
        return True, _mlx_load
    except ImportError as e:
        return False, e


# Pattern for the memory-budget warning that mlx_lm.generate emits on
# every call when the model's memory footprint is close to the
# iogpu.wired_limit_mb ceiling.  This is informational, not a failure,
# but at 750+ occurrences per run it drowns out real content.
_MLX_MEMORY_WARNING_PATTERN = "Generating with a model that requires"


class _suppress_mlx_memory_warning:
    """
    Context manager that filters a specific noisy warning line from
    mlx_lm.generate's stderr output while letting all other stderr
    content pass through unchanged.

    Rationale.  mlx_lm.generate writes the memory-budget warning via
    print(..., file=sys.stderr) at the Python level (not via the
    warnings module), so warnings.simplefilter() is ineffective.  We
    instead replace sys.stderr with a line-buffered filter that
    inspects each written line and drops it if it matches the warning
    signature.

    Safe under exceptions: sys.stderr is always restored, even if the
    wrapped code raises.  Idempotent: nesting this context manager is
    fine (the outer filter sees only the inner's post-filter output).

    Does NOT suppress:
      * Any stderr output that does not contain the pattern
      * mlx_lm tracebacks (they contain "Traceback", not the pattern)
      * The transformers rope_parameters warning (different text)
    """

    def __enter__(self):
        self._real_stderr = sys.stderr
        self._buffer = ""
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush any residual buffered content (a partial line that
        # never got a newline).
        if self._buffer:
            self._real_stderr.write(self._buffer)
            self._buffer = ""
        sys.stderr = self._real_stderr
        return False   # do not suppress exceptions

    def write(self, s):
        # Accumulate into a line-buffered stream.  Emit each complete
        # line that does NOT match the memory-warning pattern.
        self._buffer += s
        while "\n" in self._buffer:
            line, _, self._buffer = self._buffer.partition("\n")
            if _MLX_MEMORY_WARNING_PATTERN not in line:
                self._real_stderr.write(line + "\n")

    def flush(self):
        self._real_stderr.flush()

    def isatty(self):
        return self._real_stderr.isatty()

    def fileno(self):
        # mlx_lm might request the raw fd for low-level writes.
        # Return the real stderr fd; those writes will bypass the
        # filter, which is safer than pretending to be an fd we
        # don't own.
        return self._real_stderr.fileno()


def _apply_llama4_type_fixes(cfg):
    """
    Walk a Llama-4 config dict recursively and coerce known problematic
    fields to their correct types.  Modifies cfg in place; returns
    a list of (dotted-path, old_repr, new_repr) tuples for logging.

    Llama-4 configs are nested: the mlx-community-shipped config.json
    has the form

        {
          "model_type": "llama4",
          "attn_temperature_tuning": 4,        <-- may or may not be here
          "text_config": {                     <-- but is ALWAYS here
            "attn_temperature_tuning": 4,
            "rope_theta": 500000,
            "rope_scaling": {...},
            ...
          },
          "vision_config": {...}
        }

    An earlier version of this patcher only fixed the top-level
    occurrences, silently missing the nested ones inside text_config /
    vision_config -- which is where mlx_lm's Llama4Config dataclass
    actually reads them.  This visitor walks all dict nodes so no
    occurrence is missed regardless of the config's exact shape.
    """
    fixes = []

    def _visit(node, path):
        if not isinstance(node, dict):
            return

        # attn_temperature_tuning: int -> bool
        # NOTE: bool is a subclass of int in Python, so we explicitly
        # check "not isinstance(v, bool)" (True is a bool AND an int;
        # 4 is an int but NOT a bool).
        if "attn_temperature_tuning" in node:
            v = node["attn_temperature_tuning"]
            if not isinstance(v, bool):
                new = bool(v)
                node["attn_temperature_tuning"] = new
                fixes.append((f"{path}attn_temperature_tuning",
                              f"{v} (int)", f"{new} (bool)"))

        # rope_theta: int -> float
        if "rope_theta" in node:
            v = node["rope_theta"]
            if isinstance(v, int) and not isinstance(v, bool):
                node["rope_theta"] = float(v)
                fixes.append((f"{path}rope_theta",
                              f"{v} (int)", f"{float(v)} (float)"))

        # rope_scaling: nested numeric fields
        if isinstance(node.get("rope_scaling"), dict):
            rs = node["rope_scaling"]
            for k in ("factor", "high_freq_factor", "low_freq_factor",
                      "beta_fast", "beta_slow"):
                if k in rs:
                    v = rs[k]
                    if isinstance(v, int) and not isinstance(v, bool):
                        rs[k] = float(v)
                        fixes.append((f"{path}rope_scaling.{k}",
                                      f"{v} (int)", f"{float(v)} (float)"))

        # Recurse into known nested config blocks.  Llama-4 uses
        # text_config and vision_config; future variants might add
        # more, but visiting only known keys keeps us from descending
        # into unrelated large nested structures like tokenizer_config.
        for nested_key in ("text_config", "vision_config",
                           "audio_config", "encoder_config",
                           "decoder_config"):
            if nested_key in node and isinstance(node[nested_key], dict):
                _visit(node[nested_key], f"{path}{nested_key}.")

    _visit(cfg, "")
    return fixes


def _patch_llama4_config(model_id):
    """
    Return (path_or_repo_id, model_config_overrides) for mlx_lm.load().

    The Llama-4 mlx-community configs have two known type mismatches
    (attn_temperature_tuning: int should be bool; rope_theta: int
    should be float) that mlx_lm's StrictDataclass rejects.  These
    fields are NESTED inside text_config, not at the top level of
    config.json.

    Two-layer defense:

      Layer 1 (file):
        Downloads (or locates cached) config.json, walks it recursively,
        coerces every occurrence of the problem fields at every nesting
        level.  On HuggingFace cache symlinks, breaks the symlink
        BEFORE writing so the patch becomes a per-snapshot override
        that HF's cache-integrity check cannot revert.

      Layer 2 (in-memory overrides):
        Returns a nested overrides dict for mlx_lm.load(model_config=...).
        Provides a fallback if the file patch fails (permissions,
        network cache quirks, ...).  Note that mlx_lm's model_config
        is a shallow dict.update() overlay; for nested overrides to
        work we replicate the ENTIRE text_config with our fixes on
        top, preserving any other fields.

    For non-Llama-4 models, returns (model_id, {}) unchanged.
    Idempotent: running twice on an already-patched config is a no-op.
    """
    if "llama-4" not in model_id.lower() and "llama4" not in model_id.lower():
        return model_id, {}

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(f"[patch]  huggingface_hub not available.  Install with "
              f"`pip install huggingface_hub`.  Cannot patch.")
        return model_id, {}

    print(f"[patch]  Ensuring {model_id} config is patched...")

    try:
        local_dir = snapshot_download(
            repo_id=model_id, local_files_only=False
        )
    except Exception as e:
        print(f"[patch]  snapshot_download failed: "
              f"{type(e).__name__}: {e}")
        return model_id, {}

    config_path = os.path.join(local_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"[patch]  No config.json at {config_path}; skipping.")
        return local_dir, {}

    import json
    try:
        with open(config_path, "r") as fh:
            cfg = json.load(fh)
    except Exception as e:
        print(f"[patch]  Cannot read {config_path}: {e}")
        return local_dir, {}

    # Walk the tree and apply fixes in place.  This is Layer 1's
    # in-memory state, which we'll write to disk after handling the
    # symlink.
    fixes = _apply_llama4_type_fixes(cfg)

    # Build Layer-2 overrides.  Because mlx_lm.load(model_config=...)
    # is a shallow dict.update() on the parsed config, nested overrides
    # must include the ENTIRE nested subtree with our fixes applied.
    # We copy the (now-fixed) text_config and vision_config from cfg
    # into overrides so they replace the mlx_lm-loaded versions
    # verbatim.
    overrides = {}
    if fixes:
        # Copy top-level scalar fixes
        for name in ("attn_temperature_tuning", "rope_theta"):
            if name in cfg:
                overrides[name] = cfg[name]
        # Copy the FIXED nested configs whole so shallow-merge preserves
        # every field, not just the ones we changed
        for nested_key in ("text_config", "vision_config"):
            if nested_key in cfg and isinstance(cfg[nested_key], dict):
                overrides[nested_key] = cfg[nested_key]

    # Write the patched config back to disk (Layer 1).  Handle the
    # symlink case: break it first so writes create a per-snapshot
    # file and don't trigger an HF cache-integrity re-download.
    if fixes:
        try:
            if os.path.islink(config_path):
                os.unlink(config_path)
                print(f"[patch]  Broke symlink at {config_path} "
                      f"(was pointing at HF shared blob)")
            with open(config_path, "w") as fh:
                json.dump(cfg, fh, indent=2)
            print(f"[patch]  Wrote patched config to {config_path}")
            for dotted_path, old_repr, new_repr in fixes:
                print(f"[patch]     {dotted_path}: {old_repr} -> {new_repr}")
            # Verify at every nesting level that the write stuck
            with open(config_path, "r") as fh:
                verify = json.load(fh)
            _verify_llama4_fixes(verify)
        except Exception as e:
            print(f"[patch]  File patch failed ({type(e).__name__}: "
                  f"{e}), relying on Layer-2 overrides.")
    else:
        print(f"[patch]  Config already valid at every nesting level.")

    return local_dir, overrides


def _verify_llama4_fixes(cfg):
    """
    Verify (post-write) that no problem fields remain with the wrong
    type at any nesting level.  Emits a WARN line if verification
    fails so we know to trust Layer 2 rather than Layer 1.
    """
    remaining = _apply_llama4_type_fixes(dict(cfg) if isinstance(cfg, dict)
                                          else cfg)
    # Note: _apply_llama4_type_fixes mutates the dict in place, so we
    # copied first.  If the returned list is non-empty, the disk file
    # STILL has problem fields.
    if remaining:
        print(f"[patch]  WARNING: file still has {len(remaining)} bad "
              f"field(s) after write.  This can happen if the config "
              f"path resolves to a hardlinked / immutable file.  "
              f"Layer-2 model_config overrides will be used instead.")
        for dotted_path, old_repr, _ in remaining:
            print(f"[patch]     still bad: {dotted_path} = {old_repr}")


def select_llm_model():
    """
    Interactive model picker.  Returns the model identifier string that
    gets passed downstream as `model` in every send_query call, matching
    the signature the original script used.

    Side effects on module state (all choices are LOCAL):
        _LOCAL_MODEL, _LOCAL_TOKENIZER = mlx_lm.load(model_id)
        _LOCAL_MAX_TOKENS = 256 or 512 (Gemma-4 OptiQ)

    Chose the "prompt-first, load-later" pattern (rather than loading a
    default model at import time) so that invalid mlx-community repo ids
    surface immediately rather than after a long CV setup.
    """
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_MAX_TOKENS

    print("\nSelect LLM Model:")
    for i, path in LOCAL_MODEL_MAP.items():
        friendly = path.split("/")[-1]
        print(f"  {i}: {friendly}   [local MLX]")

    # Build the valid-choice set dynamically so adding a model to
    # LOCAL_MODEL_MAP automatically shows up in the prompt without
    # requiring an edit here.
    valid_choices = sorted(LOCAL_MODEL_MAP.keys())
    valid_str = ", ".join(str(c) for c in valid_choices)

    while True:
        raw = input(f"Enter choice ({valid_str}): ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print(f"  '{raw}' is not an integer.  Try again.")
            continue
        if choice in valid_choices:
            break
        print(f"  {choice} is not in the allowed set {{{valid_str}}}.  Try again.")

    model_id = LOCAL_MODEL_MAP[choice]

    ok, load_fn_or_err = _mlx_available()
    if not ok:
        print("\nERROR: mlx_lm is not installed in this Python environment.")
        print(f"       Underlying import error: {load_fn_or_err}")
        print("       Install with:  pip install mlx-lm")
        print("       (Requires Apple Silicon; not available on Intel Macs.)")
        sys.exit(1)

    print(f"\n[backend]  LOCAL (mlx_lm)")
    print(f"[model  ]  {model_id}")
    print(f"[note   ]  Loading weights once; typical times: "
          f"8B ~5 s, 24B ~15 s, 70B ~30 s.")

    # Model-family-specific tokenizer fixes.  Mistral's default regex
    # pattern in transformers is known to produce incorrect tokenization
    # for the Mistral-Small-3.1-24B family (transformers emits a
    # deprecation warning pointing at
    # huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/
    #   discussions/84).  With the broken regex the model can produce
    # garbled output that fails every downstream numeric extraction,
    # leaving predicted=[] and crashing mean_squared_error.  The fix is
    # a single kwarg on AutoTokenizer.from_pretrained, propagated
    # through mlx_lm.load's tokenizer_config.
    tokenizer_config = {}
    if "mistral" in model_id.lower():
        tokenizer_config["fix_mistral_regex"] = True
        print(f"[note   ]  Applying fix_mistral_regex=True to correct "
              f"the known Mistral tokenizer regex pattern.")

    # Pre-patch Llama-4 config.json for known type mismatches (no-op
    # on non-Llama-4 models).  Returns (path, model_config_overrides)
    # tuple; the overrides go into mlx_lm.load's model_config kwarg.
    model_path, model_config_overrides = _patch_llama4_config(model_id)

    # Build load kwargs.  model_config overrides are the RELIABLE
    # fix for Llama-4 type mismatches: they win over whatever is in
    # config.json, whether or not our file patch stuck.
    load_kwargs = {}
    if tokenizer_config:
        load_kwargs["tokenizer_config"] = tokenizer_config
    if model_config_overrides:
        load_kwargs["model_config"] = model_config_overrides

    t0 = time.time()
    try:
        _LOCAL_MODEL, _LOCAL_TOKENIZER = load_fn_or_err(
            model_path, **load_kwargs
        )
    except TypeError:
        # Older mlx_lm: tokenizer_config and/or model_config kwargs
        # not supported.  Retry progressively less aggressively.
        try:
            # Drop model_config first (rare kwarg), keep tokenizer_config
            _LOCAL_MODEL, _LOCAL_TOKENIZER = load_fn_or_err(
                model_path, tokenizer_config=tokenizer_config
            )
            if model_config_overrides:
                print(f"[warn   ]  Your mlx_lm version does not accept "
                      f"model_config; Llama-4 attn_temperature_tuning "
                      f"override was NOT applied.  If load succeeds, "
                      f"the file patch was sufficient.  If not, upgrade: "
                      f"pip install --upgrade mlx-lm")
        except TypeError:
            _LOCAL_MODEL, _LOCAL_TOKENIZER = load_fn_or_err(model_path)
            if tokenizer_config:
                print(f"[warn   ]  Your mlx_lm version does not accept "
                      f"tokenizer_config; Mistral regex may remain broken. "
                      f"Consider upgrading:  pip install --upgrade mlx-lm")
        except Exception as e2:
            print(f"\nERROR: could not load {model_id}.")
            print(f"       {type(e2).__name__}: {e2}")
            print(f"       Verify the mlx-community repo id and your "
                  f"HuggingFace access.")
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR: could not load {model_id}.")
        print(f"       {type(e).__name__}: {e}")
        print(f"       Verify the mlx-community repo id and your "
              f"HuggingFace access.  For gated repos, run "
              f"`huggingface-cli login`.")
        sys.exit(1)
    dt = time.time() - t0
    print(f"[loaded ]  in {dt:.1f} s")

    # Reasoning-family models (Gemma 4 OptiQ) emit a hidden chain-of-
    # thought before the answer; give them extra headroom.
    if "gemma" in model_id.lower() and "optiq" in model_id.lower():
        _LOCAL_MAX_TOKENS = 512
        print(f"[note   ]  Reasoning model detected; per-query "
              f"max_tokens = {_LOCAL_MAX_TOKENS}")
    else:
        print(f"[note   ]  Per-query max_tokens = {_LOCAL_MAX_TOKENS}")
    print()
    return model_id


# Counter for the diagnostic print in _run_local_generate.  We show the
# raw model output for the first _DIAGNOSTIC_MAX queries of the run so
# tokenizer / prompt / max_tokens problems are visible immediately
# rather than after N samples have all silently been rejected.
_DIAGNOSTIC_COUNT = 0
_DIAGNOSTIC_MAX   = 3

# Fallback chat templates used when tokenizer.chat_template is missing
# (a known issue in some mlx-community quantized conversions -- e.g.
# Mistral-Small-3.1-24B-Instruct-2503-4bit drops the chat_template
# field from tokenizer_config.json during quantization, so
# tokenizer.apply_chat_template raises 'chat_template is not set').
#
# The templates below reproduce the essential single-turn structure
# each model family expects.  They omit optional system-message
# handling because RCHAT_PChem_Properties.py always sends exactly one
# user turn with no system prompt.
_MODEL_FAMILY_TEMPLATES = {
    "mistral":  "<s>[INST] {content} [/INST]",
    "llama-3":  ("<|begin_of_text|><|start_header_id|>user"
                 "<|end_header_id|>\n\n{content}<|eot_id|>"
                 "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "nemotron": ("<|begin_of_text|><|start_header_id|>user"
                 "<|end_header_id|>\n\n{content}<|eot_id|>"
                 "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "gemma":    ("<start_of_turn>user\n{content}<end_of_turn>\n"
                 "<start_of_turn>model\n"),
    "generic":  "User: {content}\nAssistant: ",
}

# One-shot flag so the fallback message prints once per run, not once
# per query.
_CHAT_TEMPLATE_FALLBACK_WARNED = False


def _detect_model_family(model_id):
    """Return one of {'mistral', 'llama-3', 'nemotron', 'gemma',
    'generic'} based on substring matching in the model repo id."""
    m = model_id.lower()
    if "mistral" in m:
        return "mistral"
    if "nemotron" in m:
        return "nemotron"
    if "llama-3" in m or "llama-4" in m or "llama3" in m or "llama4" in m:
        return "llama-3"
    if "gemma" in m:
        return "gemma"
    return "generic"


def _split_priming(prompt):
    """
    RCHAT-style prompts end with an assistant-priming pattern:
    "\n<PropertyName>: " (colon + space, no value), meant to be the
    first tokens of the assistant's *response*, not part of the user
    turn.  Under a local chat template (either the tokenizer's own or
    our fallback), the naive approach wraps the ENTIRE prompt --
    including the priming -- inside the user turn, e.g. Mistral produces

        <s>[INST] ...demos... Molecule: S_query
                  logKow:  [/INST]

    At non-zero temperature the model occasionally interprets the
    trailing "logKow: " as an incomplete user request and responds
    with prose like "It seems like you forgot to include the
    molecule..." rather than a numeric value.

    This helper splits the priming out so the caller can place it
    AFTER the assistant marker, where it belongs:

        <s>[INST] ...demos... Molecule: S_query [/INST]logKow:

    Returns (body, priming). If no priming pattern is detected,
    priming == "" and body == prompt unchanged.
    """
    m = re.search(r'\n([A-Za-z_][A-Za-z0-9_ ]*)\s*:\s*$', prompt)
    if m:
        body = prompt[:m.start()]
        priming = m.group().lstrip('\n')
        return body, priming
    return prompt, ""


def _apply_chat_template_or_fallback(tokenizer, prompt, model_id):
    """
    Apply the tokenizer's chat template if it has one, else use a
    hardcoded family-specific fallback.  Assistant-priming patterns
    at the tail of `prompt` (e.g. "\nlogKow: ") are moved OUTSIDE the
    user turn so downstream models receive a well-formed [INST]...[/INST]
    (or Llama-3 / Gemma equivalent) with the priming as an assistant
    prefill.

    Two paths, both applied to the priming-stripped BODY:
      1. tokenizer.apply_chat_template(body)   [preferred]
      2. hardcoded family template             [fallback]
    Priming is then appended to whichever path produced a result.

    Only falls back on the specific "no chat template" error; any
    other exception is re-raised so we don't mask unrelated bugs.
    """
    global _CHAT_TEMPLATE_FALLBACK_WARNED

    # Peel off the assistant priming before wrapping the body in any
    # chat template.  This is the key fix for the "you forgot the
    # molecule" rejection pattern.
    body, priming = _split_priming(prompt)

    messages = [{"role": "user", "content": body}]
    try:
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        # Only fall back on the specific "no chat template" error.
        if "chat_template" not in str(e):
            raise
        family = _detect_model_family(model_id)
        template = _MODEL_FAMILY_TEMPLATES[family]
        if not _CHAT_TEMPLATE_FALLBACK_WARNED:
            _CHAT_TEMPLATE_FALLBACK_WARNED = True
            print(f"\n[note   ]  Tokenizer for '{model_id}' has no "
                  f"chat_template attribute.")
            print(f"           This is a known issue with some "
                  f"mlx-community quantized conversions that strip "
                  f"the tokenizer_config's chat_template field.")
            print(f"           Falling back to hardcoded {family.upper()} "
                  f"single-turn template.  Behavior is equivalent for "
                  f"the single-user-turn queries this script sends.")
            print(f"           (Warning suppressed for the rest of "
                  f"this run.)\n")
        templated = template.format(content=body)

    # Append the assistant-priming string AFTER the assistant-turn
    # marker so the model treats it as its own response continuation
    # rather than as user text ending in "logKow: ".
    return templated + priming


# ===========================================================================
# TEMPERATURE-AWARE send_query and _run_local_generate (LOCAL-only)
# ---------------------------------------------------------------------------
# The `temperature` argument defaults to None, in which case the previous
# hardcoded 0.9 is used, so every existing caller (Cross_Validation,
# k_shot_test, k_shot_predict, run_cell, etc.) behaves exactly as before.
# Only the forecast path passes an explicit value.
# ===========================================================================

def _run_local_generate(prompt, model, temperature=None):
    """
    One inference against the loaded local MLX model.

    temperature : float or None
        Sampling temperature. If None, uses 0.9 (the historical default
        that keeps Mc-cycle stochastic variance consistent). The forecast
        mode passes explicit values so it can sweep T.
    """
    global _DIAGNOSTIC_COUNT
    from mlx_lm import generate as _mlx_generate

    temp = DEFAULT_TEMPERATURE if temperature is None else float(temperature)

    templated = _apply_chat_template_or_fallback(
        _LOCAL_TOKENIZER, prompt, model
    )

    if "gemma" in model.lower() and "optiq" in model.lower():
        templated = templated.replace("<|think|>", "")

    with _suppress_mlx_memory_warning():
        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temp)
            response = _mlx_generate(
                _LOCAL_MODEL, _LOCAL_TOKENIZER,
                prompt=templated,
                max_tokens=_LOCAL_MAX_TOKENS,
                sampler=sampler,
                verbose=False,
            )
        except (ImportError, TypeError):
            # Older mlx_lm: sample_utils absent, or generate() doesn't
            # accept a sampler kwarg. Fall back to the deprecated
            # top-level `temp` argument.
            response = _mlx_generate(
                _LOCAL_MODEL, _LOCAL_TOKENIZER,
                prompt=templated,
                max_tokens=_LOCAL_MAX_TOKENS,
                temp=temp,
                verbose=False,
            )

    if response is None:
        return np.nan
    stripped = str(response).strip()

    if _DIAGNOSTIC_COUNT < _DIAGNOSTIC_MAX:
        _DIAGNOSTIC_COUNT += 1
        preview = stripped if len(stripped) <= 200 else stripped[:200] + "..."
        print(f"\n[DIAG {_DIAGNOSTIC_COUNT}/{_DIAGNOSTIC_MAX}] Raw local "
              f"response ({len(stripped)} chars, T={temp}):")
        print(f"    {preview!r}")
        parsed = extract_numeric_prediction(stripped)
        print(f"    Parsed as number: {parsed}")
        if _DIAGNOSTIC_COUNT == _DIAGNOSTIC_MAX:
            print(f"    (Further response diagnostics suppressed.)")

    return stripped if stripped else np.nan


# =============================================================================
#              Send the Query to the LLM Model  (LOCAL-only dispatcher)
# =============================================================================
def send_query(prompt, model, temperature=None):
    """
    Query wrapper. All inference runs through the local MLX backend.

    temperature : float or None
        If None, uses 0.9 (historical default) so all existing callers
        are unchanged. The forecast mode passes an explicit T to sweep
        temperatures.
    """
    temp = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
    try:
        return _run_local_generate(prompt, model, temperature=temp)
    except Exception as e:
        print(f"Error in output: {e}")
        return np.nan


# =============================================================================
#                          Utility Functions 
# =============================================================================
def read_jsonl_file(dataset_file):
    with open(dataset_file, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def read_csv_file(dataset_file):
    """
    Reads a CSV file and returns its contents as a list of dictionaries.

    Args:
        dataset_file (str): Path to the CSV file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the CSV file.
    """
    with open(dataset_file, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data


def whole_dataset_stat(df1, Key, Save):  
    df1[Key] = pd.to_numeric(df1[Key])  # Convert 'logCMC' to numeric
    median_value = df1[Key].median()
    mean_value = df1[Key].mean()
    std_value = df1[Key].std()
    min_value = df1[Key].min()
    max_value = df1[Key].max()
    df1["binary_class"] = (df1[Key] > median_value).astype(int)
    df1["five_class"] = pd.qcut(df1[Key], q=5, labels=False)
    counts = df1["binary_class"].value_counts()
    negatives = int(counts[1])
    positives = int(counts[0])
    Neg_percentage = float(negatives / (negatives + positives)) * 100.0
    Pos_percentage = float(positives / (negatives + positives)) * 100.0
    if Save:
        filename = Key + "_classified.csv"
        df1.to_csv(filename, index=False)
        filename = Key + "_classified.jsonl"
        with open(filename, "w") as f_out:
            for _, row in df1.iterrows():
                json.dump({
                    "prompt": row["prompt"],
                    "actual_value": row["actual_value"],
                    "binary_class": int(row["binary_class"]),
                    "five_class": int(row["five_class"])
                }, f_out)
                f_out.write("\n")
    print(f"Median= {median_value:.6f}, Mean= {mean_value:.6f}, Standard_Deviation= {std_value:.6f}", flush=True)
    print(f"Minimum_Value= {min_value:.6f}, Maximum_Value= {max_value:.6f}", flush = True)
    print(f"Number of Negatives= {negatives} ({Neg_percentage:.2f}%), Number of Positives= {positives} ({Pos_percentage:.2f}%)\n", flush = True)
    return median_value, mean_value, std_value, min_value, max_value


def get_accuracy(predicted, actual, median_value):
    p_binary = (predicted > median_value).astype(int)
    a_binary = (actual > median_value).astype(int)
    predicted_binary = p_binary
    actual_binary = a_binary
    tn = np.sum((predicted_binary == 0) & (actual_binary == 0))
    fn = np.sum((predicted_binary == 0) & (actual_binary == 1))
    tp = np.sum((predicted_binary == 1) & (actual_binary == 1))
    fp = np.sum((predicted_binary == 1) & (actual_binary == 0))
    accuracy = (tn + tp) / (tn + fn + tp + fp)
    return accuracy * 100.0


def get_f1_stats(predicted, actual, num_classes=5):
    F1_score = np.zeros(num_classes)
    p_five = pd.qcut(predicted, q=num_classes, labels=False, duplicates='drop')
    t_five = pd.qcut(actual, q=num_classes, labels=False, duplicates='drop')
    y_pred = np.array(p_five)
    y_true = np.array(t_five)
    F1_macro = 0.0
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        Precision = tp / (tp + fp) if (tp + fp) else 0.0
        Recall = tp / (tp + fn) if (tp + fn) else 0.0
        F1_score[c] = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) else 0.0
        F1_macro += F1_score[c]
    return F1_score, F1_macro / float(num_classes)

def kfold_indices(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds


def extract_numeric_prediction(raw):
    """
    Returns float if a numeric value can be extracted, else None.
    Handles direct numeric types and string responses.
    """
    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        # Avoid NaN/infinite silently propagating if desired
        try:
            x = float(raw)
            if math.isnan(x) or math.isinf(x):
                return None
            return x
        except Exception:
            return None

    s = str(raw)
    matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    if not matches:
        return None

    try:
        return float(matches[-1])
    except ValueError:
        return None

def Cross_Validation(training_samples, median, k_shot=5, kfold=5, N_total=6, prompt_template="", feature="", output="", Rate= 90, Memory="YES", model="", temperature=None):
    MSE = RMSE = MAE = R_Squared = accuracy = F1_macro = 0.0
    n_rejects = 0
    fold_indices = kfold_indices(training_samples, kfold)
    Tau = 60.0/Rate

    for train_indices, test_indices in fold_indices:
        k_shot_examples = [training_samples[i] for i in train_indices]
        test_d = [training_samples[i] for i in test_indices]
        predicted = []
        actual = []
        conversation_history = ""
        for data in test_d:
            if (Memory == "YES"):
                prompt = prompt_template + conversation_history
            elif (Memory == "NO"):
                prompt = prompt_template
            for example in k_shot_examples[:k_shot]:  # Limit to k_shot examples
                prompt += f"{feature}: {example[feature]}\n{output}: {example[output]}\n\n"
            prompt += f"{feature}: {data[feature]}\n{output}: "

            raw_value = send_query(prompt, model, temperature=temperature)
            value = extract_numeric_prediction(raw_value)

            if value is not None:
                predicted.append(float(value))
                actual.append(float(data[output]))

                # Update conversation history only when Memory is enabled
                if Memory == "YES":
                    conversation_history += f"{feature}: {data[feature]}\n{output}: {value}\n\n"

            else:
                print(f"Skipping sample {data[feature]} due to invalid prediction: {value}")
                n_rejects += 1

            time.sleep(Tau)

# Filter out NaN values from predicted and actual
        predicted = np.array(predicted)
        actual = np.array(actual)
        valid_indices = ~np.isnan(predicted)
        predicted = predicted[valid_indices]
        actual = actual[valid_indices]

        if len(predicted) > 0:        
            MSE_fold = mean_squared_error(actual, predicted)
            RMSE_fold = np.sqrt(MSE_fold)
            MAE_fold = mean_absolute_error(actual, predicted)
            if len(predicted) > 3: R2_fold = r2_score(actual, predicted)
            acc_fold = get_accuracy(predicted, actual, median)
            _, F1_fold = get_f1_stats(predicted, actual)
            MSE += MSE_fold; RMSE += RMSE_fold; MAE += MAE_fold
            if len(predicted) > 3: R_Squared += R2_fold 
            accuracy += acc_fold
            F1_macro += F1_fold
        else:
            print("No valid predictions for this fold.")

    kfold = len(fold_indices)
    if kfold > 0:    
        return MSE/kfold, RMSE/kfold, MAE/kfold, R_Squared/kfold, accuracy/kfold, F1_macro/kfold, n_rejects
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def k_shot_test(training_samples, test_samples, median, prompt_template,  feature, output, Rate, Memory, model, temperature=None):
    MSE = RMSE = MAE = R_Squared = accuracy = F1_macro = 0.0
    n_rejects = 0
    Tau = 60.0/Rate

    training_length = len(training_samples)
    test_length = len(test_samples)

    predicted = []
    actual = []
    conversation_history = ""

    for nrow in range(test_length):
        if (Memory == "YES"):
            prompt = prompt_template + conversation_history
        elif (Memory == "NO"):
            prompt = prompt_template
        else:
            prompt = prompt_template

        for krow in range(training_length):
            prompt += f"{feature}: {training_samples[krow][feature]}\n{output}: {training_samples[krow][output]}\n\n"

        prompt += f"{feature}: {test_samples[nrow][feature]}\n{output}: "

            
        raw_value = send_query(prompt, model, temperature=temperature)
        value = extract_numeric_prediction(raw_value)

        if value is not None:
            predicted.append(float(value))
            actual.append(float(test_samples[nrow][output]))

            # Update conversation history only when Memory is enabled
            if Memory == "YES":
                conversation_history += f"{feature}: {test_samples[nrow][feature]}\n{output}: {value}\n\n"

        else:
            print(f"Skipping sample {test_samples[nrow]} due to invalid prediction (raw response: {raw_value})")
            n_rejects += 1

        time.sleep(Tau)

# Convert to numpy arrays
    predicted = np.array(predicted)
    actual = np.array(actual)

    # Empty-predictions guard.  If every model call was rejected (e.g.
    # tokenizer broken, model outputting prose without numbers,
    # max_tokens too small), predicted is length 0 and mean_squared_error
    # raises ValueError.  Trap that here and print a clear diagnostic
    # instead of the sklearn stack trace.
    if len(predicted) == 0:
        print(f"\n[ERROR] k_shot_test: all {test_length} test predictions "
              f"were rejected (n_rejects = {n_rejects}).")
        print(f"        The mean_squared_error call would crash on an "
              f"empty array, so we return NaN for every metric.")
        print(f"        Common causes:")
        print(f"          * Tokenizer producing garbled tokens (Mistral "
              f"regex bug; check the [DIAG] output above)")
        print(f"          * max_tokens too small; model truncated before "
              f"emitting any number")
        print(f"          * Model outputting prose only with no numeric "
              f"content")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, n_rejects

    MSE = mean_squared_error(actual, predicted)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(actual, predicted)
    if len(predicted) > 3: R_Squared = r2_score(actual, predicted)
    accuracy = get_accuracy(predicted, actual, median)
    _, F1_macro = get_f1_stats(predicted, actual)

    if test_length > 0:    
        return MSE, RMSE, MAE, R_Squared, accuracy, F1_macro, n_rejects
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def k_shot_predict(training_samples, test_samples, median, prompt_template,
                   feature, output, Rate, Memory, model, predict_output,
                   temperature=None):
    """
    Apply a FIXED demonstration set to a pool of query molecules, writing
    predicted,actual pairs to `predict_output` and returning the usual
    metric tuple.
    """
    MSE = RMSE = MAE = R_Squared = accuracy = F1_macro = 0.0
    n_rejects = 0
    Tau = 60.0 / Rate

    training_length = len(training_samples)
    test_length = len(test_samples)

    predicted = []
    actual = []
    conversation_history = ""

    with open(predict_output, 'w', newline='') as file:
        # Optional header so the output CSV is self-describing.
        file.write("predicted,actual\n")

        for nrow in range(test_length):
            if (Memory == "YES"):
                prompt = prompt_template + conversation_history
            elif (Memory == "NO"):
                prompt = prompt_template
            else:
                prompt = prompt_template

            for krow in range(training_length):
                prompt += (f"{feature}: {training_samples[krow][feature]}\n"
                           f"{output}: {training_samples[krow][output]}\n\n")

            prompt += f"{feature}: {test_samples[nrow][feature]}\n{output}: "

            # --- all of the following is INSIDE the for-nrow loop ---
            raw_value = send_query(prompt, model, temperature=temperature)
            value = extract_numeric_prediction(raw_value)

            if value is not None:
                predicted.append(float(value))
                actual.append(float(test_samples[nrow][output]))
                file.write(f"{value},{float(test_samples[nrow][output])}\n")

                if Memory == "YES":
                    conversation_history += (
                        f"{feature}: {test_samples[nrow][feature]}\n"
                        f"{output}: {value}\n\n")
            else:
                print(f"Skipping sample {test_samples[nrow]} due to "
                      f"invalid prediction (raw response: {raw_value})")
                n_rejects += 1

            time.sleep(Tau)

    # Convert to numpy arrays
    predicted = np.array(predicted)
    actual = np.array(actual)

    # Same empty-predictions guard as k_shot_test.
    if len(predicted) == 0:
        print(f"\n[ERROR] k_shot_predict: all {test_length} predictions "
              f"were rejected (n_rejects = {n_rejects}).")
        print(f"        Returning NaN for every metric.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, n_rejects

    MSE = mean_squared_error(actual, predicted)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(actual, predicted)
    if len(predicted) > 3:
        R_Squared = r2_score(actual, predicted)
    accuracy = get_accuracy(predicted, actual, median)
    _, F1_macro = get_f1_stats(predicted, actual)

    if test_length > 0:
        return MSE, RMSE, MAE, R_Squared, accuracy, F1_macro, n_rejects
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def get_template_prompts(pchem_property):
    test_prompt_template = CV_prompt_template = pchem_variable = title = ""
#                                        GIBBS
    if (pchem_property == "GIBBS"):    
        test_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules and solutions.\n"
        "Predict the Gibbs Free Energy of Solvation, Gibbs, of a Solute, of the following Solutes.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -2.34:\n"
        )
        CV_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules and solutions.\n"
        "Predict the Gibbs Free Energy of Solvation, Gibbs, of a Solute, based on the following examples.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -2.34:\n"
        )
        pchem_variable = 'Gibbs'
        title = "Free Energy of Solvation"
#                                          LogCMC
    elif (pchem_property == "LOGCMC"):     
        test_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules.\n"
        "Predict the log of the critical micelle concentration, logCMC, of the following molecules.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: 1.234:\n"
        )
        CV_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules.\n"
        "Predict the log of the critical micelle concentration, logCMC, based on the following examples.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: 1.234:\n"
        )
        pchem_variable = 'logCMC'
        title = "LogCMC"
#                                          Melting Point
    elif (pchem_property == "MP"):   
        test_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules.\n"
        "Predict the melting point in degree celcius, mpC, of the following molecules.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -55.12:\n"
        )
        CV_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules.\n"
        "Predict the melting point in degree celcius, mpC, based on the following examples.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -55.12:\n"
        )
        pchem_variable = 'mpC'
        title = "Melting Point"
#                                          Boiling Point   
    elif (pchem_property == "BP"):   
        test_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules.\n"
        "Predict the boiling point in degree celcius, bpC, of the following molecules.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -55.12:\n"
        )
        CV_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules.\n"
        "Predict the boiling point in degree celcius, bpC, based on the following examples.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -55.12:\n"
        )
        pchem_variable = 'bpC'
        title = "Boiling Point"
#                                            LogKow
    elif (pchem_property == "LOGKOW"):  
        test_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules.\n"
        "Predict the logarithm of the octanol-water partition coefficient, LogKow, of the following molecules.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -2.06:\n"
        )
        CV_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules.\n"
        "Predict the logarithm of the octanol-water partition coefficient, LogKow, based on the following examples.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -2.06:\n"
        )
        pchem_variable = 'logKow'
        title = "LogKow"
#                                           LogS
    elif (pchem_property == "LOGS"):  
        test_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules insolution.\n"
        "Predict the logarithm of the molar solubility, logS, (S in moles per liter) of the following molecules.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: 2.12:\n"
        )
        CV_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules in solution.\n"
        "Predict the logarithm of the molar solubility, logS, (S in moles per liter) based on the following examples.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: 2.12:\n"
        )
        pchem_variable = 'logS'
        title = "LogS"
    elif (pchem_property == "PKA"):  
        test_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules.\n"
        "Predict the acid dissociation constant, pka, of the following molecules.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -2.063:\n"
        )
        CV_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules.\n"
        "Predict acid dissociation constants, pka, based on the following examples.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -2.063:\n"
        )
        pchem_variable = 'pka'
        title = "pKa"
#                                           Vapor Pressure
    elif (pchem_property == "VP"):  
        test_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules insolution.\n"
        "Predict the negative of the logarithm of the vapor pressure, logVP, (VP in mmHg) of the following molecules.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: 2.12:\n"
        )
        CV_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules in solution.\n"
        "Predict the negative of the logarithm of the vapor pressure, logVP, (VP in mmHg) based on the following examples.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: 2.12:\n"
        )
        pchem_variable = 'logVP'
        title = "LogVP"
#                                           Henry's Law Constant
    elif (pchem_property == "HLC"):  
        test_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules insolution.\n"
        "Predict the negative of the logarithm of the Henry's Law Constant, logHLC, (HLC in atm-m3 per mol) of the following molecules.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -2.12:\n"
        )
        CV_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules in solution.\n"
        "Predict the negative of the logarithm of the Henry's Law Constant, logHLC, (HLC in atm-m3 per mol) based on the following examples.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -2.12:\n"
        )
        pchem_variable = 'logHLC'
        title = "LogHLC"
    elif (pchem_property == "LOGKWA"):
        test_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules.\n"
        "Predict the logarithm of the water-air partition coefficient, LogKwa, of the following molecules.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -2.06:\n"
        )
        CV_prompt_template = (
        "You are an expert in predicting physical and chemical properties of molecules.\n"
        "Predict the logarithm of the water-air partition coefficient, LogKwa, based on the following examples.\n"
        "Do not show verbose reasoning or chain of thought.\n"
        "Just respond with a single numerical value, for example: -2.06:\n"
        )
        pchem_variable = 'logKwa'
        title = "LogKwa"

    return test_prompt_template, CV_prompt_template, pchem_variable, title


def plot_distribution(df, title="Distribution of Free Energy of Solvation", variable=""):
    plt.hist(df, bins=50, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def Initialize_Arrays():
    A1 = []
    A2 = []
    A3 = []
    A4 = []
    A5 = []
    A6 = []
    return A1, A2, A3, A4, A5, A6    


def array_means(A1, A2, A3, A4, A5, A6):
    A1_mean = np.mean(np.array(A1))
    A2_mean = np.mean(np.array(A2))
    A3_mean = np.mean(np.array(A3))
    A4_mean = np.mean(np.array(A4))
    A5_mean = np.mean(np.array(A5))
    A6_mean = np.mean(np.array(A6))
    return A1_mean, A2_mean, A3_mean, A4_mean, A5_mean, A6_mean


def print_stat(MSE, RMSE, MAE, R2, acc, f1, Mode):
    print("***********************************************************")
    print("                 Performance Stats Summary")
    if (Mode == "TRAIN"): 
        print("                     (Training Dataset)")
    elif (Mode == "TEST"):
        print("                     (Testing Dataset)")
    elif (Mode == "VALIDATION"):
        print("                  (Validation Dataset)")
    elif (Mode == "PREDICT"):
        print("                  (Prediction Dataset)")
    print("***********************************************************")
    print(f"MSE = {MSE:.5f}, RMSE = {RMSE:.5f}, MAE = {MAE:.5f}, R2 = {R2:.5f}")
    print(f"Accuracy = {acc:.2f}%, F1_macro = {f1:.4f}")
    print("***********************************************************\n")


def print_summary(model, Max_Cycles, k_shot, kfold, n_k, N_total, Avg_MSE, MSE_sd, Avg_RMSE, RMSE_sd, Avg_MAE, MAE_sd, Avg_R2, R2_sd, Avg_acc, f1, NRMSE_mean, NRMSE_range, Mode):
    print("\n")
    print("***********************************************************")
    print("            Average Performance Stats Summary")
    if (Mode == "TRAIN"): 
        print("                     (Training Dataset)")
    elif (Mode == "TEST"):
        print("                     (Testing Dataset)")
    elif (Mode == "VALIDATION"):
        print("                 (Validation Dataset)")
    elif (Mode == "PREDICT"):
        print("                  (Prediction Dataset)")
    print("***********************************************************")
    print(f"Pre-Trained LLM Model: {model}                       ")
    print(f"Number of Iterations: {Max_Cycles}")
    if (Mode == "TRAIN"): print(f"K_Shot: {k_shot}, K_Fold: {kfold}, nk: {n_k}, Validation Inquiries: {N_total}")
    print(f"Avg_MSE = {Avg_MSE:.5f} ± {MSE_sd:.3f}, Avg_RMSE = {Avg_RMSE:.5f} ± {RMSE_sd:.3f}")
    print(f"Avg_MAE = {Avg_MAE:.5f} ± {MAE_sd}, Avg_R2 = {Avg_R2:.5f} ± {R2_sd}")
    print(f"Avg_Accuracy = {Avg_acc:.2f}%, Avg_F1_macro = {f1:.4f}")
    print(f"NRMSE(mean): {NRMSE_mean:.2f}%, NRMSE(range): {NRMSE_range:.2f}%")
    print("***********************************************************")    


# =============================================================================
#                       Batch-Mode API  (non-interactive)
# =============================================================================
# For the systematic sweep across (model, property) cells this batch API
# provides:
#
#   select_llm_model_by_choice(choice)
#       Non-interactive version of select_llm_model().  Loads the MLX
#       weights and sets _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_MAX_TOKENS.
#       Returns the resolved model_name string used as the `model`
#       argument in send_query().
#
#   run_cell(...)        Testing + Validation pipeline for one cell.
#   run_cell_predict(...) Prediction with a fixed demonstration set.
#   run_cell_forecast(...) SMILES-only replicate forecasting with
#                          metrological uncertainty.
#
# optimal_samples path: run_cell writes to data_paths["optimal_samples"]
# (the batch driver resolves this from each property's `optimal_dir` in
# properties.yaml, as {optimal_dir}/{model_slug}_optimal_samples.csv).
# run_cell_predict / run_cell_forecast read the same key. If the key is
# absent, all three fall back to out_dir/optimal_samples.csv.
# =============================================================================


def select_llm_model_by_choice(choice, mlx_load_kwargs=None):
    """
    Non-interactive version of select_llm_model.  Returns the model
    name string.  Loads the MLX model + tokenizer once and stores them
    in module globals.

    Idempotent: if called twice in a row with the same choice, the
    weights are NOT reloaded.  When switching between models (e.g.
    1 -> 2), the previous model is explicitly unloaded and MLX's Metal
    cache is cleared to release GPU memory before the new model loads.
    This matters on memory-constrained hosts (e.g. M5 Max 64 GB) where
    Llama-3.3-70B-Q4 needs ~40 GB and keeping a previous model resident
    could push the process into swap.
    """
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_MAX_TOKENS
    global _CHAT_TEMPLATE_FALLBACK_WARNED

    # Reset the fallback-warned flag so each new model gets its own
    # one-shot notice about missing chat_template (if applicable).
    _CHAT_TEMPLATE_FALLBACK_WARNED = False

    if choice not in LOCAL_MODEL_MAP:
        valid = sorted(LOCAL_MODEL_MAP.keys())
        raise ValueError(f"Invalid model choice {choice!r}; "
                         f"expected one of {valid}.")

    model_id = LOCAL_MODEL_MAP[choice]

    # Idempotency check: if we already have this exact model loaded,
    # skip the reload.  Uses a module-level cache key.
    global _LOCAL_LOADED_MODEL_ID
    try:
        already_loaded = (_LOCAL_LOADED_MODEL_ID == model_id
                          and _LOCAL_MODEL is not None
                          and _LOCAL_TOKENIZER is not None)
    except NameError:
        already_loaded = False
    if already_loaded:
        print(f"[backend]  LOCAL (mlx_lm)  [already loaded]")
        print(f"[model  ]  {model_id}")
        return model_id

    # Different model requested: free the previous one first.
    _unload_local_model()

    print(f"[backend]  LOCAL (mlx_lm)")
    print(f"[model  ]  {model_id}")

    ok, load_fn_or_err = _mlx_available()
    if not ok:
        raise ImportError(f"mlx_lm not available: {load_fn_or_err}")

    tokenizer_config = {}
    if "mistral" in model_id.lower():
        tokenizer_config["fix_mistral_regex"] = True

    # Pre-patch Llama-4 config.json for known type mismatches (no-op
    # on non-Llama-4 models).  Returns (path, model_config_overrides)
    # tuple; the overrides go into mlx_lm.load's model_config kwarg,
    # bypassing HF-cache symlink issues that made the file-only patch
    # unreliable on macOS.
    model_path, model_config_overrides = _patch_llama4_config(model_id)

    load_kwargs = {}
    if tokenizer_config:
        load_kwargs["tokenizer_config"] = tokenizer_config
    if model_config_overrides:
        load_kwargs["model_config"] = model_config_overrides

    t0 = time.time()
    try:
        _LOCAL_MODEL, _LOCAL_TOKENIZER = load_fn_or_err(
            model_path, **load_kwargs
        )
    except TypeError:
        # Older mlx_lm: retry without model_config, then without any
        # kwargs at all.
        try:
            _LOCAL_MODEL, _LOCAL_TOKENIZER = load_fn_or_err(
                model_path, tokenizer_config=tokenizer_config
            )
            if model_config_overrides:
                print(f"[warn   ]  mlx_lm does not accept model_config; "
                      f"Llama-4 overrides not applied.")
        except TypeError:
            _LOCAL_MODEL, _LOCAL_TOKENIZER = load_fn_or_err(model_path)
    dt = time.time() - t0
    print(f"[loaded ]  in {dt:.1f} s")

    _LOCAL_LOADED_MODEL_ID = model_id
    if "gemma" in model_id.lower() and "optiq" in model_id.lower():
        _LOCAL_MAX_TOKENS = 512
    else:
        _LOCAL_MAX_TOKENS = 256
    return model_id


def _unload_local_model():
    """
    Release the currently-loaded MLX model + tokenizer and clear the
    MLX Metal cache.  Safe no-op if no model is loaded.  Called between
    model switches in select_llm_model_by_choice() to keep peak memory
    footprint at max(m_i) rather than sum(m_i) for the run.
    """
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_LOADED_MODEL_ID
    if _LOCAL_MODEL is None and _LOCAL_TOKENIZER is None:
        return
    _LOCAL_MODEL = None
    _LOCAL_TOKENIZER = None
    try:
        _LOCAL_LOADED_MODEL_ID = None
    except NameError:
        pass
    import gc
    gc.collect()
    try:
        import mlx.core as _mx
        # mx.metal.clear_cache() exists in mlx_lm >= 0.18 or so.  Best-
        # effort: newer versions may have renamed or removed it.  We
        # don't rely on the cache clearing succeeding, just on the
        # Python refs being dropped.
        if hasattr(_mx, "metal") and hasattr(_mx.metal, "clear_cache"):
            _mx.metal.clear_cache()
    except Exception:
        pass
    print("[unload ]  Previous local model released; MLX cache cleared.")


def _perm_shuffle_labels(training_pool, property_var, rng):
    """
    Return a copy of the training pool with property values permuted
    across records (SMILES stay in place; only y-labels are shuffled).
    Used for the permutation-control null run.
    """
    y_values = [rec[property_var] for rec in training_pool]
    rng.shuffle(y_values)
    shuffled = []
    for rec, y in zip(training_pool, y_values):
        new_rec = dict(rec)
        new_rec[property_var] = y
        shuffled.append(new_rec)
    return shuffled


def run_cell(model_name, model_slug, property_key, data_paths, out_dir,
             M_c=50, N_shots=32, test_split=0.20, permutation=False,
             seed_base=0, temperature=None):
    """
    Run the Testing + Validation pipeline for a single (model, property)
    cell.  Writes per-cycle NPZ, the optimal_samples CSV, and
    human-readable summaries.  Returns a dict with rollup metrics.

    The optimal_samples CSV is written to data_paths["optimal_samples"]
    when provided (the driver resolves this from the property's
    `optimal_dir` in properties.yaml), else to out_dir/optimal_samples.csv.
    """
    import numpy as _np
    import pandas as _pd
    import random as _random
    import warnings as _warnings

    os.makedirs(out_dir, exist_ok=True)
    property_var = data_paths["var"]

    # ---- Data loading -----------------------------------------------
    df_train = _pd.DataFrame(read_csv_file(data_paths["training"]))
    df_train[property_var] = _pd.to_numeric(df_train[property_var],
                                             errors="coerce")
    df_train = df_train.dropna(subset=[property_var])

    df_val = _pd.DataFrame(read_csv_file(data_paths["validation"]))
    df_val[property_var] = _pd.to_numeric(df_val[property_var],
                                           errors="coerce")
    df_val = df_val.dropna(subset=[property_var])

    N_total = len(df_train) + len(df_val)
    if N_total < 20:
        raise ValueError(f"Property {property_key}: too few valid "
                         f"records ({N_total}); need at least 20.")

    # Compute median from the COMBINED distribution (train + val),
    # matching the __main__ convention for binary-classification.
    df_full = _pd.concat([df_train, df_val], ignore_index=True)
    median = float(df_full[property_var].median())
    mean_all = float(df_full[property_var].mean())
    y_min = float(df_full[property_var].min())
    y_max = float(df_full[property_var].max())

    print(f"\n{'='*68}")
    print(f"  RUN CELL   {model_slug} x {property_key}"
          f"   {' (PERMUTATION CONTROL)' if permutation else ''}")
    print(f"  train={len(df_train)}  val={len(df_val)}  "
          f"median={median:.4f}  M_c={M_c}  N_shots={N_shots}")
    print(f"{'='*68}\n")

    # ---- Testing loop ------------------------------------------------
    train_test_data, test_data = train_test_split(
        df_train, test_size=test_split, random_state=seed_base
    )
    training_pool = train_test_data.to_dict("records")
    test_pool = test_data.to_dict("records")
    validation_samples = df_val.to_dict("records")

    # Add binary_class / five_class columns to test and validation
    # samples (the RCHAT k_shot_test function reads them).
    def _add_class_cols(records, y_var, med, quintile_edges):
        for r in records:
            y = float(r[y_var])
            r["binary_class"] = int(y > med)
            # Quintile bin (0..4)
            k = 0
            for edge in quintile_edges:
                if y > edge:
                    k += 1
            r["five_class"] = min(k, 4)
        return records

    q_edges = _np.percentile(df_full[property_var].values,
                              [20, 40, 60, 80]).tolist()
    test_pool = _add_class_cols(test_pool, property_var, median, q_edges)
    validation_samples = _add_class_cols(validation_samples, property_var,
                                          median, q_edges)

    prompt_template = get_template_prompts(property_key)[0]

    # Storage for per-cycle metrics
    metric_keys = ["MSE", "RMSE", "MAE", "R2", "Acc", "F1",
                   "NRMSE_mean", "NRMSE_range", "n_rejects"]
    per_cycle = {k: _np.full(M_c, _np.nan) for k in metric_keys}
    per_cycle["seed"] = _np.zeros(M_c, dtype=_np.int64)

    best_MAE = _np.inf
    best_cycle = -1
    optimal_training_samples = []
    Rate = 6000   # LOCAL backend: no remote rate budget

    t_cell_start = time.time()

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        for k in range(M_c):
            seed = seed_base + k
            rng = _random.Random(seed)
            per_cycle["seed"][k] = seed

            # Fresh subsample of training pool per cycle
            n_subsample = min(N_shots, len(training_pool))
            training_samples = rng.sample(training_pool, n_subsample)
            # Add class cols
            training_samples = _add_class_cols(training_samples,
                                                property_var, median,
                                                q_edges)

            if permutation:
                training_samples = _perm_shuffle_labels(
                    training_samples, property_var, rng
                )

            print(f"    Cycle {k+1:>3}/{M_c}  seed={seed}", end="",
                  flush=True)

            try:
                res = k_shot_test(
                    training_samples, test_pool, median,
                    prompt_template, "Molecule", property_var,
                    Rate, "NO", model_name, temperature=temperature,
                )                
                MSE, RMSE, MAE, R2, acc, f1, n_rej = res
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                continue

            per_cycle["MSE"][k]  = MSE
            per_cycle["RMSE"][k] = RMSE
            per_cycle["MAE"][k]  = MAE
            per_cycle["R2"][k]   = R2
            per_cycle["Acc"][k]  = acc
            per_cycle["F1"][k]   = f1
            per_cycle["n_rejects"][k] = n_rej
            if not _np.isnan(RMSE):
                if abs(mean_all) > 0:
                    per_cycle["NRMSE_mean"][k]  = RMSE / abs(mean_all) * 100
                if (y_max - y_min) > 0:
                    per_cycle["NRMSE_range"][k] = RMSE / (y_max - y_min) * 100

            print(f"  MAE={MAE:.3f}  RMSE={RMSE:.3f}  R2={R2:.3f}  "
                  f"n_rej={n_rej}")

            if not _np.isnan(MAE) and MAE < best_MAE:
                best_MAE = MAE
                best_cycle = k
                optimal_training_samples = training_samples

    t_testing = time.time() - t_cell_start

    # ---- Validation using optimal_training_samples -------------------
    val_metrics = {k: float("nan") for k in metric_keys}
    if optimal_training_samples:
        print(f"\n  Validation using cycle {best_cycle+1} demos "
              f"(MAE = {best_MAE:.4f})")
        try:
            res = k_shot_test(
                optimal_training_samples, validation_samples, median,
                prompt_template, "Molecule", property_var,
                Rate, "NO", model_name, temperature=temperature,
            )
            v_MSE, v_RMSE, v_MAE, v_R2, v_acc, v_f1, v_rej = res
            val_metrics["MSE"]  = v_MSE
            val_metrics["RMSE"] = v_RMSE
            val_metrics["MAE"]  = v_MAE
            val_metrics["R2"]   = v_R2
            val_metrics["Acc"]  = v_acc
            val_metrics["F1"]   = v_f1
            val_metrics["n_rejects"] = v_rej
            if not _np.isnan(v_RMSE):
                if abs(mean_all) > 0:
                    val_metrics["NRMSE_mean"]  = v_RMSE/abs(mean_all)*100
                if (y_max - y_min) > 0:
                    val_metrics["NRMSE_range"] = v_RMSE/(y_max-y_min)*100
            print(f"  Val:  MAE={v_MAE:.3f}  RMSE={v_RMSE:.3f}  "
                  f"R2={v_R2:.3f}  Acc={v_acc:.2f}%  n_rej={v_rej}")
        except Exception as e:
            print(f"  Validation ERROR: {type(e).__name__}: {e}")

    t_total = time.time() - t_cell_start

    # ---- Persist outputs --------------------------------------------
    npz_path = os.path.join(out_dir, "cycles.npz")
    _np.savez_compressed(
        npz_path,
        MAE_test        = per_cycle["MAE"],
        RMSE_test       = per_cycle["RMSE"],
        MSE_test        = per_cycle["MSE"],
        R2_test         = per_cycle["R2"],
        Acc_test        = per_cycle["Acc"],
        F1_test         = per_cycle["F1"],
        NRMSE_mean_test = per_cycle["NRMSE_mean"],
        NRMSE_range_test= per_cycle["NRMSE_range"],
        n_rejects_test  = per_cycle["n_rejects"],
        seed_per_cycle  = per_cycle["seed"],
        best_cycle      = _np.int64(best_cycle),
        MAE_val         = _np.float64(val_metrics["MAE"]),
        RMSE_val        = _np.float64(val_metrics["RMSE"]),
        MSE_val         = _np.float64(val_metrics["MSE"]),
        R2_val          = _np.float64(val_metrics["R2"]),
        Acc_val         = _np.float64(val_metrics["Acc"]),
        F1_val          = _np.float64(val_metrics["F1"]),
        NRMSE_mean_val  = _np.float64(val_metrics["NRMSE_mean"]),
        NRMSE_range_val = _np.float64(val_metrics["NRMSE_range"]),
        n_rejects_val   = _np.float64(val_metrics["n_rejects"]),
        median_value    = _np.float64(median),
        mean_value      = _np.float64(mean_all),
        y_min           = _np.float64(y_min),
        y_max           = _np.float64(y_max),
        wall_time_sec   = _np.float64(t_total),
        wall_time_testing_sec = _np.float64(t_testing),
    )
    print(f"\n  cycles.npz written to {npz_path}")

    if optimal_training_samples:
        # Write the best-cycle demonstrations.  Prefer the explicit
        # path supplied by the caller (driver resolves this from the
        # property's `optimal_dir` in properties.yaml); fall back to the
        # per-cell run folder for backward compatibility.
        opt_csv = data_paths.get(
            "optimal_samples",
            os.path.join(out_dir, "optimal_samples.csv"),
        )
        if len(optimal_training_samples) > 0:
            os.makedirs(os.path.dirname(opt_csv) or ".", exist_ok=True)
            with open(opt_csv, "w", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=optimal_training_samples[0].keys()
                )
                writer.writeheader()
                writer.writerows(optimal_training_samples)
            print(f"  optimal_samples.csv written to {opt_csv}")

    # ---- Rollup for summary.csv --------------------------------------
    def _nanstat(a, stat):
        arr = _np.asarray(a, dtype=float)
        m = ~_np.isnan(arr)
        if not m.any():
            return float("nan")
        return float(stat(arr[m]))

    mae_arr = per_cycle["MAE"]
    q1 = _nanstat(mae_arr, lambda a: _np.percentile(a, 25))
    q3 = _nanstat(mae_arr, lambda a: _np.percentile(a, 75))

    rollup = {
        "model_slug"         : model_slug,
        "model_name"         : model_name,
        "property"           : property_key,
        "backend"            : "LOCAL",
        "M_c"                : M_c,
        "N_shots"            : N_shots,
        "permutation"        : bool(permutation),
        "median_MAE_test"    : _nanstat(mae_arr, _np.median),
        "IQR_MAE_test"       : (q3 - q1) if (q3 == q3 and q1 == q1) else float("nan"),
        "mean_MAE_test"      : _nanstat(mae_arr, _np.mean),
        "std_MAE_test"       : _nanstat(mae_arr, lambda a: _np.std(a, ddof=1)) if _np.sum(~_np.isnan(mae_arr)) > 1 else float("nan"),
        "best_MAE_test"      : float(best_MAE) if best_MAE != _np.inf else float("nan"),
        "best_cycle"         : int(best_cycle) + 1,
        "MAE_val"            : val_metrics["MAE"],
        "RMSE_val"           : val_metrics["RMSE"],
        "R2_val"             : val_metrics["R2"],
        "Acc_val"            : val_metrics["Acc"],
        "F1_val"             : val_metrics["F1"],
        "NRMSE_range_val"    : val_metrics["NRMSE_range"],
        "n_rejects_test_total": int(_np.nansum(per_cycle["n_rejects"])),
        "n_rejects_val"      : val_metrics["n_rejects"],
        "wall_time_h"        : t_total / 3600.0,
    }

    # ---- Human-readable summaries ------------------------------------
    with open(os.path.join(out_dir, "testing_summary.txt"), "w") as fh:
        fh.write(f"Cell: {model_slug} x {property_key}\n")
        fh.write(f"Permutation control: {permutation}\n")
        fh.write(f"Model: {model_name}\n")
        fh.write(f"M_c={M_c}, N_shots={N_shots}, seed_base={seed_base}\n")
        fh.write(f"Testing dataset: {len(test_pool)} molecules\n")
        fh.write(f"Median (binary-class threshold): {median:.4f}\n")
        fh.write(f"\n--- Per-cycle MAE distribution ---\n")
        fh.write(f"Median MAE : {rollup['median_MAE_test']:.4f}\n")
        fh.write(f"Mean MAE   : {rollup['mean_MAE_test']:.4f} "
                 f"(std {rollup['std_MAE_test']:.4f})\n")
        fh.write(f"IQR        : {rollup['IQR_MAE_test']:.4f}\n")
        fh.write(f"Best MAE   : {rollup['best_MAE_test']:.4f} "
                 f"(cycle {rollup['best_cycle']})\n")
        fh.write(f"n_rejects  : {rollup['n_rejects_test_total']}\n")
        fh.write(f"wall time  : {t_total/60:.1f} min\n")

    with open(os.path.join(out_dir, "validation_summary.txt"), "w") as fh:
        fh.write(f"Cell: {model_slug} x {property_key}\n")
        fh.write(f"Permutation control: {permutation}\n")
        fh.write(f"Validation dataset: {len(validation_samples)} molecules\n")
        for k, v in val_metrics.items():
            fh.write(f"{k:12s}: {v}\n")

    return rollup


def run_cell_predict(model_name, model_slug, property_key, data_paths,
                     out_dir, predict_pool_path=None, seed_base=0,
                     temperature=None):
    """
    Batch-mode Prediction for a single (model, property) cell.

    Loads the optimal demonstration set that a prior run_cell() saved
    (data_paths["optimal_samples"], resolved by the driver from the
    property's `optimal_dir`; else out_dir/optimal_samples.csv) and
    applies it, fixed, to a pool of query molecules.
    """
    import numpy as _np
    import pandas as _pd

    os.makedirs(out_dir, exist_ok=True)
    property_var = data_paths["var"]
    feature_col = data_paths.get("feature_col", "Molecule")

    # ---- Resolve the demonstration set (optimal_samples.csv) --------
    opt_path = data_paths.get("optimal_samples",
                              os.path.join(out_dir, "optimal_samples.csv"))
    if not os.path.exists(opt_path):
        raise FileNotFoundError(
            f"Prediction requires a saved demonstration set but none was "
            f"found at {opt_path}. Run the Testing sweep (run_cell) for "
            f"this cell first so the optimal_samples CSV exists.")

    df_opt = _pd.DataFrame(read_csv_file(opt_path))
    df_opt[property_var] = _pd.to_numeric(df_opt[property_var],
                                          errors="coerce")
    df_opt = df_opt.dropna(subset=[property_var])
    training_samples = df_opt.to_dict("records")
    if len(training_samples) == 0:
        raise ValueError(f"optimal_samples CSV at {opt_path} has no valid "
                         f"rows for property {property_key}.")

    # ---- Resolve the query pool -------------------------------------
    predict_path = (data_paths.get("predict")
                    or predict_pool_path
                    or data_paths.get("validation"))
    if not predict_path or not os.path.exists(predict_path):
        raise FileNotFoundError(
            f"No query pool found for {property_key}. Provide "
            f"data_paths['predict'] or a validation CSV. Got: {predict_path}")

    df_pred = _pd.DataFrame(read_csv_file(predict_path))
    df_pred[property_var] = _pd.to_numeric(df_pred[property_var],
                                           errors="coerce")
    df_pred = df_pred.dropna(subset=[property_var])
    predict_samples = df_pred.to_dict("records")

    # ---- Median for binary accuracy, consistent with run_cell -------
    combined = _pd.concat([df_opt[[property_var]], df_pred[[property_var]]],
                          ignore_index=True)
    median = float(combined[property_var].median())
    mean_all = float(combined[property_var].mean())
    y_min = float(combined[property_var].min())
    y_max = float(combined[property_var].max())

    print(f"\n{'='*68}")
    print(f"  PREDICT CELL   {model_slug} x {property_key}")
    print(f"  demos={len(training_samples)}  queries={len(predict_samples)}"
          f"  median={median:.4f}")
    print(f"  demos from : {opt_path}")
    print(f"  queries    : {predict_path}")
    print(f"{'='*68}\n")

    prompt_template = get_template_prompts(property_key)[0]
    Rate = 6000   # LOCAL backend

    predictions_csv = os.path.join(out_dir, "predictions.csv")

    t0 = time.time()
    try:
        res = k_shot_predict(
            training_samples, predict_samples, median,
            prompt_template, "Molecule", property_var,
            Rate, "NO", model_name, predictions_csv,
            temperature=temperature,
        )
        MSE, RMSE, MAE, R2, acc, f1, n_rej = res
    except Exception as e:
        print(f"  PREDICT ERROR: {type(e).__name__}: {e}")
        raise
    t_total = time.time() - t0

    # NRMSE (range-normalized) on the prediction pool, native units.
    nrmse_range = (RMSE / (y_max - y_min) * 100.0
                   if (y_max - y_min) > 0 and not _np.isnan(RMSE)
                   else float("nan"))

    rollup = {
        "model_slug"      : model_slug,
        "model_name"      : model_name,
        "property"        : property_key,
        "backend"         : "LOCAL",
        "phase"           : "PREDICT",
        "n_demos"         : len(training_samples),
        "n_queries"       : len(predict_samples),
        "MAE_predict"     : float(MAE) if not _np.isnan(MAE) else float("nan"),
        "RMSE_predict"    : float(RMSE) if not _np.isnan(RMSE) else float("nan"),
        "R2_predict"      : float(R2) if not _np.isnan(R2) else float("nan"),
        "Acc_predict"     : float(acc) if not _np.isnan(acc) else float("nan"),
        "F1_predict"      : float(f1) if not _np.isnan(f1) else float("nan"),
        "NRMSE_range_predict": nrmse_range,
        "n_rejects_predict": int(n_rej),
        "wall_time_h"     : t_total / 3600.0,
    }

    with open(os.path.join(out_dir, "predict_summary.txt"), "w") as fh:
        fh.write(f"Cell: {model_slug} x {property_key} [PREDICT]\n")
        fh.write(f"Model: {model_name}\n")
        fh.write(f"Demonstrations: {len(training_samples)} (from {opt_path})\n")
        fh.write(f"Query pool: {len(predict_samples)} (from {predict_path})\n")
        fh.write(f"Median (binary-class threshold): {median:.4f}\n")
        fh.write(f"MAE   : {rollup['MAE_predict']}\n")
        fh.write(f"RMSE  : {rollup['RMSE_predict']}\n")
        fh.write(f"R2    : {rollup['R2_predict']}\n")
        fh.write(f"Acc   : {rollup['Acc_predict']}\n")
        fh.write(f"NRMSE(range): {rollup['NRMSE_range_predict']}\n")
        fh.write(f"n_rejects: {rollup['n_rejects_predict']}\n")
        fh.write(f"predictions.csv: {predictions_csv}\n")
        fh.write(f"wall time: {t_total/60:.1f} min\n")

    print(f"  predictions.csv written to {predictions_csv}")
    print(f"  MAE={rollup['MAE_predict']}  RMSE={rollup['RMSE_predict']}  "
          f"R2={rollup['R2_predict']}  n_rej={n_rej}")

    return rollup


def _student_t_95(nu):
    """
    Two-tailed Student's t critical value for 95% coverage with nu
    degrees of freedom. Uses scipy if available, else a small lookup
    table with interpolation, else the normal limit 1.96.

    nu = Nexp - 1. Returns np.nan for nu < 1 (cannot form a std).
    """
    import numpy as _np
    if nu < 1:
        return float("nan")
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(0.975, nu))
    except Exception:
        pass
    # Lookup table of t_{0.975, nu} for common nu; interpolate in 1/nu.
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 24: 2.064, 30: 2.042,
        40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984,
        120: 1.980,
    }
    if nu in table:
        return table[nu]
    if nu >= 120:
        return 1.960
    keys = sorted(table)
    lo = max(k for k in keys if k <= nu)
    hi = min(k for k in keys if k >= nu)
    if lo == hi:
        return table[lo]
    # Linear interpolation in 1/nu (t varies smoothly in 1/nu)
    x0, x1 = 1.0 / lo, 1.0 / hi
    y0, y1 = table[lo], table[hi]
    x = 1.0 / nu
    return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))


def run_cell_forecast(model_name, model_slug, property_key, data_paths,
                      out_dir, N_exp=20, temperatures=(0.7,),
                      seed_base=0):
    """
    Pure-prediction (forecast) mode with metrological uncertainty.

    For each temperature T in `temperatures`, and each query molecule,
    issue N_exp independent model queries (the LLM is stochastic at
    T > 0), collect the numeric predictions, and compute per-molecule
    statistics suitable for a measurement-uncertainty report:

        mean, median, std (sample, ddof=1), min, max,
        uA   = std / sqrt(N_valid)                 (Type A std uncertainty)
        U95  = t_{0.975, N_valid-1} * uA           (expanded uncertainty,
                                                     95% coverage)

    No comparison against a true value is made: only the SMILES input
    column is read from the query CSV.

    Demonstrations are the optimal_samples CSV that a prior Testing
    run_cell wrote (data_paths["optimal_samples"], resolved from the
    property's `optimal_dir`; else out_dir/optimal_samples.csv), reused
    fixed across all temperatures and molecules.
    """
    import numpy as _np
    import pandas as _pd
    import math as _math

    os.makedirs(out_dir, exist_ok=True)
    property_var = data_paths["var"]
    feature_col = data_paths.get("feature_col", "Molecule")

    if N_exp < 1:
        raise ValueError(f"N_exp must be >= 1; got {N_exp}.")
    temps = [float(t) for t in temperatures]
    if any(t == 0.0 for t in temps):
        print("[forecast] WARNING: T = 0 (greedy) yields identical "
              "replicates, so std = 0 and U95 = 0 for those cells.")

    # ---- Demonstrations (optimal_samples.csv) -----------------------
    opt_path = data_paths.get("optimal_samples",
                              os.path.join(out_dir, "optimal_samples.csv"))
    if not os.path.exists(opt_path):
        raise FileNotFoundError(
            f"Forecast requires a saved demonstration set but none was "
            f"found at {opt_path}. Run the Testing sweep (run_cell) for "
            f"this cell first so the optimal_samples CSV exists.")
    df_opt = _pd.DataFrame(read_csv_file(opt_path))
    if property_var not in df_opt.columns or feature_col not in df_opt.columns:
        raise KeyError(
            f"optimal_samples CSV at {opt_path} must contain both "
            f"'{feature_col}' and '{property_var}' columns; found "
            f"{list(df_opt.columns)}.")
    df_opt[property_var] = _pd.to_numeric(df_opt[property_var],
                                          errors="coerce")
    df_opt = df_opt.dropna(subset=[property_var])
    training_samples = df_opt.to_dict("records")
    if not training_samples:
        raise ValueError(f"No valid demonstration rows in {opt_path}.")

    # ---- Query pool: read ONLY the SMILES column --------------------
    predict_path = data_paths.get("predict")
    if not predict_path or not os.path.exists(predict_path):
        raise FileNotFoundError(
            f"Forecast needs data_paths['predict'] (query CSV). "
            f"Got: {predict_path}")
    df_q = _pd.DataFrame(read_csv_file(predict_path))
    if feature_col not in df_q.columns:
        raise KeyError(
            f"Query CSV {predict_path} has no '{feature_col}' column; "
            f"found {list(df_q.columns)}.")
    query_smiles = [str(s) for s in df_q[feature_col].tolist()
                    if str(s).strip() != ""]
    n_mol = len(query_smiles)
    if n_mol == 0:
        raise ValueError(f"No SMILES found in {predict_path} "
                         f"column '{feature_col}'.")

    prompt_template = get_template_prompts(property_key)[0]
    Rate = 6000   # LOCAL backend
    Tau = 60.0 / Rate

    print(f"\n{'='*68}")
    print(f"  FORECAST CELL   {model_slug} x {property_key}")
    print(f"  demos={len(training_samples)}  molecules={n_mol}  "
          f"N_exp={N_exp}  temps={temps}")
    print(f"  demos from : {opt_path}")
    print(f"  queries    : {predict_path}  (SMILES column '{feature_col}')")
    print(f"{'='*68}\n")

    # Pre-build the fixed demonstration block once (it never changes).
    demo_block = ""
    for rec in training_samples:
        demo_block += (f"Molecule: {rec[feature_col]}\n"
                       f"{property_var}: {rec[property_var]}\n\n")

    raw_cols = [f"raw_{i+1}" for i in range(N_exp)]
    out_rows = []
    n_reject_total = 0

    t_start = time.time()

    for T in temps:
        print(f"\n[forecast] Temperature T = {T}")
        for mi, smi in enumerate(query_smiles):
            prompt = prompt_template + demo_block
            prompt += f"Molecule: {smi}\n{property_var}: "

            preds = []
            raws = []
            n_rej = 0
            for rep in range(N_exp):
                raw = send_query(prompt, model_name, temperature=T)
                val = extract_numeric_prediction(raw)
                if val is not None:
                    preds.append(float(val))
                    raws.append(float(val))
                else:
                    raws.append(None)
                    n_rej += 1
                time.sleep(Tau)

            n_reject_total += n_rej
            arr = _np.array(preds, dtype=float)
            n_valid = len(arr)

            mean = float(_np.mean(arr)) if n_valid >= 1 else float("nan")
            median = float(_np.median(arr)) if n_valid >= 1 else float("nan")
            vmin = float(_np.min(arr)) if n_valid >= 1 else float("nan")
            vmax = float(_np.max(arr)) if n_valid >= 1 else float("nan")
            if n_valid >= 2:
                std = float(_np.std(arr, ddof=1))
                uA = std / _math.sqrt(n_valid)
                ts = _student_t_95(n_valid - 1)
                U95 = ts * uA if not _math.isnan(ts) else float("nan")
            else:
                std = float("nan")
                uA = float("nan")
                ts = float("nan")
                U95 = float("nan")

            row = {
                "temperature": T,
                "molecule": smi,
                "N_exp": N_exp,
                "N_valid": n_valid,
                "n_reject": n_rej,
                "mean": mean,
                "median": median,
                "std": std,
                "min": vmin,
                "max": vmax,
                "uA": uA,
                "t_crit_95": ts,
                "U95": U95,
            }
            for ci, cname in enumerate(raw_cols):
                row[cname] = raws[ci] if ci < len(raws) else None
            out_rows.append(row)

            print(f"    [T={T}] mol {mi+1}/{n_mol}  "
                  f"mean={mean:.4f}  std={std:.4f}  U95={U95:.4f}  "
                  f"n_valid={n_valid}  n_rej={n_rej}")

    t_total = time.time() - t_start

    # ---- Write output CSV -------------------------------------------
    base_cols = ["temperature", "molecule", "N_exp", "N_valid", "n_reject",
                 "mean", "median", "std", "min", "max",
                 "uA", "t_crit_95", "U95"]
    fieldnames = base_cols + raw_cols
    out_csv = os.path.join(out_dir, f"forecast_{property_key}.csv")
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k, ""))
                             for k in fieldnames})

    print(f"\n  forecast CSV written to {out_csv}")

    rollup = {
        "model_slug": model_slug,
        "model_name": model_name,
        "property": property_key,
        "backend": "LOCAL",
        "phase": "FORECAST",
        "n_molecules": n_mol,
        "N_exp": N_exp,
        "n_temperatures": len(temps),
        "temperatures": ";".join(str(t) for t in temps),
        "n_demos": len(training_samples),
        "n_reject_total": int(n_reject_total),
        "forecast_csv": out_csv,
        "wall_time_h": t_total / 3600.0,
    }

    with open(os.path.join(out_dir, "forecast_summary.txt"), "w") as fh:
        fh.write(f"Cell: {model_slug} x {property_key} [FORECAST]\n")
        fh.write(f"Model: {model_name}\n")
        fh.write(f"Demonstrations: {len(training_samples)} (from {opt_path})\n")
        fh.write(f"Query molecules: {n_mol} (from {predict_path})\n")
        fh.write(f"N_exp per molecule per T: {N_exp}\n")
        fh.write(f"Temperatures: {temps}\n")
        fh.write(f"Total rejected replicates: {n_reject_total}\n")
        fh.write(f"Output: {out_csv}\n")
        fh.write(f"Wall time: {t_total/60:.1f} min\n")

    return rollup


# =============================================================================
#                                Main Entry Point
# =============================================================================
if __name__ == "__main__":

#   Initialize some counters        
    train_rejects = test_rejects = vali_rejects = 0
    MAE_Optimal = 1000000

    # Initialize the "best training set so far" container at the same
    # scope as MAE_Optimal.
    optimal_training_samples = []

#   Setup some parameters that control the query loops
    Zero_Shot = "NO"
    Save = False  

#   Start the clock counter    
    start_elapsed = time.time()
    start_cpu = time.process_time()

# =============================================================================
#                     Choose LLM Model
# =============================================================================
    # Prompt the user for the local model.  Loads the MLX model +
    # tokenizer once so every subsequent send_query() reuses them.
    model = select_llm_model()

#   Setup Model Hyperparameters
    Max_Cycles = 50
    llm_temp = DEFAULT_TEMPERATURE
    max_tokens = 100000 
    # LOCAL backend: no remote request-per-minute budget, so collapse the
    # inter-query throttle to ~10 ms.
    Rate = 6000

#   Initialize some lists   
    Train_MSE_avg, Train_RMSE_avg, Train_MAE_avg, Train_R2_avg, Train_acc_avg, Train_f1_avg = Initialize_Arrays()
    Test_MSE_avg, Test_RMSE_avg, Test_MAE_avg, Test_R2_avg, Test_acc_avg, Test_f1_avg = Initialize_Arrays()
    Vali_MSE_avg, Vali_RMSE_avg, Vali_MAE_avg, Vali_R2_avg, Vali_acc_avg, Vali_f1_avg = Initialize_Arrays()
    Predict_MSE_avg, Predict_RMSE_avg, Predict_MAE_avg, Predict_R2_avg, Predict_acc_avg, Predict_f1_avg = Initialize_Arrays()

#   Determine the property to be predicted
    print("Enter name of property to be predicted")
    PChem_Property = input("Gibbs, LogCMC,  LogKow, LogKwa, LogS, pKa, MP, BP, VP, or HLC?: ")
    PChem_Property = PChem_Property.upper()

#   Determine type of model: Testing or Cross-Validation
    Vali_File_Available = "NO"
    Model_Type = input("Type of Prediction: Testing, K-Fold, LOO, or Predict: ")
    Model_Type = Model_Type.upper()   
    if(Model_Type == "K-FOLD"):
        K_Fold_CV = "KF"
        kfold_user = input("Enter number of folds: ") 
        kfold_user= int(kfold_user)
        Perform_Cross_Validation = "YES"
        Testing = "NO"
        Validation = "NO"
        Prediction = "NO"
    elif (Model_Type == "LOO"):
        K_Fold_CV = "LOO"
        Perform_Cross_Validation = "YES"
        Testing = "NO"
        Validation = "NO"
        Prediction = "NO"
    elif (Model_Type == "TESTING"):
        Testing = "YES"
        Perform_Cross_Validation = "NO"
        Prediction = "NO"
        Answer = input("Is a validation step required (yes/no)? ")
        Answer = Answer.upper()
        if (Answer == "YES"):
           Validation = "YES" 
           Answer1 = input("Is there a validation file available (yes/no)? ")
           Answer1 = Answer1.upper()
           if (Answer1 == "YES"):
               Vali_File_Available = "YES"
           else:
               Vali_File_Available = "NO"
        elif (Answer == "NO"):
            Validation = "NO"
    elif (Model_Type == "PREDICT"):
        Testing = "NO"
        Perform_Cross_Validation = "NO"
        Validation = "NO"
        Prediction ="YES"

#   Determine if log file of the IDE session will be created 
    Log_Answer = input("Do you want to create a log of the IDE session (Yes/No)?: ")
    Log_Answer = Log_Answer.strip().upper()
    # Anything other than an explicit YES means no log (previously an
    # unexpected answer left Create_Log undefined -> NameError later).
    Create_Log = "YES" if Log_Answer in ("YES", "Y") else "NO"

#   Setup the Templates
    test_prompt_template, CV_prompt_template, pchem_variable, prop_title = get_template_prompts(PChem_Property)

#   Setup File directories        
    training_file = dump_file = "" 
    JSONL_DIR = "/usr/local/projects/LLM/Data_Sets/"    
    if (PChem_Property == "GIBBS"):
        training_file = JSONL_DIR + "DeltaG_Solv/" + "Gibbs_Water.csv"
        validation_file = JSONL_DIR + "DeltaG_Solv/" + "Gibbs_Water_Val.csv"
        predict_file = JSONL_DIR + "DeltaG_Solv/" + "Gibbs_Water_Predict.csv"
        dump_file = JSONL_DIR + "DeltaG_Solv/" + "Optimal_Training_Set/optimal_samples.csv"
        if (Model_Type == "K-FOLD"):
            session_file = JSONL_DIR + "DeltaG_Solv/DeltaG_Solv_KFold.log"
        elif (Model_Type == "LOO"):
            session_file = JSONL_DIR + "DeltaG_Solv/DeltaG_Solv_LOO.log"
        elif (Model_Type == "TESTING"):
            session_file = JSONL_DIR + "DeltaG_Solv/DeltaG_Solv_Testing.log"
        Species_Name = "Solute"
        Property_Name = pchem_variable
    elif (PChem_Property == "LOGCMC"):
        training_file = JSONL_DIR +  "CMC/PFAS/LogCMC_PFAS_Benoit.csv"
        validation_file = JSONL_DIR + "CMC/PFAS/LogCMC_PFAS_Benoit_55.csv"
        predict_file = JSONL_DIR + "CMC/PFAS/LogCMC_PFAS_Benoit_Predict.csv"
        dump_file = JSONL_DIR + "CMC/PFAS/Optimal_Training_Set/optimal_samples.csv"
        if (Model_Type == "K-FOLD"):
            session_file = JSONL_DIR + "CMC/PFAS/PFAS_CMC_KFold.log"
        elif (Model_Type == "LOO"):
            session_file = JSONL_DIR + "CMC/PFAS/PFAS_CMC_LOO.log"
        elif (Model_Type == "TESTING"):
            session_file = JSONL_DIR + "CMC/PFAS/PFAS_CMC_Testing.log"
        elif (Model_Type == "PREDICT"):
            session_file = JSONL_DIR + "CMC/PFAS/PFAS_CMC_Predict.log"
        Species_Name = "Molecule"
        Property_Name = pchem_variable
    elif (PChem_Property == "MP"):
        training_file = JSONL_DIR + "PFAS/Martin/Melting_Point/PFAS_MP_Martin_Training.csv"
        validation_file = JSONL_DIR + "PFAS/Martin/Melting_Point/PFAS_MP_Martin_Validation.csv"
        predict_file = JSONL_DIR + "PFAS/Martin/Melting_Point/PFAS_MP_Martin_Predict.csv"
        dump_file = JSONL_DIR + "PFAS/Martin/Melting_Point/" + "Optimal_Training_Set/optimal_samples.csv"
        if (Model_Type == "K-FOLD"):
            session_file = JSONL_DIR + "PFAS/Martin/Melting_Point/PFAS_MP_KFold.log"
        elif (Model_Type == "LOO"):
            session_file = JSONL_DIR + "PFAS/Martin/Melting_Point/PFAS_MP_LOO.log"
        elif (Model_Type == "TESTING"):
            session_file = JSONL_DIR + "PFAS/Martin/Melting_Point/PFAS_MP_Testing.log"
        elif (Model_Type == "PREDICT"):
            session_file = JSONL_DIR + "PFAS/Martin/Melting_Point/PFAS_MP_Predict.log"
        Species_Name = "Molecule"
        Property_Name = pchem_variable
    elif (PChem_Property == "BP"):
        training_file = JSONL_DIR + "PFAS/Martin/Boiling_Point/PFAS_BP_Martin_Training.csv"
        validation_file = JSONL_DIR + "PFAS/Martin/Boiling_Point/PFAS_BP_Martin_Validation.csv"
        predict_file = JSONL_DIR + "PFAS/Martin/Boiling_Point/PFAS_BP_Martin_Predict.csv"
        dump_file = JSONL_DIR + "PFAS/Martin/Boiling_Point/" + "Optimal_Training_Set/optimal_samples.csv"
        if (Model_Type == "K-FOLD"):
            session_file = JSONL_DIR + "PFAS/Martin/Boiling_Point/PFAS_BP_KFold.log"
        elif (Model_Type == "LOO"):
            session_file = JSONL_DIR + "PFAS/Martin/Boiling_Point/PFAS_BP_LOO.log"
        elif (Model_Type == "TESTING"):
            session_file = JSONL_DIR + "PFAS/Martin/Boiling_Point/PFAS_BP_Testing.log"
        Species_Name = "Molecule"
        Property_Name = pchem_variable
    elif (PChem_Property == "LOGKOW"):
        training_file = JSONL_DIR + "PFAS/Martin/LogKow/PFAS_LogKow_Martin_Training.csv"
        validation_file = JSONL_DIR + "PFAS/Martin/LogKow/PFAS_LogKow_Martin_Validation.csv"
        predict_file = JSONL_DIR + "PFAS/Martin/LogKow/PFAS_LogKow_Martin_Predict.csv"
        dump_file = JSONL_DIR + "PFAS/Martin/LogKow/" + "Optimal_Training_Set/optimal_samples.csv"
        if (Model_Type == "K-FOLD"):
            session_file = JSONL_DIR + "PFAS/Martin/LogKow/PFAS_LogKow_KFold.log"
        elif (Model_Type == "LOO"):
            session_file = JSONL_DIR + "PFAS/Martin/LogKow/PFAS_LogKow_LOO.log"
        elif (Model_Type == "TESTING"):
            session_file = JSONL_DIR + "PFAS/Martin/LogKow/PFAS_LogKow_Testing.log"
        Species_Name = "Molecule"
        Property_Name = pchem_variable
    elif (PChem_Property == "LOGS"):
        training_file = JSONL_DIR + "PFAS/Martin/Solubility/PFAS_Sol_Martin_Training.csv"
        validation_file = JSONL_DIR + "PFAS/Martin/Solubility/PFAS_Sol_Martin_Validation.csv"
        predict_file = JSONL_DIR + "PFAS/Martin/Solubility/PFAS_Sol_Martin_Predict.csv"
        dump_file = JSONL_DIR + "PFAS/Martin/Solubility/Optimal_Training_Set/optimal_samples.csv"
        if (Model_Type == "K-FOLD"):
            session_file = JSONL_DIR + "PFAS/Martin/Solubility/PFAS_Sol_KFold.log"
        elif (Model_Type == "LOO"):
            session_file = JSONL_DIR + "PFAS/Martin/Solubility/PFAS_Sol_LOO.log"
        elif (Model_Type == "TESTING"):
            session_file = JSONL_DIR + "PFAS/Martin/Solubility/PFAS_Sol_Testing.log"
        Species_Name = "Molecule"
        Property_Name = pchem_variable
    elif (PChem_Property == "PKA"):
        training_file = JSONL_DIR + "PFAS/PKa/PFAS_pKa_Antle.csv"
        validation_file = JSONL_DIR + "PFAS/PKa/PKa/PFAS_pKa_Validation.csv"
        predict_file = JSONL_DIR + "PFAS/PKa/PFAS_pKa_Antle_Predict.csv"
        dump_file = JSONL_DIR + "PFAS/PKa/Optimal_Training_Set/optimal_samples.csv"
        if (Model_Type == "K-FOLD"):
            session_file = JSONL_DIR + "PFAS/PKa/PFAS_pKa_Antle_KFold.log"
        elif (Model_Type == "LOO"):
            session_file = JSONL_DIR + "PFAS/PKa/PFAS_pKa_Antle_LOO.log"
        elif (Model_Type == "TESTING"):
            session_file = JSONL_DIR + "PFAS/PKa/PFAS_pKa_Antle_Testing.log"
        Species_Name = "Molecule"
        Property_Name = pchem_variable
    elif (PChem_Property == "VP"):
        training_file = JSONL_DIR + "PFAS/Martin/Vapor_Pressure/PFAS_VP_Martin_Training.csv"
        validation_file = JSONL_DIR + "PFAS/Martin/Vapor_Pressure/PFAS_VP_Martin_Validation.csv"
        predict_file = JSONL_DIR + "PFAS/Martin/Vapor_Pressure/PFAS_VP_Martin_Predict.csv"
        dump_file = JSONL_DIR + "PFAS/Martin/Vapor_Pressure/" + "Optimal_Training_Set/optimal_samples.csv"
        if (Model_Type == "K-FOLD"):
            session_file = JSONL_DIR + "PFAS/Martin/Vapor_Pressure/PFAS_VP_KFold.log"
        elif (Model_Type == "LOO"):
            session_file = JSONL_DIR + "PFAS/Martin/Vapor_Pressure/PFAS_VP_LOO.log"
        elif (Model_Type == "TESTING"):
            session_file = JSONL_DIR + "PFAS/Martin/Vapor_Pressure/PFAS_VP_Testing.log"
        elif (Model_Type == "PREDICT"):
            session_file = JSONL_DIR + "PFAS/Martin/Vapor_Pressure/PFAS_VP_Predict.log"
        Species_Name = "Molecule"
        Property_Name = pchem_variable
    elif (PChem_Property == "HLC"):
        training_file = JSONL_DIR + "PFAS/Martin/Henry_Law/PFAS_HL_Martin_Training.csv"
        validation_file = JSONL_DIR + "PFAS/Martin/Henry_Law/PFAS_HL_Martin_Validation.csv"
        predict_file = JSONL_DIR + "PFAS/Martin/Henry_Law/PFAS_HL_Martin_Predict.csv"
        dump_file = JSONL_DIR + "PFAS/Martin/Henry_Law/" + "Optimal_Training_Set/optimal_samples.csv"
        if (Model_Type == "K-FOLD"):
            session_file = JSONL_DIR + "PFAS/Martin/Henry_Law/PFAS_HL_KFold.log"
        elif (Model_Type == "LOO"):
            session_file = JSONL_DIR + "PFAS/Martin/Henry_Law/PFAS_HL_LOO.log"
        elif (Model_Type == "TESTING"):
            session_file = JSONL_DIR + "PFAS/Martin/Henry_Law/PFAS_HL_Testing.log"
        Species_Name = "Molecule"
        Property_Name = pchem_variable
    elif (PChem_Property == "LOGKWA"):
        training_file = JSONL_DIR + "PFAS/Becky/Kwa/PFAS_LogKwa.csv"
        validation_file = JSONL_DIR + "PFAS/Becky/Kwa/PFAS_LogKwa_Validation.csv"
        predict_file = JSONL_DIR + "PFAS/Becky/Kwa/PFAS_LogKwa_Predict.csv"
        dump_file = JSONL_DIR + "PFAS/Becky/Kwa/" + "Optimal_Training_Set/optimal_samples.csv"
        if (Model_Type == "K-FOLD"):
            session_file = JSONL_DIR + "PFAS/Becky/Kwa/PFAS_LogKwa_KFold.log"
        elif (Model_Type == "LOO"):
            session_file = JSONL_DIR + "PFAS/Becky/Kwa/PFAS_LogKwa_LOO.log"
        elif (Model_Type == "TESTING"):
            session_file = JSONL_DIR + "PFAS/Becky/Kwa/PFAS_LogKwa_Testing.log"
        Species_Name = "Molecule"
        Property_Name = pchem_variable

#   Dump Spyder Session into a file
    # Not every (property, mode) branch above defines session_file (e.g.
    # PREDICT for several properties).  Fall back to a log named after
    # the property and mode in the training file's directory.
    if (Create_Log == "YES") and ("session_file" not in globals()):
        session_file = os.path.join(
            os.path.dirname(training_file) or ".",
            f"{PChem_Property}_{Model_Type}.log")
    if(Create_Log == "YES"):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_file = open(session_file, 'w')        
        sys.stdout = log_file
        sys.stderr = log_file

#   Load the data into a Pandas DataFrame    
    data_set = read_csv_file(training_file)   
    df = pd.DataFrame(data_set)

    median, mean, std, min_value, max_value = whole_dataset_stat(df, pchem_variable, Save)
    print("#######################################################")
    print(f"# Median to be used for Binary Classification: {float(median):.4f} #")
    print("#######################################################\n")

#   Plot the Class Distributions of the original dataset before the split
    Data_for_Plot = df[pchem_variable].astype(float)
    base_title = "Whole Dataset Distribution of "
    title = base_title + prop_title
    plot_distribution(Data_for_Plot, title, pchem_variable)

#   Perform a split  
    if (Testing == "YES"):
        test_split = input("Enter percentage of dataset to be used as testing: ")
        test_split = float(test_split)
        test_split = test_split/100.00
        if (Validation == "YES"): 
            if (Vali_File_Available == "NO"): 
                validation_split = input("Enter percentage of dataset to be used as validation: ")
                validation_split = float(validation_split)
                validation_split = validation_split/100.00
        training_data, test_data = train_test_split(df, test_size=test_split)
    elif(Perform_Cross_Validation == "YES"):
        training_data = df

#   Plot the Class Distributions of the training dataset before the split
    if(Prediction == "NO"):
        Data_for_Plot = training_data[pchem_variable].astype(float)
        base_title = "Training Dataset Distribution of "
        title = base_title + prop_title
        plot_distribution(Data_for_Plot, title, pchem_variable)

    if(Validation == "YES"):
        if(Vali_File_Available == "YES"):
            vali_data_set = read_csv_file(validation_file)
            vali_pool = pd.DataFrame(vali_data_set)
        else:
            testing_pool, vali_pool = train_test_split(test_data, test_size=validation_split)
            test_data = testing_pool

        Data_for_Plot = vali_pool[pchem_variable].astype(float)               
        base_title = "Validation Set Distribution of "
        title = base_title + prop_title
        plot_distribution(Data_for_Plot, title, pchem_variable)

        validation_samples = vali_pool.to_dict('records')
        N_vali = len(validation_samples)

#   Conver data to dict             
    if(Testing == "YES"): 
        training_pool = training_data.to_dict('records')
        test_pool = test_data.to_dict('records')
        N_test = len(test_pool)
    elif(Perform_Cross_Validation == "YES"):
        training_pool = df.to_dict('records')

#   Compute the length of the training and the testing datasets
    if (Prediction == "NO"): N_train = len(training_pool)


# ***************    Beginning of Model Task Block    *******************

# =============================================================================
#                              Cross-Validation   
# =============================================================================
    if(Perform_Cross_Validation == "YES"):
        if(K_Fold_CV == "LOO"):
            k_shot = N_train
            kfold = k_shot
            N_total = (round((k_shot*kfold)/(kfold-1))) - 1
            n_k = round(N_total/kfold)
            training_samples = random.sample(training_pool, N_total)
            Memory = "NO"
            tr_MSE, tr_RMSE, tr_MAE, tr_R2, tr_acc, tr_f1, tr_rejects = Cross_Validation(training_samples, median, k_shot, kfold, N_total, CV_prompt_template, Species_Name, Property_Name, Rate, Memory, model, temperature=llm_temp)
            train_rejects += tr_rejects          
            Mode = "TRAIN"
            print_stat(tr_MSE, tr_RMSE, tr_MAE, tr_R2, tr_acc, tr_f1, Mode)      
            sys.stdout.flush()
        elif(K_Fold_CV == "KF"):
            k_shot = kfold = N_train
            N_total = (round((k_shot*kfold)/(kfold-1))) - 1
            n_k = round(N_total/kfold)
            kfold = kfold_user 
            Memory = "NO"
            for ncv_cycle in range(Max_Cycles):
                training_samples = random.sample(training_pool, N_total)
                current_datetime = datetime.datetime.now()
                formatted_time = current_datetime.strftime("%A, %B, %d %Y, %I:%M %p")
                print("                  --------------------------")
                print(f"                  |     Cycle Number: {ncv_cycle+1}   |")
                print("                  --------------------------")    
                print(formatted_time)
                print("\n")
                tr_MSE, tr_RMSE, tr_MAE, tr_R2, tr_acc, tr_f1, tr_rejects = Cross_Validation(training_samples, median, k_shot, kfold, N_total, CV_prompt_template, Species_Name, Property_Name, Rate, Memory, model, temperature=llm_temp)
                train_rejects += tr_rejects          
                Mode = "TRAIN"
                print_stat(tr_MSE, tr_RMSE if False else tr_RMSE, tr_MAE, tr_R2, tr_acc, tr_f1, Mode)            
                Train_MSE_avg.append(tr_MSE)
                Train_RMSE_avg.append(tr_RMSE)
                Train_MAE_avg.append(tr_MAE)
                Train_R2_avg.append(tr_R2)
                Train_acc_avg.append(tr_acc)
                Train_f1_avg.append(tr_f1)
                sys.stdout.flush()

                cycle_cpu = time.process_time()
                cycle_elapsed = time.time()
                cycle_cpu_time = cycle_cpu - start_cpu
                cycle_elapsed_time = cycle_elapsed - start_elapsed
                print(f"CPU time for this cycle: {int(cycle_cpu_time//3600)}h {int((cycle_cpu_time%3600)//60)}m {int(cycle_cpu_time%60)}s")
                sys.stdout.flush()
                print(f"Elapsed time for this cycle: {int(cycle_elapsed_time//3600)}h {int((cycle_elapsed_time%3600)//60)}m {int(cycle_elapsed_time%60)}s")
                sys.stdout.flush()
                print("\n")
                sys.stdout.flush()

            tr_Avg_MSE,tr_Avg_RMSE,tr_Avg_MAE,tr_Avg_R2,tr_Avg_acc,tr_Avg_f1 = array_means(Train_MSE_avg, Train_RMSE_avg, Train_MAE_avg, Train_R2_avg, Train_acc_avg, Train_f1_avg)
            tr_NRMSE_mean = (tr_Avg_RMSE/abs(mean))*100.0
            tr_NRMSE_range = (tr_Avg_RMSE/(max_value - min_value))*100.0  

            tr_MSE_sd = np.std(tr_Avg_MSE,ddof=1) 
            tr_RMSE_sd = np.std(tr_Avg_RMSE,ddof=1)
            tr_MAE_sd = np.std(tr_Avg_MAE,ddof=1)
            tr_R2_sd = np.std(tr_Avg_R2,ddof=1)    

            Mode = "TRAIN"           
            print_summary(model, Max_Cycles, k_shot, kfold, n_k, N_total, tr_Avg_MSE, tr_MSE_sd, tr_Avg_RMSE, tr_RMSE_sd, tr_Avg_MAE, tr_MAE_sd, tr_Avg_R2, tr_R2_sd, tr_Avg_acc, tr_Avg_f1, tr_NRMSE_mean, tr_NRMSE_range, Mode) 
            print(f"Number of rejected Cross-Validation samples = {train_rejects}")
            sys.stdout.flush()

# =============================================================================
#                                    Testing 
# =============================================================================
    if(Testing == "YES"): 
        N_total = min(50,N_train)
        Memory = "NO"
        for ncycle in range(Max_Cycles):
            random.seed(ncycle)
            training_samples = random.sample(training_pool, N_total)
            test_samples = random.sample(test_pool, N_test)
            current_datetime = datetime.datetime.now()
            formatted_time = current_datetime.strftime("%A, %B, %d %Y, %I:%M %p")
            print("                  --------------------------")
            print(f"                  |     Cycle Number: {ncycle+1}   |")
            print("                  --------------------------")  
            print(formatted_time)
            print("\n")

            te_MSE, te_RMSE, te_MAE, te_R2, te_acc, te_f1, te_rejects = k_shot_test(training_samples, test_samples, median, test_prompt_template, Species_Name, Property_Name, Rate, Memory, model, temperature=llm_temp)
            test_rejects += te_rejects        
            Mode = "TEST"
            print_stat(te_MSE, te_RMSE, te_MAE, te_R2, te_acc, te_f1, Mode)
            sys.stdout.flush()

            cycle_cpu = time.process_time()
            cycle_elapsed = time.time()
            cycle_cpu_time = cycle_cpu - start_cpu
            cycle_elapsed_time = cycle_elapsed - start_elapsed
            print(f"CPU time for this cycle: {int(cycle_cpu_time//3600)}h {int((cycle_cpu_time%3600)//60)}m {int(cycle_cpu_time%60)}s")
            sys.stdout.flush()
            print(f"Elapsed time for this cycle: {int(cycle_elapsed_time//3600)}h {int((cycle_elapsed_time%3600)//60)}m {int(cycle_elapsed_time%60)}s")
            sys.stdout.flush()
            print("\n")
            sys.stdout.flush()

            Test_MSE_avg.append(te_MSE)
            Test_RMSE_avg.append(te_RMSE)
            Test_MAE_avg.append(te_MAE)
            Test_R2_avg.append(te_R2)
            Test_acc_avg.append(te_acc)
            Test_f1_avg.append(te_f1)

#   Store the training sample data with lowest test MAE
            if (te_MAE < MAE_Optimal):
                optimal_training_samples = training_samples
                MAE_Optimal = te_MAE    

        te_Avg_MSE,te_Avg_RMSE,te_Avg_MAE,te_Avg_R2,te_Avg_acc,te_Avg_f1 = array_means(Test_MSE_avg, Test_RMSE_avg, Test_MAE_avg, Test_R2_avg, Test_acc_avg, Test_f1_avg)
        te_NRMSE_mean = (te_Avg_RMSE/abs(mean))*100.0
        te_NRMSE_range = (te_Avg_RMSE/(max_value - min_value))*100.0 

        te_MSE_sd = np.std(te_Avg_MSE,ddof=1) 
        te_RMSE_sd = np.std(te_Avg_RMSE,ddof=1)
        te_MAE_sd = np.std(te_Avg_MAE,ddof=1)
        te_R2_sd = np.std(te_Avg_R2,ddof=1)    

        Mode = "TEST"
        k_shot = kfold = n_k = 0
        print_summary(model, Max_Cycles, k_shot, kfold, n_k, N_test, te_Avg_MSE, te_MSE_sd, te_Avg_RMSE, te_RMSE_sd, te_Avg_MAE, te_MAE_sd, te_Avg_R2, te_R2_sd , te_Avg_acc, te_Avg_f1, te_NRMSE_mean, te_NRMSE_range, Mode)
        print(f"Number of rejected testing samples = {test_rejects}")
        sys.stdout.flush()

# =============================================================================
#                                      Validation
# =============================================================================        
    if (Validation == 'YES'):
        Memory = "NO"
        Mode = "VALIDATION"
        if(len(optimal_training_samples) != 0):
            print("Using Optimal Training Set from best Testing cycle "
                  f"(MAE = {MAE_Optimal:.4f})")
            training_samples = optimal_training_samples
        else:
            print("[NOTE] No optimal training set from Testing; using "
                  "the currently bound training_samples for validation. "
                  "This usually means Testing was skipped or every "
                  "cycle rejected every prediction.")
        va_MSE, va_RMSE, va_MAE, va_R2, va_acc, va_f1, va_rejects = k_shot_test(training_samples, validation_samples, median, test_prompt_template, Species_Name, Property_Name, Rate, Memory, model, temperature=llm_temp)
        vali_rejects += va_rejects
        print("\n")
        print_stat(va_MSE, va_RMSE, va_MAE, va_R2, va_acc, va_f1, Mode)
        print(f"Number of rejected validation samples = {vali_rejects}")
        sys.stdout.flush()

#   Dump the dataset with optimal training samples into a csv file    
    if (Testing == "YES"):
        length_ots = len(optimal_training_samples)
        if length_ots > 0:
            with open(dump_file, 'w', newline='') as optimal:
                fieldnames = optimal_training_samples[0].keys()
                writer = csv.DictWriter(optimal, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(optimal_training_samples)
            print(f"Optimal training samples ({length_ots} molecules) "
                  f"written to {dump_file}")
        else:
            print(f"\n[NOTE] No optimal training samples to dump: no "
                  f"Testing cycle produced a valid MAE < {MAE_Optimal}. "
                  f"Skipping {dump_file}.")

# =============================================================================
#                              Prediction
# =============================================================================
    if (Prediction == "YES"):
        prediction_rejects = 0
        Memory = "NO"
        Mode = "PREDICT"
        train_data_set = read_csv_file(dump_file)   
        df_train = pd.DataFrame(train_data_set)
        df_training_pool = df_train.to_dict('records')

        predict_data_set = read_csv_file(validation_file)   
        df_predict = pd.DataFrame(predict_data_set)
        df_predict_pool = df_predict.to_dict('records')

        pred_MSE, pred_RMSE, pred_MAE, pred_R2, pred_acc, pred_f1, pred_rejects = k_shot_predict(df_training_pool, df_predict_pool, median, test_prompt_template, Species_Name, Property_Name, Rate, Memory, model, predict_file, temperature=llm_temp)
        prediction_rejects += pred_rejects
        print("\n")
        print_stat(pred_MSE, pred_RMSE, pred_MAE, pred_R2, pred_acc, pred_f1, Mode)
        print(f"Number of rejected from prediction samples = {prediction_rejects}")
        sys.stdout.flush()

# *********************     End of Model Task Block    ************************

#   Print the values for N_train, N_test, and N_Total        
    if(Prediction == "NO"):   
        print(f"Length of Training Set: {N_train} ")
        if(Testing == "YES"): print(f"Length of Testing Set: {N_test}")
        if(Validation == "YES"): print(f"Length of Validation Set: {N_vali}")
        print(f"Number of Prompting Data Points Used on Each Cycle: {N_total}") 

#   Print the PChem Property 
    print(f"Property: {PChem_Property}\n")

#    At the end of the loop stop the clock counter and record the cpu & elapsed time   
    end_cpu = time.process_time()
    end_elapsed = time.time()
    cpu_time = end_cpu - start_cpu
    elapsed_time = end_elapsed - start_elapsed
    print(f"Completions requests per minute: {Rate}")
    print(f"CPU time: {int(cpu_time//3600)}h {int((cpu_time%3600)//60)}m {int(cpu_time%60)}s")
    print(f"Elapsed time: {int(elapsed_time//3600)}h {int((elapsed_time%3600)//60)}m {int(elapsed_time%60)}s")
    sys.stdout.flush()

#   Close session log file
    if(Create_Log == "YES"):   
        sys.stdout.flush()
        sys.stdout = original_stdout
        sys.stdout.close()
        sys.exit()
