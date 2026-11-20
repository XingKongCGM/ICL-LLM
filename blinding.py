#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blinding.py

Created on Tue Aug 26 14:17:09 2025

@author: Carlos Gonzalez
Material Data Division,
National Institute of Standatds and Technology,
Gaithersburg, MD. 20889


===========

Blinding / transformation controls for the PFAS LLM-ICL study.

This module implements the "controlled information-access experiment"
described in the project notes: for each (model, property) cell it can
run three prompt conditions that progressively strip away information a
model might exploit from pretraining rather than from the in-context
demonstrations:

  1. NATIVE     - standard SMILES, explicit property name + units,
                  original target values.  (Baseline; delegates to the
                  existing RCHAT prompt templates.)

  2. GENERIC    - standard SMILES, anonymous response identifier (e.g.
                  "Response R-17"), labels passed through an invertible
                  map: z-score shift (recommended, whole-real-line
                  inverse), invert + rescale to [0,100], or a nonlinear
                  tanh map.  Removes named-property priors and familiar
                  raw target scales.

  3. AGNOSTIC   - deterministically recoded SMILES tokens, anonymous
                  identifier, transformed labels.  Additionally hides
                  ordinary SMILES / chemical semantics.

Design principle
----------------

This module is deliberately self-contained and side-effect free.  It
does NOT import RCHAT_PChem_Properties, does NOT touch module globals,
and does NOT call any model.  It provides:

  * Label transforms         (LinearStandardize, LinearInvertRescale,
                              TanhStandardize, Identity) with exact
                              analytic inverses.
  * A SMILES token recoder    (SmilesRecoder) that preserves formal
                              structure while suppressing chemical
                              identity.
  * Prompt builders           (build_prompt) for each condition, matched
                              in length / instruction structure / prefill
                              so the only planned variable is the
                              information content.
  * A single per-cell config  (BlindingSpec) that bundles condition +
                              transform + recoder so the run driver can
                              pass one object around.

Every transform derives its parameters from the TRAINING POOL ONLY, per
cycle, and is applied identically to demonstrations, test queries, and
validation queries of that aligned cycle.  Predictions are inverse-
transformed back to native units before any metric is computed.

No em dashes are used in output strings (project style).
"""

from __future__ import annotations

import re
import math
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ===========================================================================
#                       Condition name constants
# ===========================================================================

NATIVE = "native"
GENERIC = "generic"
AGNOSTIC = "agnostic"

VALID_CONDITIONS = (NATIVE, GENERIC, AGNOSTIC)

# The anonymous response identifier used in blinded prompts.  Chosen to
# be a neutral token that does not encode the property identity.  Kept
# as a module constant so prompt builder and parser agree.
RESPONSE_ID = "R-17"

# Words that must NOT appear in blinded prompts.  Used by a self-check
# in build_prompt() so an accidental leak of the property identity fails
# loudly rather than silently contaminating the control.
_LEAK_WORDS = (
    "pfas", "molecule", "molecular", "chemical", "chemistry",
    "solubility", "soluble", "partition", "octanol", "water",
    "acid", "dissociation", "surfactant", "micelle", "micellar",
    "boiling", "melting", "vapor", "vapour", "pressure", "henry",
    "smiles", "logkow", "logs", "logcmc", "pka", "hlc", "bp", "mp",
    "vp", "kow", "cmc", "fluor", "perfluoro", "celsius", "mmhg",
    "atm", "mol", "property", "physicochemical",
)


# ===========================================================================
#                          Label transforms
# ===========================================================================
#
# Each transform exposes:
#     fit(y_train)              -> sets internal parameters from TRAIN only
#     forward(y)                -> native  -> transformed   (scalar or list)
#     inverse(z)               -> transformed -> native    (scalar or list)
#     describe()               -> dict for logging / SI reproducibility
#
# forward/inverse operate elementwise and accept either a single float or
# a sequence of floats, returning the same shape.  They never clip: an
# out-of-range model response is inverse-transformed as-is, and the
# caller is responsible for tallying the out-of-range rate.
# ===========================================================================


def _as_list(x):
    """Return (list_of_floats, was_scalar)."""
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x], False
    return [float(x)], True


def _restore(vals, was_scalar):
    return vals[0] if was_scalar else vals


class LabelTransform:
    """Abstract base.  Subclasses must set self._fitted = True in fit()."""

    name = "abstract"

    def __init__(self):
        self._fitted = False

    def fit(self, y_train: Sequence[float]) -> "LabelTransform":
        raise NotImplementedError

    def forward(self, y):
        raise NotImplementedError

    def inverse(self, z):
        raise NotImplementedError

    def describe(self) -> Dict:
        raise NotImplementedError

    def _check(self):
        if not self._fitted:
            raise RuntimeError(
                f"{self.name} transform used before fit(); call fit() on "
                f"the cycle training pool first."
            )


class Identity(LabelTransform):
    """No-op transform.  Used by the NATIVE condition so the same code
    path serves all conditions."""

    name = "identity"

    def fit(self, y_train):
        self._fitted = True
        return self

    def forward(self, y):
        return y

    def inverse(self, z):
        return z

    def describe(self):
        return {"name": self.name}


class LinearInvertRescale(LabelTransform):
    """
    Invert and rescale native targets to a fixed [0, out_max] window,
    using the TRAIN-pool min/max only.

        z = out_max * (1 - (y - y_min) / (y_max - y_min))

    Inverse:

        y = y_min + (1 - z / out_max) * (y_max - y_min)

    The inversion (the "1 -" term) flips the ordering so that a model
    cannot rely on "larger native value -> larger response" being the
    familiar direction; combined with the [0,100] rescale it removes the
    familiar raw numeric scale as a cue.
    """

    name = "linear_invert_rescale"

    def __init__(self, out_max: float = 100.0):
        super().__init__()
        self.out_max = float(out_max)
        self.y_min = None
        self.y_max = None

    def fit(self, y_train):
        vals = [float(v) for v in y_train]
        if not vals:
            raise ValueError("LinearInvertRescale.fit got empty y_train.")
        self.y_min = min(vals)
        self.y_max = max(vals)
        if self.y_max == self.y_min:
            # Degenerate: all training labels equal.  Widen by a tiny
            # epsilon so the map is invertible.  This is extremely
            # unlikely for continuous PFAS properties but guarded anyway.
            self.y_max = self.y_min + 1e-9
        self._fitted = True
        return self

    def forward(self, y):
        self._check()
        vals, scalar = _as_list(y)
        span = self.y_max - self.y_min
        z = [self.out_max * (1.0 - (v - self.y_min) / span) for v in vals]
        return _restore(z, scalar)

    def inverse(self, z):
        self._check()
        vals, scalar = _as_list(z)
        span = self.y_max - self.y_min
        y = [self.y_min + (1.0 - v / self.out_max) * span for v in vals]
        return _restore(y, scalar)

    def in_range(self, z) -> bool:
        """Whether a transformed response falls inside [0, out_max]."""
        return 0.0 <= float(z) <= self.out_max

    def describe(self):
        return {
            "name": self.name,
            "out_max": self.out_max,
            "y_min_train": self.y_min,
            "y_max_train": self.y_max,
        }


class TanhStandardize(LabelTransform):
    """
    Nonlinear monotonic invertible control (optional stress test):

        z = center + amp * tanh( (y - mu_train) / s_train )

    Inverse (valid while z is strictly inside (center-amp, center+amp)):

        y = mu_train + s_train * atanh( (z - center) / amp )

    Because atanh diverges at +/-1, an out-of-window model response has
    no finite native value.  We clamp the atanh argument to
    +/-(1 - eps) for the inverse and flag such responses as out-of-range
    so the caller can tally them; this keeps scoring finite without
    silently discarding data.
    """

    name = "tanh_standardize"

    def __init__(self, center: float = 50.0, amp: float = 40.0,
                 eps: float = 1e-6):
        super().__init__()
        self.center = float(center)
        self.amp = float(amp)
        self.eps = float(eps)
        self.mu = None
        self.s = None

    def fit(self, y_train):
        vals = [float(v) for v in y_train]
        if not vals:
            raise ValueError("TanhStandardize.fit got empty y_train.")
        n = len(vals)
        self.mu = sum(vals) / n
        if n > 1:
            var = sum((v - self.mu) ** 2 for v in vals) / (n - 1)
            self.s = math.sqrt(var)
        else:
            self.s = 0.0
        if self.s == 0.0:
            self.s = 1e-9
        self._fitted = True
        return self

    def forward(self, y):
        self._check()
        vals, scalar = _as_list(y)
        z = [self.center + self.amp * math.tanh((v - self.mu) / self.s)
             for v in vals]
        return _restore(z, scalar)

    def inverse(self, z):
        self._check()
        vals, scalar = _as_list(z)
        out = []
        for v in vals:
            arg = (v - self.center) / self.amp
            # Clamp into (-1, 1) so atanh stays finite.
            hi = 1.0 - self.eps
            arg = max(-hi, min(hi, arg))
            out.append(self.mu + self.s * math.atanh(arg))
        return _restore(out, scalar)

    def in_range(self, z) -> bool:
        arg = (float(z) - self.center) / self.amp
        return -1.0 < arg < 1.0

    def describe(self):
        return {
            "name": self.name,
            "center": self.center,
            "amp": self.amp,
            "mu_train": self.mu,
            "s_train": self.s,
            "eps": self.eps,
        }


class LinearStandardize(LabelTransform):
    """
    Standardize-and-shift transform with a WHOLE-REAL-LINE inverse.

        z = center + scale * (y - mu_train) / sigma_train

    Inverse (defined for every real z; no boundary, never out of range):

        y = mu_train + (sigma_train / scale) * (z - center)

    Purpose.  This is the corrected blinded-label map. Unlike the
    [0,100] invert-rescale (LinearInvertRescale) and the tanh map
    (TanhStandardize), the inverse here is a plain affine function with
    no finite window, so EVERY model output maps back to a finite native
    value. That removes the "out-of-range" failure mode entirely: an
    elevated blinded MAE can then only reflect how well the model learned
    the in-context mapping, not whether it stayed inside a target window.

    It still strips the two cues the generic condition targets:
      * property name / units -> handled by the anonymous R-17 label
        (unchanged, in the prompt builder),
      * familiar native scale  -> values are recentered on `center` and
        rescaled by `scale`, so the raw numbers no longer resemble any
        native property range.

    Choose center/scale so transformed values look unlike any native
    property range (defaults 500 +/- 100 -> typically ~300 to ~700).
    Use a NEGATIVE scale if you also want to flip the ordering as an
    extra cue-removal; the inverse formula is unchanged and still whole
    line.
    """

    name = "linear_standardize"

    def __init__(self, center: float = 500.0, scale: float = 100.0):
        super().__init__()
        self.center = float(center)
        if scale == 0:
            raise ValueError("LinearStandardize scale must be nonzero.")
        self.scale = float(scale)
        self.mu = None
        self.sigma = None

    def fit(self, y_train):
        vals = [float(v) for v in y_train]
        if not vals:
            raise ValueError("LinearStandardize.fit got empty y_train.")
        n = len(vals)
        self.mu = sum(vals) / n
        if n > 1:
            var = sum((v - self.mu) ** 2 for v in vals) / (n - 1)
            self.sigma = math.sqrt(var)
        else:
            self.sigma = 0.0
        if self.sigma == 0.0:
            # Degenerate (all training labels equal). Widen slightly so
            # the map is invertible; vanishingly unlikely for continuous
            # PFAS properties but guarded for safety.
            self.sigma = 1e-9
        self._fitted = True
        return self

    def forward(self, y):
        self._check()
        vals, scalar = _as_list(y)
        z = [self.center + self.scale * (v - self.mu) / self.sigma
             for v in vals]
        return _restore(z, scalar)

    def inverse(self, z):
        self._check()
        vals, scalar = _as_list(z)
        y = [self.mu + (self.sigma / self.scale) * (v - self.center)
             for v in vals]
        return _restore(y, scalar)

    def in_range(self, z) -> bool:
        # Whole-real-line inverse: every output is representable.
        return True

    def describe(self):
        return {
            "name": self.name,
            "center": self.center,
            "scale": self.scale,
            "mu_train": self.mu,
            "sigma_train": self.sigma,
        }


def make_transform(kind: str, **kwargs) -> LabelTransform:
    """Factory.  kind in {'identity','linear','tanh','zscore'}."""
    k = kind.lower()
    if k in ("identity", "none", NATIVE):
        return Identity()
    if k in ("linear", "linear_invert_rescale", "invert"):
        return LinearInvertRescale(out_max=kwargs.get("out_max", 100.0))
    if k in ("tanh", "tanh_standardize", "nonlinear"):
        return TanhStandardize(
            center=kwargs.get("center", 50.0),
            amp=kwargs.get("amp", 40.0),
            eps=kwargs.get("eps", 1e-6),
        )
    if k in ("zscore", "linear_standardize", "standardize", "wholeline"):
        return LinearStandardize(
            center=kwargs.get("std_center", 500.0),
            scale=kwargs.get("std_scale", 100.0),
        )
    raise ValueError(f"Unknown transform kind {kind!r}; "
                     f"expected identity|linear|tanh|zscore.")


# ===========================================================================
#                          SMILES recoder
# ===========================================================================


# SMILES token grammar (order matters: multi-char and bracketed forms
# must be tried before single chars).  This tokenizer is intentionally
# conservative: it splits a SMILES string into a stream of typed tokens
# so the recoder can substitute only ATOM tokens while leaving bonds,
# branches, ring digits, and stereo/charge syntax untouched.
_TWO_CHAR_ATOMS = ("Cl", "Br")
_ORGANIC_SUBSET = ("B", "C", "N", "O", "S", "P", "F", "I",
                   "b", "c", "n", "o", "s", "p")

# Bracket-atom body pattern, e.g. [O-], [NH+], [C@@H], [13CH3], [Si].
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
# A single element symbol inside a bracket body, e.g. "O" in "[O-]" or
# "Si" in "[Si]" or "C" in "[C@@H]".  We capture the leading element.
_BRACKET_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


@dataclass
class _Token:
    kind: str    # "atom", "bracket", "bond", "branch", "ring", "dot", "other"
    text: str


def tokenize_smiles(smiles: str) -> List[_Token]:
    """
    Split a SMILES string into typed tokens.  Not a full SMILES parser;
    it is a lexical splitter adequate for token-level recoding that
    preserves formal structure.
    """
    toks: List[_Token] = []
    i = 0
    n = len(smiles)
    while i < n:
        ch = smiles[i]

        # Bracketed atom: consume up to matching ]
        if ch == "[":
            m = _BRACKET_RE.match(smiles, i)
            if m:
                toks.append(_Token("bracket", m.group()))
                i = m.end()
                continue
            # Unbalanced bracket: treat as other
            toks.append(_Token("other", ch))
            i += 1
            continue

        # Two-char organic atoms
        if i + 1 < n and smiles[i:i + 2] in _TWO_CHAR_ATOMS:
            toks.append(_Token("atom", smiles[i:i + 2]))
            i += 2
            continue

        # Single-char organic-subset atoms (incl. aromatic lowercase)
        if ch in _ORGANIC_SUBSET:
            toks.append(_Token("atom", ch))
            i += 1
            continue

        # Bonds
        if ch in ("-", "=", "#", "$", ":", "/", "\\"):
            toks.append(_Token("bond", ch))
            i += 1
            continue

        # Branch open/close
        if ch in ("(", ")"):
            toks.append(_Token("branch", ch))
            i += 1
            continue

        # Ring-closure digit or %nn
        if ch.isdigit():
            toks.append(_Token("ring", ch))
            i += 1
            continue
        if ch == "%" and i + 2 < n and smiles[i + 1:i + 3].isdigit():
            toks.append(_Token("ring", smiles[i:i + 3]))
            i += 3
            continue

        # Disconnected-component separator
        if ch == ".":
            toks.append(_Token("dot", ch))
            i += 1
            continue

        # Anything else (wildcards, unexpected chars)
        toks.append(_Token("other", ch))
        i += 1

    return toks


class SmilesRecoder:
    """
    Deterministic one-to-one token-level SMILES recoder.

    The recoder builds a fixed substitution dictionary over the ATOM
    alphabet actually present across a corpus of SMILES (both organic-
    subset single/double-char atoms and the leading element of each
    bracket atom).  It then rewrites each string, substituting atom
    identities while preserving:

      * token order
      * bond sequence
      * branch (parenthesis) structure
      * ring-closure digits and their correspondences
      * bracket bodies except the leading element symbol
      * formal charge, H-count, stereo markers inside brackets
      * disconnected-component dots

    The recoded strings are syntactically consistent under this abstract
    encoding but are no longer chemically valid SMILES, which is the
    intent: it suppresses learned SMILES semantics while retaining a
    consistent formal pattern that context learning could still exploit.

    One GLOBAL mapping is built for the whole experiment (recommended in
    the notes) so that a given atom maps to the same code everywhere,
    across cycles and properties.

    The mapping is seeded deterministically so runs are reproducible and
    the dictionary can be dumped to the Supporting Information.
    """

    # A pool of substitute element-like symbols that are (a) valid as a
    # leading bracket element so the bracket stays lexically well formed,
    # and (b) not in the PFAS-relevant organic subset, so the encoding
    # visibly departs from real chemistry.  We use two-letter symbols to
    # avoid collision with aromatic-lowercase single chars.
    _SUBSTITUTE_POOL = (
        "Xa", "Xb", "Xc", "Xd", "Xe", "Xf", "Xg", "Xh", "Xi", "Xj",
        "Xk", "Xl", "Xm", "Xn", "Xo", "Xp", "Xq", "Xr", "Xs", "Xt",
        "Xu", "Xv", "Xw", "Xx", "Xy", "Xz", "Ya", "Yb", "Yc", "Yd",
        "Ye", "Yf", "Yg", "Yh", "Yi", "Yj", "Yk", "Yl", "Ym", "Yn",
    )

    def __init__(self, seed: int = 20260101):
        self.seed = int(seed)
        self._map: Dict[str, str] = {}
        self._locked = False

    # ---- dictionary construction ------------------------------------

    def _atom_alphabet(self, smiles_list: Sequence[str]) -> List[str]:
        """Collect the set of atom identities across a corpus.  An atom
        identity is either an organic-subset token (e.g. 'C', 'Cl', 'c')
        or the leading element of a bracket atom (e.g. 'O' from '[O-]',
        'Si' from '[Si]')."""
        alpha = set()
        for s in smiles_list:
            for t in tokenize_smiles(s):
                if t.kind == "atom":
                    alpha.add(t.text)
                elif t.kind == "bracket":
                    body = t.text[1:-1]  # strip [ ]
                    m = _BRACKET_ELEMENT_RE.match(body)
                    if m:
                        alpha.add("[elem]" + m.group(1))
        # Deterministic ordering: sort so seed-based assignment is stable
        return sorted(alpha)

    def build(self, smiles_list: Sequence[str]) -> "SmilesRecoder":
        """Build the global substitution dictionary from a corpus.  Call
        once with ALL SMILES (train + test + validation across every
        property) so the mapping is global and consistent."""
        alphabet = self._atom_alphabet(smiles_list)

        # Deterministic shuffle of the substitute pool via a seeded
        # ordering derived from a hash, avoiding a dependency on the
        # global random module state.
        pool = list(self._SUBSTITUTE_POOL)
        def _rank(sym):
            h = hashlib.sha256(f"{self.seed}:{sym}".encode()).hexdigest()
            return int(h[:16], 16)
        pool.sort(key=_rank)

        if len(alphabet) > len(pool):
            raise ValueError(
                f"SMILES atom alphabet has {len(alphabet)} symbols but the "
                f"substitute pool only has {len(pool)}. Extend "
                f"_SUBSTITUTE_POOL."
            )

        self._map = {}
        for atom_id, sub in zip(alphabet, pool):
            self._map[atom_id] = sub
        self._locked = True
        return self

    # ---- application -------------------------------------------------

    def recode(self, smiles: str) -> str:
        """Rewrite one SMILES string using the built dictionary."""
        if not self._locked:
            raise RuntimeError("SmilesRecoder.recode called before build().")
        out = []
        for t in tokenize_smiles(smiles):
            if t.kind == "atom":
                out.append(self._map.get(t.text, t.text))
            elif t.kind == "bracket":
                body = t.text[1:-1]
                m = _BRACKET_ELEMENT_RE.match(body)
                if m:
                    elem = m.group(1)
                    sub = self._map.get("[elem]" + elem, elem)
                    # Replace only the leading element, keep the rest
                    # (charge, H count, stereo) intact.
                    new_body = sub + body[len(elem):]
                    out.append("[" + new_body + "]")
                else:
                    out.append(t.text)
            else:
                out.append(t.text)
        return "".join(out)

    def describe(self) -> Dict:
        return {
            "seed": self.seed,
            "n_symbols": len(self._map),
            "mapping": dict(self._map),
        }


# ===========================================================================
#                          Prompt building
# ===========================================================================
#
# The prompt builders keep instruction length, example ordering, response
# format, and assistant prefill as similar as possible across conditions.
# The NATIVE builder is intentionally NOT here: native prompts come from
# the existing get_template_prompts() in RCHAT_PChem_Properties.py so the
# baseline stays byte-identical to the main study.  This module only
# builds the BLINDED prompt bodies, which the run hook concatenates with
# the recoded/transformed demonstrations exactly the way k_shot_test
# builds native prompts.
# ===========================================================================


_GENERIC_INSTRUCTIONS = (
    "Examples contain a string and its associated response {rid}.\n"
    "Infer the response {rid} for the final string from the examples.\n"
    "Return only one numeric value.\n"
)

_AGNOSTIC_INSTRUCTIONS = (
    "Examples contain a coded string and its associated response {rid}.\n"
    "Infer the response {rid} for the final coded string from the "
    "examples.\n"
    "Return only one numeric value.\n"
)


def _assert_no_leak(text: str) -> None:
    """Fail loudly if a blinded instruction block contains a banned word
    that would reveal the property identity."""
    low = text.lower()
    for w in _LEAK_WORDS:
        # word-ish boundary check to avoid absurd false positives
        if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", low):
            raise ValueError(
                f"Blinded prompt instruction leaked a property cue: "
                f"'{w}'. Revise _GENERIC/_AGNOSTIC instructions."
            )


def instruction_block(condition: str, rid: str = RESPONSE_ID) -> str:
    """Return the leading instruction text for a blinded condition.
    NATIVE is delegated elsewhere and raises here on purpose."""
    if condition == GENERIC:
        block = _GENERIC_INSTRUCTIONS.format(rid=rid)
    elif condition == AGNOSTIC:
        block = _AGNOSTIC_INSTRUCTIONS.format(rid=rid)
    else:
        raise ValueError(
            f"instruction_block only serves blinded conditions; got "
            f"{condition!r}. NATIVE uses get_template_prompts()."
        )
    _assert_no_leak(block)
    return block


def feature_label(condition: str) -> str:
    """The per-line feature key used in demonstrations and the query.
    GENERIC uses 'String', AGNOSTIC uses 'Code string'."""
    if condition == GENERIC:
        return "String"
    if condition == AGNOSTIC:
        return "Code string"
    raise ValueError(f"feature_label: unexpected condition {condition!r}")


def output_label(rid: str = RESPONSE_ID) -> str:
    """The per-line output key (the anonymous response identifier)."""
    return f"Response {rid}"


# ===========================================================================
#                          Per-cell spec
# ===========================================================================


@dataclass
class BlindingSpec:
    """
    Bundles everything the run hook needs to realize one blinded cell:

      condition   : native | generic | agnostic
      transform   : a LabelTransform (Identity for native)
      recoder     : an optional SmilesRecoder (only used by agnostic)
      rid         : anonymous response identifier
      value_fmt   : format string for transformed labels (e.g. '{:.3f}')

    For NATIVE, transform is Identity and recoder is None, so the hook
    reproduces the existing native behavior exactly.
    """

    condition: str = NATIVE
    transform: LabelTransform = field(default_factory=Identity)
    recoder: Optional[SmilesRecoder] = None
    rid: str = RESPONSE_ID
    value_fmt: str = "{:.3f}"

    def __post_init__(self):
        if self.condition not in VALID_CONDITIONS:
            raise ValueError(
                f"BlindingSpec.condition must be one of {VALID_CONDITIONS}; "
                f"got {self.condition!r}."
            )
        if self.condition == AGNOSTIC and self.recoder is None:
            raise ValueError(
                "AGNOSTIC condition requires a built SmilesRecoder."
            )

    # -- structure representation --------------------------------------

    def represent_structure(self, smiles: str) -> str:
        """Return the structure string to place in the prompt: native or
        recoded depending on condition."""
        if self.condition == AGNOSTIC:
            return self.recoder.recode(smiles)
        return smiles

    # -- label handling ------------------------------------------------

    def format_label(self, y_native: float) -> str:
        """Native target -> transformed, formatted string for the prompt."""
        z = self.transform.forward(float(y_native))
        return self.value_fmt.format(z)

    def parse_and_invert(self, raw_numeric: float) -> Tuple[float, bool]:
        """
        Given a numeric model output in TRANSFORMED space, return
        (native_value, in_range_flag).  For native/identity the value
        passes through and in_range is always True.
        """
        if self.condition == NATIVE:
            return float(raw_numeric), True
        in_range = True
        if hasattr(self.transform, "in_range"):
            in_range = bool(self.transform.in_range(raw_numeric))
        native = self.transform.inverse(float(raw_numeric))
        return float(native), in_range

    def describe(self) -> Dict:
        d = {
            "condition": self.condition,
            "rid": self.rid,
            "value_fmt": self.value_fmt,
            "transform": self.transform.describe(),
        }
        if self.recoder is not None:
            d["recoder"] = {"seed": self.recoder.seed,
                            "n_symbols": len(self.recoder._map)}
        return d


def make_spec(condition: str,
              transform_kind: str = "linear",
              recoder: Optional[SmilesRecoder] = None,
              out_max: float = 100.0,
              tanh_center: float = 50.0,
              tanh_amp: float = 40.0,
              std_center: float = 500.0,
              std_scale: float = 100.0,
              rid: str = RESPONSE_ID,
              value_fmt: str = "{:.3f}") -> BlindingSpec:
    """
    Convenience constructor.

    NATIVE always gets an Identity transform regardless of
    transform_kind, so a single sweep config can request e.g.
    transform_kind='zscore' and the native arm still behaves natively.
    """
    if condition == NATIVE:
        return BlindingSpec(condition=NATIVE, transform=Identity(),
                            recoder=None, rid=rid, value_fmt=value_fmt)

    tf = make_transform(transform_kind, out_max=out_max,
                        center=tanh_center, amp=tanh_amp,
                        std_center=std_center, std_scale=std_scale)
    return BlindingSpec(condition=condition, transform=tf, recoder=recoder,
                        rid=rid, value_fmt=value_fmt)


# ===========================================================================
#                          Self-test
# ===========================================================================

if __name__ == "__main__":
    # Minimal smoke test of transforms and recoder round-trips.
    print("[blinding] self-test")

    y_train = [-2.0, 0.5, 3.1, 7.4, 1.2, -0.3]

    lin = LinearInvertRescale(100.0).fit(y_train)
    for y in y_train:
        z = lin.forward(y)
        y_back = lin.inverse(z)
        assert abs(y_back - y) < 1e-9, (y, z, y_back)
    print("  linear invert/rescale round-trip OK; "
          f"describe={lin.describe()}")

    tanh = TanhStandardize().fit(y_train)
    for y in y_train:
        z = tanh.forward(y)
        y_back = tanh.inverse(z)
        assert abs(y_back - y) < 1e-6, (y, z, y_back)
    print("  tanh round-trip OK; describe includes mu/s")

    zsc = LinearStandardize(center=500.0, scale=100.0).fit(y_train)
    for y in y_train:
        z = zsc.forward(y)
        y_back = zsc.inverse(z)
        assert abs(y_back - y) < 1e-9, (y, z, y_back)
    # Whole-line inverse: even an absurd out-of-window output is finite
    assert zsc.in_range(1e6) is True and zsc.in_range(-1e6) is True
    finite_native = zsc.inverse(999999.0)
    assert finite_native == finite_native, "inverse must be finite"
    print("  zscore (LinearStandardize) round-trip OK; whole-line "
          "inverse verified; describe=" + str(zsc.describe()))

    smi = ["OC(=O)C(F)(F)C(F)(F)F",
           "FC(F)(F)C(F)(F)S(=O)(=O)[O-]",
           "OC(=O)CCCCCCC(F)(F)F"]
    rec = SmilesRecoder(seed=123).build(smi)
    # Exact round-trip check.  (Re-tokenizing the RECODED string with the
    # SMILES lexer is not a valid test: codes such as 'Xg' are not SMILES
    # atoms, so the lexer splits them into two 'other' characters.)
    # Every code is a unique two-character symbol, so decoding by
    # substitution must reproduce the original string exactly.
    inv = {code: atom_id.replace("[elem]", "")
           for atom_id, code in rec.describe()["mapping"].items()}
    assert len(inv) == len(rec.describe()["mapping"]), "codes not unique"
    for s in smi:
        r = rec.recode(s)
        decoded = re.sub(r"[XY][a-z]", lambda m: inv[m.group()], r)
        assert decoded == s, (s, r, decoded)
        print(f"  {s}  ->  {r}")
    print("  recoder round-trip (decode == original) OK")
    print(f"  recoder map: {rec.describe()['mapping']}")

    g = make_spec(GENERIC, "zscore")
    print(f"  generic instruction:\n{instruction_block(GENERIC)}")
    print(f"  spec.describe()={g.describe()}")
    print("[blinding] all self-tests passed")
