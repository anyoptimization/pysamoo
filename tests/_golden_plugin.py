"""Vendored golden (engine + pytest plugin) from pyclawd 0.1.0 — do not edit.

Self-contained, dependency-free. Register it in your top-level conftest.py with
``pytest_plugins = ["tests._golden_plugin"]``, then write ``@pytest.mark.golden`` tests that
``return`` a value. Regenerate with ``pyclawd golden vendor tests/_golden_plugin.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: JSON-serializable canonical form of a snapshot value.
Canonical = Any


def _as_numpy(value: object) -> Any:
    """Return the ``numpy`` module if *value* is a numpy array/scalar, else ``None``.

    numpy is an **optional** dependency: it is imported lazily here so the engine
    works unchanged (pure-python path) when numpy is absent. The module handle is
    returned (rather than a bool) so the caller can reuse it for ``isinstance``
    checks against ``np.floating`` / ``np.integer`` / ``np.bool_``.

    Args:
        value: The candidate snapshot value to classify.

    Returns:
        The imported ``numpy`` module when *value* is an ``np.ndarray`` or
        ``np.generic`` scalar, otherwise ``None`` (including when numpy is not
        installed).
    """
    try:
        import numpy as np
    except ImportError:
        return None
    return np if isinstance(value, (np.ndarray, np.generic)) else None


class GoldenError(AssertionError):
    """A snapshot drifted from its committed baseline beyond tolerance.

    Subclasses :class:`AssertionError` so a failed ``golden`` comparison reads as
    an ordinary test failure to pytest.
    """


def canonicalize(value: Any, precision: int) -> Canonical:
    """Reduce *value* to a stable, JSON-serializable form with floats rounded.

    Rounding to *precision* decimals is what makes the fast-path hash stable
    across runs and platforms; the un-rounded value is never what we compare
    against semantically (that is the tolerant comparison's job).

    Args:
        value: The snapshot value — a float, int, bool, str, ``None``, or a
            (possibly nested) list/tuple/dict of those. When numpy is installed,
            an ``np.ndarray`` (canonicalized via ``.tolist()``) and numpy scalars
            (``np.floating`` / ``np.integer`` / ``np.bool_``) are also accepted.
        precision: Number of decimal places to round floats to.

    Returns:
        A canonical structure safe to ``json.dumps`` deterministically.

    Raises:
        GoldenError: If *value* contains a type with no JSON-safe canonical form
            (the "no silent pickle" rule — ask for an explicit serializer).
    """
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        # Normalise -0.0 to 0.0 so the hash doesn't flip on sign of zero.
        rounded = round(value, precision)
        return rounded + 0.0
    if isinstance(value, (list, tuple)):
        return [canonicalize(v, precision) for v in value]
    if isinstance(value, dict):
        return {str(k): canonicalize(value[k], precision) for k in sorted(value, key=str)}
    np = _as_numpy(value)
    if np is not None:
        if isinstance(value, np.ndarray):
            # Recurse on the nested-list form so rounding + NaN/Inf handling apply.
            return canonicalize(value.tolist(), precision)
        # numpy scalars (np.generic). Order matters: np.bool_ is NOT an np.integer,
        # but check it first so a boolean never falls through to the integer branch.
        if isinstance(value, np.bool_):
            return canonicalize(bool(value), precision)
        if isinstance(value, np.integer):
            return canonicalize(int(value), precision)
        if isinstance(value, np.floating):
            return canonicalize(float(value), precision)
    raise GoldenError(
        f"golden: no canonical form for type {type(value).__name__!r}. "
        "Pass a float/int/str/bool/None or a nested list/dict of those, "
        "or provide an explicit serializer (the sidecar path)."
    )


def digest(canonical: Canonical) -> str:
    """Return the ``sha256:`` hash of a canonical value's deterministic JSON.

    Args:
        canonical: A structure returned by :func:`canonicalize`.

    Returns:
        A string ``"sha256:<hex>"`` — the fast-path equality key.
    """
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def values_close(a: Canonical, b: Canonical, rtol: float, atol: float) -> bool:
    """Compare two canonical values structurally with a numeric tolerance.

    Numbers compare within ``atol + rtol * |b|`` (the ``numpy.allclose`` rule);
    everything else compares for exact structural equality. This is the
    **semantic gate** — the hash is only an optimization in front of it.

    Args:
        a: The freshly computed canonical value.
        b: The committed baseline canonical value.
        rtol: Relative tolerance for numeric leaves.
        atol: Absolute tolerance for numeric leaves.

    Returns:
        ``True`` if *a* matches *b* within tolerance, else ``False``.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            values_close(x, y, rtol, atol) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_close(a[k], b[k], rtol, atol) for k in a)
    return a == b


@dataclass(frozen=True)
class Comparison:
    """Outcome of comparing a new value against a committed baseline entry.

    Args:
        ok: Whether the value matches the baseline (fast-path or tolerant).
        fast_path: ``True`` if the hash matched and no tolerant compare was needed.
        detail: Human-readable explanation (empty on a fast-path pass).
    """

    ok: bool
    fast_path: bool
    detail: str


def compare(new_value: Any, entry: dict[str, Any]) -> Comparison:
    """Compare *new_value* against a stored baseline *entry*.

    The two-speed gate:

    1. Canonicalize + hash *new_value*. If the hash equals the stored hash →
       **pass** immediately (the fast path; no tolerant compare).
    2. Otherwise fall back to a tolerant value comparison against the stored
       inline ``value``. Within tolerance → **pass** (the hash only flipped from
       sub-tolerance jitter). Outside → **fail** with a diff. This fallback is
       why the hash is an optimization, not the gate.

    Args:
        new_value: The freshly computed snapshot value.
        entry: The committed baseline entry (``value``/``hash``/``rtol``/``atol``/
            ``precision``).

    Returns:
        A :class:`Comparison` describing the outcome.
    """
    precision = int(entry.get("precision", 10))
    rtol = float(entry.get("rtol", 1e-9))
    atol = float(entry.get("atol", 1e-12))
    new_canon = canonicalize(new_value, precision)
    new_hash = digest(new_canon)

    if new_hash == entry.get("hash"):
        return Comparison(ok=True, fast_path=True, detail="")

    if "value" not in entry:
        return Comparison(
            ok=False,
            fast_path=False,
            detail=f"hash changed and no inline value stored to fall back on\n  stored: "
            f"{entry.get('hash')}\n  actual: {new_hash}",
        )

    if values_close(new_canon, entry["value"], rtol, atol):
        return Comparison(
            ok=True,
            fast_path=False,
            detail="within tolerance (hash differed by sub-tolerance jitter)",
        )

    return Comparison(
        ok=False,
        fast_path=False,
        detail=(
            f"value drifted beyond tolerance (rtol={rtol:g}, atol={atol:g})\n"
            f"  baseline: {json.dumps(entry['value'])}\n"
            f"  actual:   {json.dumps(new_canon)}"
        ),
    )


def make_entry(
    value: Any,
    *,
    precision: int = 10,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Build a committed-baseline entry from a value (the record/bless path).

    Stores the inline canonical ``value`` (readable, tolerant-comparable) plus a
    ``hash`` (the fast path). Per-snapshot ``rtol``/``atol``/``precision`` travel
    *in the entry*, not in a central config, so each snapshot owns its tolerance.
    Provenance (when/what release changed a number) is git's job — the commit that
    edits a baseline records it better than any self-reported field could.

    Args:
        value: The snapshot value to record.
        precision: Decimal places floats are rounded to before hashing.
        rtol: Relative tolerance stored for future comparisons.
        atol: Absolute tolerance stored for future comparisons.

    Returns:
        A JSON-serializable baseline entry.
    """
    canon = canonicalize(value, precision)
    entry: dict[str, Any] = {"value": canon, "hash": digest(canon)}
    if precision != 10:
        entry["precision"] = precision
    if rtol != 1e-9:
        entry["rtol"] = rtol
    if atol != 1e-12:
        entry["atol"] = atol
    return entry


class GoldenStore:
    """A per-test-module baseline file (``key → entry``) on disk.

    One JSON file per test module keeps git diffs surgical and avoids the
    merge-conflict storm a single global manifest would cause under a fleet of
    agents editing different modules.

    Args:
        path: Path to the module's baseline JSON (created on first write).
    """

    def __init__(self, path: Path) -> None:
        """Load the baseline file at *path* (empty store if it does not exist)."""
        self.path = path
        self._data: dict[str, Any] = {}
        if path.exists():
            self._data = json.loads(path.read_text())

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the baseline entry for *key*, or ``None`` if unrecorded."""
        return self._data.get(key)

    def set(self, key: str, entry: dict[str, Any]) -> None:
        """Merge-record *entry* under *key*, leaving every other key untouched."""
        self._data[key] = entry

    def keys(self) -> list[str]:
        """Return all recorded snapshot keys in this store."""
        return list(self._data)

    def remove(self, key: str) -> bool:
        """Drop *key* if present; return whether anything was removed."""
        return self._data.pop(key, None) is not None

    def is_empty(self) -> bool:
        """Whether the store holds no entries (used to prune empty files)."""
        return not self._data

    def save(self) -> None:
        """Write the store back to disk as stable, diff-friendly JSON."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self._data, indent=2, sort_keys=True) + "\n"
        self.path.write_text(text)


def module_baseline_path(baseline_dir: Path, module_stem: str) -> Path:
    """Resolve the baseline JSON path for a test module under *baseline_dir*.

    The single source of truth shared by the pytest plugin (which records/reads a
    module's baseline) and the ``pyclawd golden`` command layer (which scans them),
    so the two can never disagree on where a baseline lives.

    Args:
        baseline_dir: The configured baseline directory (``GoldenConfig.baseline_dir``).
        module_stem: The test module's file stem (e.g. ``"test_minimize"``).

    Returns:
        ``<baseline_dir>/<module_stem>.json``.
    """
    return baseline_dir / f"{module_stem}.json"


def iter_baseline_files(baseline_dir: Path) -> list[Path]:
    """List the baseline JSON files under *baseline_dir* (sorted, empty if absent).

    Args:
        baseline_dir: The configured baseline directory.

    Returns:
        Sorted ``*.json`` paths directly under *baseline_dir*.
    """
    if not baseline_dir.is_dir():
        return []
    return sorted(baseline_dir.glob("*.json"))


# === pytest plugin (spliced from pyclawd.pytest_plugin) ===



import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest


#: Default marker / baseline directory when nothing is configured.
DEFAULT_MARKER = "golden"
DEFAULT_DIR = "tests/golden"


class GoldenIdWarning(UserWarning):
    """A golden snapshot key is built from an auto-generated parametrize id.

    Index-based ids (e.g. ``algorithm0``) silently shift when parametrize cases
    are added or reordered, which would orphan committed baselines. Emitted as a
    nudge to pin explicit ``ids=`` on the parametrize.
    """


# --------------------------------------------------------------------------- #
# Snapshot-key derivation + the parametrize-id guardrail (pure, no pytest state).
# --------------------------------------------------------------------------- #


def derive_node_key(nodeid: str) -> str:
    """Derive the snapshot key from a pytest node id (its last ``::`` segment).

    Args:
        nodeid: The pytest node id (e.g. ``tests/test_x.py::test_f[zdt1-de]``).

    Returns:
        The function name including any ``[param]`` suffix — the stable per-case key.
    """
    return nodeid.split("::")[-1]


def param_id(node_key: str) -> str | None:
    """Extract the bracketed parametrization id from a node key, if any.

    Args:
        node_key: A key from :func:`derive_node_key`.

    Returns:
        The text inside ``[...]`` (e.g. ``"zdt1-de"``), or ``None`` if unparametrized.
    """
    if "[" not in node_key:
        return None
    return node_key.split("[", 1)[1].rsplit("]", 1)[0]


def _argnames_from_spec(spec: object) -> list[str]:
    """Normalise a ``parametrize`` argname spec (``"a,b"`` or ``["a","b"]``) into names."""
    if isinstance(spec, str):
        return [name.strip() for name in spec.split(",") if name.strip()]
    if isinstance(spec, (list, tuple)):
        return [str(name).strip() for name in spec if str(name).strip()]
    return []


def auto_argnames(node: pytest.Item) -> set[str]:
    """Argnames whose ids pytest may auto-number for this test node.

    Only ``parametrize`` marks without an explicit ``ids=`` can produce the fragile
    ``<argname><index>`` form; a mark with ``ids=`` owns its ids and is excluded.

    Args:
        node: The running pytest item.

    Returns:
        Argnames eligible for auto-numbered ids.
    """
    names: set[str] = set()
    for mark in node.iter_markers("parametrize"):
        if mark.kwargs.get("ids") is not None:
            continue
        if mark.args:
            names.update(_argnames_from_spec(mark.args[0]))
    return names


def looks_autogenerated(pid: str, argnames: Iterable[str]) -> bool:
    """Whether a parametrize id is an index-based ``<argname><digits>`` (pytest-auto).

    An explicit id whose text merely ends in a digit (``zdt1``, ``nsga2``, ``s2``)
    is **not** flagged, because its prefix is not a parametrize argname.

    Args:
        pid: The parametrization id (text between ``[`` and ``]``).
        argnames: Argnames eligible for auto-numbering (see :func:`auto_argnames`).

    Returns:
        ``True`` if any ``-``-separated segment is an ``<argname><digits>``.
    """
    argset = set(argnames)
    if not argset:
        return False
    for seg in pid.split("-"):
        for name in argset:
            tail = seg[len(name) :]
            if seg.startswith(name) and tail.isdigit():
                return True
    return False


# --------------------------------------------------------------------------- #
# pytest hooks — config, marker, and the return-value capture.
# --------------------------------------------------------------------------- #


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--golden-update`` flag and the ``golden_*`` ini settings."""
    parser.addoption(
        "--golden-update",
        action="store_true",
        default=False,
        help="Record/bless golden baselines from test return values instead of comparing.",
    )
    parser.addini(
        "golden_dir", "Directory holding committed golden baselines.", default=DEFAULT_DIR
    )
    parser.addini("golden_marker", "Marker selecting golden tests.", default=DEFAULT_MARKER)
    parser.addini("golden_precision", "Default decimal places for float rounding.", default="10")
    parser.addini("golden_rtol", "Default relative tolerance.", default="1e-9")
    parser.addini("golden_atol", "Default absolute tolerance.", default="1e-12")


def _marker(config: pytest.Config) -> str:
    """The configured golden marker name (default ``golden``)."""
    return str(config.getini("golden_marker") or DEFAULT_MARKER)


def _baseline_dir(config: pytest.Config) -> Path:
    """The baseline directory, resolved against the pytest rootdir when relative."""
    raw = str(config.getini("golden_dir") or DEFAULT_DIR)
    path = Path(raw)
    return path if path.is_absolute() else config.rootpath / path


def _tolerances(config: pytest.Config) -> tuple[int, float, float]:
    """The configured ``(precision, rtol, atol)`` defaults for new baselines."""
    return (
        int(config.getini("golden_precision") or 10),
        float(config.getini("golden_rtol") or 1e-9),
        float(config.getini("golden_atol") or 1e-12),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the golden marker so ``-m <marker>`` selects golden tests."""
    config.addinivalue_line(
        "markers",
        f"{_marker(config)}: capture the test's return value as a committed golden baseline.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Run a golden-marked test ourselves and snapshot its return value.

    For a test carrying the golden marker, this calls the function, captures its
    return value, and records it (``--golden-update``) or compares it against the
    committed baseline (the default), then returns ``True`` so pytest does not call
    the function a second time. Non-golden tests are left untouched.

    Args:
        pyfuncitem: The pytest function item about to be called.

    Returns:
        ``True`` when the golden test was handled here, else ``None`` (defer to
        pytest's normal call).

    Raises:
        GoldenError: When the return value drifts from its baseline, or no baseline
            exists yet in compare mode.
    """
    config = pyfuncitem.config
    if pyfuncitem.get_closest_marker(_marker(config)) is None:
        return None

    testargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    result = pyfuncitem.obj(**testargs)
    _snapshot(pyfuncitem, result)
    return True


def _snapshot(pyfuncitem: pytest.Function, value: Any) -> None:
    """Record or compare *value* (a test's return) against its committed baseline."""
    config = pyfuncitem.config
    module_file = pyfuncitem.module.__file__
    assert module_file is not None
    store = GoldenStore(module_baseline_path(_baseline_dir(config), Path(module_file).stem))

    key = derive_node_key(pyfuncitem.nodeid)
    pid = param_id(key)
    if pid is not None and looks_autogenerated(pid, auto_argnames(pyfuncitem)):
        warnings.warn(
            f"golden: parametrize id {pid!r} in {key!r} looks auto-generated (index-based) — "
            "snapshot keys shift if cases are reordered. Pin explicit ids= on the parametrize.",
            GoldenIdWarning,
            stacklevel=3,
        )

    precision, rtol, atol = _tolerances(config)
    if config.getoption("--golden-update"):
        store.set(key, make_entry(value, precision=precision, rtol=rtol, atol=atol))
        store.save()
        return

    entry = store.get(key)
    if entry is None:
        raise GoldenError(
            f"golden: no baseline for {key!r}. Record it with `pytest --golden-update` "
            "and commit the baseline."
        )
    result = compare(value, entry)
    if not result.ok:
        raise GoldenError(f"golden: {key}\n  {result.detail}")
