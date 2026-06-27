# pysamoo Performance & Speed-up Guide

This document explains **why surrogate-assisted runs can be slow**, gives **measured
numbers** for every algorithm, and lists **concrete levers to speed things up**. It
also records the dependency-version findings that affect runnability.

> TL;DR
> - One iteration of every algorithm is **sub-second to ~1s**. Slowness comes from
>   large *demo budgets* multiplied by Gaussian-Process (Kriging) fitting that scales
>   **O(n³)** in the archive size.
> - The biggest single offender is **Bayesian Optimization with `model_selection=True`**
>   (≈121 model fits per generation). Turning it off is a ~6–7× speedup.
> - Pick the surrogate to match the budget: **Kriging** is most accurate but expensive;
>   **RBF** is far cheaper and often good enough.

---

## 1. Where the time goes

Surrogate-assisted algorithms repeat this loop:

1. **Fit** a surrogate to all evaluated points (the *archive*).
2. **Optimize** the cheap surrogate (an inner evolutionary loop) to propose infills.
3. **Evaluate** the (few) infills on the true, expensive problem; append to the archive.

Two costs dominate, both growing as the run proceeds:

- **Surrogate fit — O(n³).** Kriging/GP fitting solves a dense `n × n` system (Cholesky /
  least-squares) where `n` is the archive size. Doubling the archive ~8× the fit cost.
  Over hundreds of infills the archive grows large and late iterations dominate the run.
- **Model selection — multiplicative.** When enabled, each fit step cross-validates a whole
  *grid* of candidate models and refits the winner, multiplying the per-iteration fit count
  (see §3).

The acquisition / inner-optimization loop is usually secondary, but it is not free: it runs
a full evolutionary algorithm against the surrogate every iteration.

---

## 2. Measured per-algorithm floor (one iteration)

Initial DOE + ~1 infill, small problems, no plotting (`conda env default`, numpy 2.4.6,
pymoo 0.6.1.6):

| Algorithm | DOE + 1 infill |
|---|---:|
| PSAF | 0.27 s |
| GPSAF (single-obj) | 0.58 s |
| SSANSGA2 (bi-obj) | 1.16 s |
| Bayesian Optimization (`n_gen=1`) | ~0 s (DOE only) |

**Takeaway:** the algorithms themselves are cheap per step. The full demo scripts are slow
only because of their presentation budgets (hundreds of evaluations) and the O(n³) growth.

### Full usage-example sweep (demo budgets, 180 s cap per script)

Measured before the dependency fix in §5; "ran" = optimization completed.

| Example | Wall time | Outcome |
|---|---:|---|
| `usage_gpsaf_constr.py` | 1.7 s | crashed early on `np.math` (now fixed, §5) |
| `usage_constr_sampling.py` | 5.3 s | ok |
| `usage_lqcmaes.py` | 6.6 s | ok |
| `usage_psaf.py` | 92 s | ok (full budget) |
| `usage_gpsaf_many.py` | 101 s | optimization ran; crashed at *plot* (`cm.get_cmap`, now fixed) |
| `usage_gpsaf_multi.py` | 129 s | optimization ran; crashed at *plot* (now fixed) |
| `usage_ssansga2.py` | 133 s | optimization ran; crashed at *plot* (now fixed) |
| `usage_gpsaf_single.py` | >180 s | timeout (budget) |
| `usage_gpsaf_cmoo.py` | >180 s | timeout (budget) |
| `usage_bayesian_optimization.py` | >180 s | timeout (see §3) |
| `usage_smac.py` | — | skipped (optional `smac` dep) |

None of these are algorithm bugs: the failures were dependency drift (§5), and the timeouts
were budget size. This is why the test suite exercises **one iteration per algorithm**
instead of running the demo scripts verbatim (see `tests/test_usage.py`).

---

## 3. Deep dive: Bayesian Optimization

`usage_bayesian_optimization.py` runs `("n_gen", 150)` and never finishes within 180 s.

**Root cause:** `BayesianOptimization(model_selection=True, ...)`. Every generation
(`src/pysamoo/experimental/bo.py`, `_infill`) expands the full Kriging hyperparameter grid
(regr × corr × thetaU × ARD = **24 configurations**), runs **5-fold cross-validation** over
all of them, then refits the winner — **24×5 + 1 = 121 Kriging fits per generation**. Profiling
shows `ModelSelection.do` accounts for **~87 %** of runtime.

Each model-selection step also grows super-linearly with archive size:

| archive size | one `ModelSelection.do` |
|---:|---:|
| 30 | 2.8 s |
| 90 | 10.7 s |
| 170 | >115 s |

Per-generation cost, `n_var=10`:

| archive | `model_selection=True` | `model_selection=False` |
|---:|---:|---:|
| 22 | 2.57 s | 0.31 s |
| 32 | 3.07 s | 0.52 s |

So the 180 s budget is exhausted around generation ~35–40; the loop never approaches 150.

**Fixes, ranked:**

1. **`model_selection=False`** (`usage_bayesian_optimization.py`) — ~6–7× faster per
   generation; falls back to a single `Kriging(regr="linear", corr="gauss", ARD=True)`.
2. **Reduce `n_gen`** 150 → ~50 — 150 sequential GP infills is unusually large for BO
   (typical total budgets are 50–100); also keeps the archive (and O(n³) cost) small.
3. **Re-select only every k generations** (`bo.py:_infill`) — refit the chosen config in
   between; removes ~120 of the 121 fits on most generations.
4. **Cap the GP archive** (`bo.py`, `self._archive.get(...)`) — fit on the best/nearest N
   points (e.g. N=80) to bound per-fit cost regardless of run length.
5. **Shrink the hyperparameter grid** (`Kriging.hyperparameters()`) — fewer configs → fewer
   CV fits.
6. **Cheaper acquisition search** (`robust_fmin_acquisition`, NicheGA pop/LHS sizes) — the
   secondary ~13 %.

---

## 4. Speed-up levers (all algorithms)

| Lever | How | Effect | Trade-off |
|---|---|---|---|
| Smaller evaluation budget | `("n_evals", N)` / `("n_gen", N)` | Linear; also caps archive growth | Fewer evals → worse final solution |
| Cheaper surrogate | `surrogate=make_surrogate(problem, obj_models=only_rbf)` | RBF ≈ 6× faster than Kriging here | Lower model accuracy |
| Disable model selection | `model_selection=False` (BO) / smaller model dict | Removes the per-iter CV multiplier | No per-iteration model adaptation |
| Cap the archive used for fitting | subset best/nearest-N before `surrogate.fit` | Bounds the O(n³) term | GP ignores far/old points |
| Fewer infills per round | `n_max_infills` / `n_infills` smaller | Fewer surrogate-optimize loops | Slower convergence per eval |
| Smaller inner search | surrogate-side `pop_size` / `n_gen` | Cheaper step 2 | Weaker infill proposals |
| Skip plotting in batch runs | `matplotlib.use("Agg")`; don't call `.show()` | Removes GUI/render cost | None for headless runs |

Rule of thumb: **match the surrogate to the budget.** Tiny budgets → RBF or a single Kriging
config. Generous budgets where each true evaluation is very expensive → Kriging with model
selection can pay off.

---

## 5. Dependency version notes (runnability)

Environment: **numpy 2.4.6**, **matplotlib 3.11.0**.

`setup.py` originally pinned `pymoo==0.6.1.1`, which predates two upstream removals and
crashes against this environment:

- `numpy>=1.25` removed `numpy.math`, still used by pymoo's ES code
  (`pymoo/algorithms/soo/nonconvex/es.py`) → broke the ISRES-based `usage_gpsaf_constr.py`.
- `matplotlib>=3.9` removed `matplotlib.cm.get_cmap`, still used by pymoo's plotting →
  broke the plot step of several examples (after the optimization had already run).

Both are **pymoo-vs-dependency drift, not pysamoo bugs.** The pin is now
**`pymoo>=0.6.1.5,<0.6.2`** (resolves to 0.6.1.6), which replaced `np.math`→`math` and
`cm.get_cmap`→`pyplot.get_cmap` natively. A compatibility audit found every pymoo API
pysamoo imports is present and signature-compatible in 0.6.1.6, so the bump is **low risk**.
The only code touching changed/removed pymoo APIs lives in `src/pysamoo/experimental/`
(`SACOBRA.py`), which was already broken under 0.6.1.1 and is out of scope.

If you ever need to run under an older pymoo, restore `np.math`/`cm.get_cmap` shims as in an
earlier version of `tests/conftest.py`.

---

## 6. How this maps to the test suite

`tests/test_usage.py` runs **one minimal iteration per algorithm** (and the sampling
routine) with no plotting — the whole suite is a few seconds and validates that every
algorithm wires up and produces finite results. The large, slow demo scripts remain under
`src/pysamoo/usage/` as illustrative examples, not as the test contract.

See also [BENCHMARKING.md](BENCHMARKING.md) for measuring solution *quality* (not just
runtime) and for developing better algorithms/models.
