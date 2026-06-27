# Research dossier: faster, reproducible surrogate model selection in pysamoo

*Companion to [model-selection-loop.md](model-selection-loop.md). Experiment harness:
[model_selection_bench.py](model_selection_bench.py).*

## 1. Why this matters

pysamoo is a **generic** surrogate-assisted optimizer: a pymoo algorithm wrapped so
that, each infill iteration, it fits surrogate models to the evaluated archive and
optimizes *those* instead of the expensive true problem. The framework "only needs
models" — so the quality of the **model-selection** step (which model to trust) is
the heart of the method.

Two problems make that step the dominant pain point:

- **Cost.** Profiling earlier put model fitting at ~**87 %** of runtime; a single
  selection step grows from ~3 s to **>100 s** as the archive grows.
- **Reproducibility.** Repeated runs with a *fixed* pymoo seed produce different
  results — which blocks golden tests and makes science hard.

> **The framing that drives this study (the key insight):** model selection is not
> about fitting the current points well — it is about **generalization**: choosing a
> model that predicts well on *unseen* points. Any speed-up must therefore be judged
> on **out-of-sample accuracy**, never on training fit or runtime alone. Cutting cost
> while preserving generalization is the whole game.

## 2. How model selection actually works today (code facts)

Sources: `src/pysamoo/core/{target,defaults,surrogate,algorithm}.py`,
`ezmodel/core/{partitioning,benchmark}.py`, `ezmodel/util/partitioning/crossvalidation.py`.

- **The pool is large.** `DEFAULT_OBJ_MODELS` builds **38 candidates per objective**
  (32 RBF = kernel{linear,cubic,gaussian,mq} × tail{const,lin,quad,lin+quad} ×
  normalized{T,F}, plus 6 Kriging = {const,lin,quad}×{plain,ARD}). Constraints add
  36 (ineq) / 21 (eq) more — *per constraint*.
- **Selection = k-fold CV over the whole pool.** `Target.validate` runs a
  `Benchmark` over all models with **5-fold** CV, scores each fold, and
  `find_best` ranks lexicographically by `[kendall_tau, mae]` (mean over a rolling
  window of the last 5 validations). ⇒ **5 × 38 = 190 model fits per objective per
  selection call** (serial; the single-train/test path costs 38).
- **It runs every iteration.** `find_best=True` is the default on every `validate`.
  GPSAF/PSAF/SSANSGA2 call `validate` at init (CV) and again every `_advance`
  (single split). **`nth_validate` (the intended "re-select only every N iters"
  throttle) is defined but never read — dead code.** This is the biggest structural
  inefficiency: the pool is re-benchmarked and the winner re-chosen constantly.
- **Cost drivers (ranked):** (1) DACE **Kriging** boxmin hyperparameter optimization
  (ARD optimizes one θ per dimension); (2) **GP O(n³)** in archive size n;
  (3) **pool × folds** = 190 fits; (4) RBF SVD solves (cheap by comparison).

## 3. Why runs were non-reproducible (root cause — now FIXED)

> **Status (2026-06-27): resolved.** `self.random_state` is now threaded through
> every stochastic site (DOE sampling, GPSAF/PSAF/SSANSGA2 infill, `knockout.noisy`,
> pymoo `compare`/`RouletteWheelSelection`), CV folds are deterministic
> (`randomize=False`), and the tie-break is `models[0]`. All three algorithms are
> bit-reproducible under a fixed seed; guarded by `tests/test_reproducibility.py`.
> Best practice applied: thread the `Generator`, never seed globals (NumPy's
> guidance). The historical analysis below is retained for context.


pymoo 0.6.1 changed seeding: `Algorithm.setup` now creates a **local**
`np.random.default_rng(seed)` and **no longer seeds the global `np.random`/`random`**.
pysamoo's selection/infill code still uses the **global** RNGs, which are therefore
left at OS-entropy state and drift between runs. The two selection-specific sites:

1. **CV fold shuffle is unseeded** — `target.py:73` constructs
   `CrossvalidationPartitioning(self.n_folds)` with **no seed**, and
   `crossvalidation.py` does `random.shuffle(indices)` on the global `random`.
   Different folds ⇒ different CV scores ⇒ different winner.
2. **Random tie-break** — `target.py:119` ends `find_best` with
   `np.random.choice(models)`. Because `kendall_tau` returns an **integer** disorder
   count, ties are *frequent*, so this fires often and flips the choice.

Float noise from threaded BLAS (`np.linalg.svd`, DACE) can also flip near-ties and
route into the unseeded tie-break. **Verified:** with unseeded folds the selected
model flips across perturbed ambient states; **seeding the folds makes it
deterministic** (`model_selection_bench.py`, determinism section).

## 4. Measured cost vs generalization (the central trade-off)

From `model_selection_bench.py` (5-var problems, held-out test set of 400 points):

| problem | pool | sel time | **test RMSE** | test τ | picked |
|---|---|---:|---:|---:|---|
| ackley | full (38) | 0.78 s | **0.80** | 0.51 | RBF mq |
| ackley | small (3) | 0.02 s | **33.5** ❌ | -0.04 | RBF cubic |
| ackley | **family (8)** | **0.07 s** | **0.80** ✅ | 0.51 | RBF mq |
| ackley(n=80) | full (38) | 1.41 s | 0.70 | 0.58 | RBF mq |
| ackley(n=80) | **family (8)** | **0.12 s** | **0.69** ✅ | 0.58 | RBF mq |
| rastrigin | full (38) | 0.73 s | 18.8 | 0.42 | RBF mq |
| rastrigin | small (3) | 0.02 s | 148.8 ❌ | 0.15 | RBF cubic |
| rastrigin | **family (8)** | **0.07 s** | **18.5** ✅ | 0.43 | RBF lin |

**Findings**
- **Naively shrinking the pool destroys generalization** (small(3): RMSE 4–170×
  worse). The cheap pool simply lacked the kernel family that fits the function.
- **But a redundancy-free pool keeps it.** `family(8)` (one model per kernel family,
  ±normalization) matches full(38) generalization at **~10× lower cost**. The 38-pool
  spends most of its budget on near-duplicate RBF tail/normalization variants that
  rarely win.
- ⇒ The win is **cover the model families, drop the redundancy** — not "use fewer
  models" blindly.

## 5. What the literature says (cited)

Full survey with URLs in the agent report; the load-bearing techniques:

1. **Closed-form LOO-CV for GP/Kriging** (Rasmussen & Williams, *GPML* §5.4.2,
   eq. 5.12): leave-one-out mean/variance from the *one* factorization the GP already
   computes — `μ_i = y_i − [K⁻¹y]_i / [K⁻¹]_ii`, `σ²_i = 1/[K⁻¹]_ii` — total overhead
   O(n²), **no k-fold refitting**. Rank by **log pseudo-likelihood** (eq. 5.11), not
   squared error. RBF/kernel-ridge analogue: PRESS via the hat matrix,
   `resid_i/(1−H_ii)`, and GCV (Golub–Heath–Wahba 1979). *Biggest single win; also
   deterministic.* https://gaussianprocess.org/gpml/chapters/RW5.pdf
2. **Lazy / periodic re-selection** — decouple selection frequency from infill
   frequency; re-select every k iters or on a trust-region degradation trigger.
   Evidence the quality cost is small: Ahrari & Verstraete, *SWEVO* 2023;
   Hanawa et al. 2025. *This is exactly what the dead `nth_validate` was meant to do.*
3. **Cap the GP training set** to an L-nearest / most-recent subset ⇒ O(L³),
   constant in archive size, often *more* accurate where search is active
   (GPEME, Liu et al. 2014; TuRBO, Eriksson et al. 2019).
4. **Stop electing a single winner** — PRESS-weighted ensemble reuses the LOO
   residuals, removes winner-flip variance, gives free uncertainty (Goel et al. 2007;
   HeE-MOEA, Guo et al. 2019).
5. **Racing / successive-halving** over the pool (resource = folds/subsample) for any
   candidate lacking a closed-form LOO (Hoeffding races, Maron & Moore 1993;
   Successive Halving, Jamieson & Talwalkar 2016).
6. **Determinism**: seed folds once per iteration and reuse for all candidates;
   fixed pool order + deterministic argmin tie-break; seed GP restarts
   (scikit-learn common-pitfalls guidance). *LOO (item 1) sidesteps fold randomness
   entirely.*

## 6. Synthesis → what to try (feeds the loop)

The independent cost angles compose: **redundancy-free pool** (×10, measured) ×
**closed-form LOO instead of k-fold** (×k, removes refits) × **cap the fit set**
(bounds O(n³)) × **lazy re-selection** (×k in frequency). Determinism comes for free
from LOO, or from seeding folds + a deterministic tie-break. Each is a hypothesis in
the loop, ranked by impact-per-effort there.
