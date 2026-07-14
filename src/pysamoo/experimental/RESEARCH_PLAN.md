# BO Method Search — Research Plan

Goal: a **single, fast** `BayesianOptimization` default that performs well across the whole
difficulty spectrum, measured by the ECDF benchmark (`experimental/benchmark.py`). "Fast" is a
first-class constraint — the benchmark is time-boxed (~20s) and wall-clock is part of the score.

## 1. The oracle

`experimental/benchmark.py` (do **not** edit when iterating) runs the suite below at 2D and 10D,
optimum `F*=0`, budget `30 + 5d` evals, **3 seeds**, and reports **ECDF** = fraction of
`(function × target × seed)` solved, where targets are **scale-normalized** per function
(`R_f·10^(-k)`, `k∈[0,8]`) so every function gives a smooth gradient. (2 seeds was too noisy to
reward real gains — hardened to 3.)

**Original baseline (AutoModel select): tuned 0.468 / held-out 0.438, ~31s.**
**Shipped default (single Kriging[exp], upstream pg4/pool256): tuned 0.473 / held-out 0.458, ~14s**
— a modest but *generalizing* win (2x faster, stable). Bigger tuned-only gains (pg6, pool2048) were
**overfitting** — they raised tuned ECDF but lowered held-out. Always check both (`evaluate_heldout`).

## 2. Problem taxonomy — what each function tests

| function | modality | conditioning | separable | tests | final 10D |
|---|---|---|---|---|---|
| Sphere | uni | well | yes | baseline (local refinement) | solved ✓ |
| Rosenbrock | uni | ill | **no** | **local** model / curved valley | 0.16 (open) |
| Rastrigin | **multi** | well | yes | **global** exploration (funnel) | 0.10 (open) |
| RotatedEllipsoid | uni | **ill (1e6)** | **no** | **metric / rotation** learning | 0.49 (was 0.22) |

The three 10D failures each demand a *different* fix — local model (Rosenbrock), global
exploration (Rastrigin), learned metric (Ellipsoid). A good global+local method lifts all three.

## 3. Method ideas (from the literature), ranked by expected payoff

Global+local is the dominant paradigm (Ong et al. 2003). Combination patterns: **switch**
(current Hybrid), **competition** (LAGO — both propose each step, best wins), **localize the global
model** (TuRBO — local GP in adaptive trust regions), **memetic filter-then-refine**.

Ideas to try, each a self-contained change in `experimental/`:

1. **Competition Hybrid (LAGO-style)** — global EI *and* local each propose every iteration; take
   whichever actually improves. Removes the brittle patience switch. *Helps: all.*
2. **PCA-rotated local (LABCAT-style)** — weighted-PCA rotate the local points, fit ARD-GP in the
   rotated frame (= a Mahalanobis kernel for free), EI in a rotated trust region. *Helps:
   RotatedEllipsoid, Rosenbrock.*
3. **Better local model** — replace the single quadratic with a *moving* trust region that refits
   each step (NEWUOA-style), or an RBF/local-GP local model that follows a curved valley. *Helps:
   Rosenbrock.*
4. **Multiple trust regions / restarts (TuRBO-style)** — several local regions to escape Rastrigin
   basins; restart the global phase from diverse seeds. *Helps: Rastrigin.*
5. **Acquisition-space whitening** — search EI in coordinates whitened by the local Hessian / sample
   covariance, so EI explores along the valley. *Helps: Rosenbrock, Ellipsoid.*
6. **CMA-style covariance adaptation** for the local sampling distribution. *Helps: Ellipsoid,
   Rosenbrock.*

## 4. Iteration protocol (the loop)

Each iteration: pick one idea → implement in `experimental/` (bo/infill/optimizer/acquisition only;
**never** benchmark.py/problems.py) → `pyclawd check` must pass → run benchmark → keep only if
`score` (ECDF, speed tie-break) beats the best AND time stays ~≤25s → append to `findings.md`.
Time-boxed; keep the best version. Agents compare, humans bless commits.

## 5. Success criteria

- Strictly beat baseline ECDF 0.359 while staying fast (~≤25s benchmark).
- Stretch: solve ≥1 of the three 10D-failing functions to a non-trivial target without regressing
  Sphere or the 2D cases.

## References

- Ong, Nair, Keane (2003) — Combining global and local surrogate models.
- Eriksson et al. (2019) — TuRBO: Scalable Global Optimization via Local BO.
- LABCAT (2023, arXiv:2311.11328) — PCA-aligned trust regions.
- CMA-BO (2024, arXiv:2402.03104) — covariance matrix adaptation for BO.
- LAGO (2026, arXiv:2603.02970) — local-global competition framework.
- ALEBO (NeurIPS 2020) — Mahalanobis kernel for linear embeddings.
- Ament et al. (2023) — LogEI (numerically stable acquisition).

## 6. Open ideas (untried / in progress)

- **Sequential CMA-ES local strategy** (`LocalCMAES`): learns the metric (covariance ~ inverse
  Hessian) — the proven method for curved-valley (Rosenbrock) and ill-conditioned (Ellipsoid)
  problems. One sample per infill, rank-mu update each generation. *In progress.*
- **Periodic AutoModel re-selection** (user idea, 2026-06-30): we dropped per-infill `AutoModel`
  for fixed `Kriging[exp]` (speed). But re-running *full* selection occasionally (every N evals,
  not every infill) could recover AutoModel's adaptivity cheaply — and crucially lets the model
  *type* switch as more data reveals structure (a kernel that wins at 20 pts may lose at 80). Test
  `AutoModel` refit-from-scratch every ~15 evals vs fixed Kriging[exp]. Open question: worth the cost?
