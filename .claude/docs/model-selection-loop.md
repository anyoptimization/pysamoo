# Research loop: cheap, reproducible surrogate model selection

An iterative, self-contained loop to investigate and improve pysamoo's
model-selection step. Run it yourself or drive it with `/loop`. Background and
measured evidence: [model-selection-research.md](model-selection-research.md).
Harness: [model_selection_bench.py](model_selection_bench.py).

---

## Research question

> **How can we cut the per-iteration cost of surrogate model selection by ≥10× while
> preserving (or improving) out-of-sample generalization, and make selection
> reproducible under a fixed seed?**

Sub-questions (each is a hypothesis in the backlog):

- **RQ1 (pool).** How small can the candidate pool be before generalization drops?
  Is "one model per kernel family" enough?
- **RQ2 (CV cost).** Can closed-form LOO-CV replace k-fold refitting without changing
  which model is selected?
- **RQ3 (frequency).** How often must we *re-select*? Does lazy re-selection (honor
  `nth_validate` / a degradation trigger) keep convergence quality?
- **RQ4 (scale).** Does capping the GP fit to an L-nearest subset bound cost without
  hurting generalization where search is active?
- **RQ5 (ensemble).** Does a PRESS-weighted ensemble beat single-best generalization
  at zero extra fit cost?
- **RQ6 (determinism).** What minimal changes make selection bit-reproducible under a
  fixed seed, with no quality loss?

---

## Invariants (do not break these)

1. **Generalization first.** Every candidate change is judged on **held-out test
   error** (RMSE + rank correlation τ), never on training fit or runtime alone. A
   speed-up that worsens test error beyond the tolerance below is rejected.
2. **Fixed evaluation suite.** Score on the same problems/sizes every iteration:
   `ackley`, `rastrigin` (multimodal) and at least one `zdt`/`dtlz` for multi-objective;
   archive sizes n ∈ {40, 80, 160}; seeds {0,1,2,3,4}. Extend the suite only by
   *adding*, never removing.
   - **Caveat (validate on REAL archives).** The harness trains on uniform random
     samples generated all at once; a real optimization archive is sequential,
     clustered near promising regions, and non-stationary. Cost/determinism results
     are distribution-independent and hold, but **generalization/quality claims
     (family≈full, racing≈full) must be confirmed on archives produced by actual
     optimization runs.** Preferred realistic metric: select/fit on iteration *t*'s
     archive, test on iteration *t+1*'s **actual infills** (the real prediction
     task). The decisive test is end-to-end: plug a strategy into the real algorithm
     and compare final IGD/best-F + wall-time. Beware CV leakage from spatial
     clustering (nearby points across folds → optimistic CV).
3. **Two baselines stay in every comparison:** `full(38)` (current behaviour,
   generalization reference) and the plain pymoo algorithm with no surrogate
   (does the surrogate still help?).
4. **Acceptance tolerance.** A variant is "no worse" if its mean test RMSE is within
   **+5 %** of `full(38)` on every suite problem. Cost must improve by the target of
   the hypothesis.
5. **Determinism is a gate, not a nicety.** A variant that is faster but
   non-reproducible is not "done" until RQ6 is also satisfied for it.

---

## Iteration protocol

Each loop iteration:

1. **Pick** the top `pending` hypothesis from the backlog (highest impact/effort).
2. **Implement** it behind a flag or as a new strategy in `model_selection_bench.py`
   (research first; only touch `src/pysamoo/**` once a hypothesis is confirmed and you
   intend to ship it — then add a golden/regression test).
3. **Measure** cost, generalization (RMSE + τ), and determinism via the harness on the
   fixed suite.
4. **Decide** against the invariants: `confirmed` (meets cost goal within tolerance),
   `rejected` (hurts generalization), or `partial` (works in some regime — record
   where).
5. **Record** one row in the Results log below (append-only) and update the
   hypothesis status. Note any surprise as a new hypothesis.
6. **Stop** when the Stopping criteria are met; otherwise go to 1.

---

## Hypothesis backlog (prioritized)

| # | Hypothesis | Where to change | Experiment | Success criterion | Status |
|---|---|---|---|---|---|
| H1 | A redundancy-free **family pool (~8)** generalizes like full(38) at ~10× less cost | `defaults.py` pool / bench `POOLS` | family(8) vs full(38) vs small(3) on suite | ≥8× faster, RMSE within +5% | **confirmed** (ackley/rastrigin, see log) — extend to MOO + constraints |
| H2 | **Closed-form LOO-CV** (GPML eq. 5.12) selects the same model as 5-fold at a fraction of cost | new strategy; later `target.py`/`ezmodel` | LOO-rank vs kfold-rank agreement; cost | same top-1 ≥90% of cases, ≥3× faster, deterministic | **REJECTED** — `loo_vs_kfold.py`: LOO picks a worse-generalizing model than 5-fold (Ackley test-gap 0.025 vs 0.0001) and disagrees 20–40% of seeds. 5-fold is more robust; keep it. Speed must come from H3/H8/H4, not from changing the CV scheme. |
| H3 | **Lazy re-selection** every k iters (wire up dead `nth_validate`) keeps convergence | `algorithm.py` `revalidate()` gates the `_advance` validate | full-run wall + quality vs k∈{1,5} | big wall drop, quality not worse | **SHIPPED** — ~3.8× faster at equal/better quality; `nth_validate=5` is now honored (was dead code); guarded by `tests/test_selection.py` |
| H4 | **Cap GP fit to L-nearest subset** bounds O(n³) without hurting generalization | `_doe()` / fit path | RMSE & cost vs cap L∈{80,160,∞}, large n | cost ~constant in n, RMSE within +5% | pending |
| H5 | **Reproducible runs under a fixed seed** | thread `random_state` everywhere + deterministic CV folds + deterministic tie-break | run each algorithm twice, assert identical | identical results across runs | **SHIPPED** — see below; `tests/test_reproducibility.py` guards GPSAF/PSAF/SSANSGA2 |
| H6 | **PRESS-weighted ensemble** ≥ single-best generalization, free uncertainty | `target.py` find_best/predict | ensemble vs best RMSE on suite | RMSE ≤ best, no extra fits | pending |
| H7 | **Rank by log pseudo-likelihood** (not integer kendall_tau) removes frequent ties | `target.py` indicators | tie frequency; selection stability | fewer ties, stable choice | pending |
| H8 | **Adaptive racing pool**: shrink the active set over time (prune dominated models by rolling CV error) with a per-family diversity floor + periodic re-admission | new `RacingTarget` / `target.py` | cost (fits/wall) + generalization vs full, over a simulated run | ≥40% fewer fits, RMSE within tol, still adapts | **confirmed (prototype)** — see `adaptive_pool_prototype.py`; matches full RMSE at 56% of fits |

---

## Stopping criteria

Stop when **either**:
- a combination (expected: H1 + H2 + H3, optionally H4) achieves **≥10× median cost
  reduction** on the suite at large n **with** mean test RMSE within +5 % of full(38)
  **and** reproducible selection; or
- the backlog is exhausted (all `confirmed`/`rejected`) — then write up the
  recommended default configuration and open a PR that ships it with golden tests.

---

## Results log (append-only)

| date | iter | hypothesis | setup | cost | generalization (test RMSE) | determinism | verdict |
|---|---|---|---|---|---|---|---|
| 2026-06-27 | 0 | baseline | full(38), 5-fold, ackley/rastrigin n=40/80 | 0.7–1.4 s | ackley 0.70–0.80; rastrigin 15.7–18.8 | **no** (unseeded folds) | reference |
| 2026-06-27 | 0 | H1 first look | small(3) | 0.02 s | ackley 4.5–33; rastrigin 149–171 | n/a | **rejected** (kills generalization) |
| 2026-06-27 | 1 | H1 | family(8) | 0.07–0.12 s | ackley 0.69–0.80; rastrigin 15.7–18.5 (≈full) | n/a | **confirmed** (~10× faster, RMSE within tol) — TODO: MOO + constraints |
| 2026-06-27 | 1 | H5 | full(38), seeded folds | same | unchanged | **yes** (identical across runs) | **confirmed** |
| 2026-06-27 | 2 | H5 | **SHIPPED**: random_state threaded through GPSAF/PSAF/SSANSGA2 + deterministic CV folds (`randomize=False`) + deterministic tie-break (`models[0]`) | unchanged | unchanged | **yes** — all 3 algos bit-identical across runs | **done** — guarded by `tests/test_reproducibility.py` |
| 2026-06-27 | 3 | H8 | racing pool (warmup 3, window 4, keep 0.6, floor 8, re-admit 4 every 5) vs full, simulated 20-iter growing archive (ackley, rastrigin) | **56% of full's fits** (and the *late, expensive* O(n³) fits run on the shrunk pool) | **identical to full** (ackley 0.666, rastrigin 16.225) | reproducible (Generator) | **confirmed** — beats fixed family(8) (which was 0.688) |
| 2026-06-27 | 4 | H8 | **SHIPPED**: `RacingTarget` + pluggable `selection="racing"` (modular registry) | real GPSAF run: pool shrinks 38→24→18→12→9; ~20-25% faster | equal quality | reproducible | **done** — `tests/test_selection.py` |
| 2026-06-27 | 5 | H3 | **SHIPPED**: `revalidate()` honors `nth_validate` (lazy re-selection). GPSAF Ackley(10), 220 evals, n_max_doe=200 | nth=1: 42s → **nth=5: 11s (~3.8×)**; racing+nth=5: **10.5s (~4×)** | best_F equal/better (3.78 → 2.59) | reproducible | **done** — biggest single wall-time win; stacks with racing |
| 2026-06-27 | 6 | H2 | LOO-CV vs 5-fold as model selectors (ackley/rastrigin/sphere, 9-model pool, 5 seeds) | n/a (selection-quality study) | **5-fold picks better-generalizing model** (Ackley gap 0.0001 vs LOO 0.0246); disagree 20–40% | n/a | **REJECTED** — confirms field experience that LOO is less robust; keep 5-fold |

---

## How to run

```bash
# the harness (cost vs generalization + determinism check)
pyclawd python .claude/docs/model_selection_bench.py

# add a new strategy: edit model_selection_bench.py (strat_* / POOLS), re-run,
# then append a row to the Results log above.
```

When a hypothesis is confirmed *and* you decide to ship it, make the change in
`src/pysamoo/**`, add a regression/golden test, and run `pyclawd check`.

---

## Notes for the next session

- H1 confirmed on single-objective; **next**: rerun H1 on a multi-objective problem
  (e.g. `zdt1`) and on constraint pools (`DEFAULT_IEQ_CONSTR_MODELS` has 36) — those
  pools are even larger and likely have the same redundancy.
- H5 **DONE**. The whole run is now reproducible: `self.random_state` is threaded
  through every stochastic site in `gpsaf.py` (tournament/alpha/beta/restart),
  `psaf.py`, `ssansga2.py` (roulette selection), `knockout.noisy`, the DOE sampling
  (`algorithm.py` `_initialize_infill`), pymoo's `compare`/`RouletteWheelSelection`
  (which accept `random_state`); CV folds are deterministic (`randomize=False`) and
  the tie-break is `models[0]`. **Best practice followed: thread the Generator, never
  seed globals.** Guarded by `tests/test_reproducibility.py`. Note: this enables a
  same-machine full-run golden if ever wanted, but a *fixed-value* golden may differ
  across BLAS/platforms because GP model selection can flip on float noise — the
  equality-across-runs test is the robust guard.
- H2 (closed-form LOO) is the highest-impact algorithmic change but needs care:
  pysamoo's RBF defaults are interpolating (H_ii = 1 ⇒ PRESS degenerates), so LOO
  applies cleanly to Kriging/GP and *regularized* RBF only.
