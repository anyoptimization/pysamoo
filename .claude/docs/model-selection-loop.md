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
| H2 | **Closed-form LOO-CV** (GPML eq. 5.12) selects the same model as 5-fold at a fraction of cost | new strategy; later `target.py`/`ezmodel` | LOO-rank vs kfold-rank agreement; cost | same top-1 ≥90% of cases, ≥3× faster, deterministic | pending |
| H3 | **Lazy re-selection** every k iters (wire up dead `nth_validate`) keeps convergence | `algorithm.py`/`gpsaf.py` `_advance` | full-run IGD/gap vs k∈{1,5,10} at fixed budget | IGD within +5% at k≥5, large wall-clock drop | pending |
| H4 | **Cap GP fit to L-nearest subset** bounds O(n³) without hurting generalization | `_doe()` / fit path | RMSE & cost vs cap L∈{80,160,∞}, large n | cost ~constant in n, RMSE within +5% | pending |
| H5 | **Seed folds + deterministic tie-break** ⇒ reproducible runs, no quality change | `target.py:73`, `target.py:119` | determinism check + RMSE unchanged | identical selection across runs; RMSE unchanged | **confirmed** for selection (seeding); ship + golden test |
| H6 | **PRESS-weighted ensemble** ≥ single-best generalization, free uncertainty | `target.py` find_best/predict | ensemble vs best RMSE on suite | RMSE ≤ best, no extra fits | pending |
| H7 | **Rank by log pseudo-likelihood** (not integer kendall_tau) removes frequent ties | `target.py` indicators | tie frequency; selection stability | fewer ties, stable choice | pending |

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
| 2026-06-27 | 1 | H5 | full(38), seeded folds | same | unchanged | **yes** (identical across runs) | **confirmed** — TODO: ship + deterministic tie-break |

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
- H5 (seeding) is the cheapest shippable win and unblocks **full-run golden tests**
  (currently only deterministic kernels are baselined; see `tests/test_golden.py`).
  Shipping H5 should pair `CrossvalidationPartitioning(self.n_folds, seed=…)` at
  `target.py:73` with a deterministic tie-break at `target.py:119`, *and* thread
  pymoo's per-run `self.random_state` into pysamoo's global-RNG sites
  (`gpsaf.py`, `psaf.py`) so the *whole* run — not just selection — is reproducible.
- H2 (closed-form LOO) is the highest-impact algorithmic change but needs care:
  pysamoo's RBF defaults are interpolating (H_ii = 1 ⇒ PRESS degenerates), so LOO
  applies cleanly to Kriging/GP and *regularized* RBF only.
