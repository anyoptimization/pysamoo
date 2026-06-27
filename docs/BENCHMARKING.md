# Benchmarking & Developing Better Methods

pysamoo is, at its core, a **generic surrogate-assisted framework**: a pysamoo algorithm
wraps an ordinary pymoo algorithm and drives it with a **surrogate** built from pluggable
**models**. To develop a better method you therefore vary one of three things and *measure*:

1. the **base algorithm** (NSGA2, NSGA3, GA, DE, …),
2. the **surrogate model(s)** (Kriging, RBF, your own ezmodel model),
3. the **strategy parameters** (`alpha`, `beta`, infill counts, archive caps).

The `pysamoo.benchmark` package makes that measurement a few lines of code.

## The harness in one minute

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.multi import ZDT1
from pysamoo.algorithms.gpsaf import GPSAF
from pysamoo.benchmark import Scenario, ProblemSpec, run_benchmark, summarize, format_table

problem = ZDT1(n_var=10)

scenarios = [
    Scenario("NSGA2 (baseline)", lambda: NSGA2(pop_size=20, n_offsprings=10)),
    Scenario("GPSAF",            lambda: GPSAF(NSGA2(pop_size=20, n_offsprings=10), n_initial_doe=30)),
]
problems = [ProblemSpec("ZDT1", problem, n_evals=200)]

records = run_benchmark(scenarios, problems, n_seeds=5)
print(format_table(summarize(records)))
```

- **`Scenario(name, factory)`** — `factory` is a zero-arg callable returning a *fresh*
  algorithm (a new object per run, so seeds stay isolated).
- **`ProblemSpec(name, problem, n_evals)`** — a pymoo problem and its evaluation budget.
- **`run_benchmark(...)`** — runs every scenario × problem × seed and scores each run.
- **`summarize` / `format_table`** — aggregate to mean/std/best/time per pair.

### The score (lower is better)

| Problem type | Metric | Meaning |
|---|---|---|
| single-objective | `f_gap` | best feasible objective minus the known optimum |
| multi-objective (Pareto front known) | `igd` | inverted generational distance to the front |
| multi-objective (no front) | `hv_gap` | negative hypervolume vs an auto reference point |

Infeasible/empty runs score `inf` and are flagged via `feasible_rate`.

## Swapping the surrogate model — the "only needs models" idea

`SurrogateAssistedAlgorithm` accepts a `surrogate=` object. `make_surrogate` builds one from
per-output **model-set factories**; each factory returns a `{name: model}` dict and the
surrogate cross-validates/selects among them automatically.

```python
from ezmodel.models.rbf import RBF
from pysamoo.benchmark import make_surrogate

def only_rbf(**defaults):
    return {"rbf-cubic": RBF(kernel="cubic", **defaults)}

scenarios = [
    Scenario("GPSAF (Kriging, default)", lambda: GPSAF(NSGA2(), n_initial_doe=30)),
    Scenario("GPSAF (RBF only)",
             lambda: GPSAF(NSGA2(), n_initial_doe=30,
                           surrogate=make_surrogate(problem, obj_models=only_rbf))),
]
```

To try a brand-new surrogate, implement an [ezmodel](https://pypi.org/project/ezmodel/)
model and return it from your factory — no changes to the algorithms are needed.

## Example result

`src/pysamoo/usage/usage_benchmark.py` (ZDT1, `n_var=10`, 200 evals, 3 seeds) produces, for
instance:

```
problem      scenario         metric        mean       std       best  time/run  feas
-------------------------------------------------------------------------------------
ZDT1         GPSAF (Kriging)  igd        0.0378    0.0044    0.0340     30.3s  100%
ZDT1         GPSAF (RBF only) igd         0.282    0.0866     0.166      2.0s  100%
ZDT1         NSGA2 (baseline) igd         0.846     0.149     0.645      0.0s  100%
```

Reading it: at this budget the surrogate is hugely beneficial (IGD 0.04 / 0.28 vs 0.85), and
the model choice is an accuracy/speed trade-off — Kriging is far more accurate here, RBF is
~15× faster per run. This is exactly the loop you iterate when developing a better method.
(At a tighter 60-eval budget the gap narrows: Kriging ≈ 0.16, RBF ≈ 0.29, baseline ≈ 0.88.)

## Recommended workflow for algorithm development

1. **Fix a budget and a problem set** representative of your target (single- and
   multi-objective, constrained where relevant).
2. **Baseline first** — include the plain pymoo algorithm (no surrogate) as a scenario so you
   can prove the surrogate actually helps at that budget.
3. **Vary one thing at a time** — model family, `alpha`/`beta`, infill count, archive cap.
4. **Use several seeds** (`n_seeds>=5`) and compare `mean ± std`, not a single run.
5. **Watch `time/run`** alongside quality — see [PERFORMANCE.md](PERFORMANCE.md) for the
   cost model and speed-up levers.
6. **Lock in gains** — once a change helps, consider a `@pytest.mark.golden` baseline so a
   later refactor can't silently regress the numbers.
