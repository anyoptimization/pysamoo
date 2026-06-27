"""Benchmark runner for comparing surrogate-assisted algorithms and models.

The goal is to make it cheap to answer "is algorithm/surrogate A better than B
under a small evaluation budget?". You describe *scenarios* (named factories that
each build a fresh algorithm) and *problem specs* (a problem + an evaluation
budget), and :func:`run_benchmark` runs every scenario on every problem across
several seeds, scoring each run with a single "lower-is-better" metric:

* single-objective problems -> best feasible objective value (the gap to the
  known optimum when the problem exposes one);
* multi-objective problems  -> IGD against the problem's Pareto front, or
  hypervolume distance from a reference point when no front is available.

Because :class:`~pysamoo.core.algorithm.SurrogateAssistedAlgorithm` accepts a
pluggable ``surrogate=`` object, the very same harness compares *surrogate
models* — see :func:`pysamoo.benchmark.models.make_surrogate`.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize


@dataclass
class Scenario:
    """A named, repeatable way to build an algorithm.

    Args:
        name: Label shown in result tables.
        factory: Zero-argument callable returning a *fresh* algorithm instance.
            It must build a new object on every call so seeds/runs stay isolated.
    """

    name: str
    factory: object  # Callable[[], Algorithm]


@dataclass
class ProblemSpec:
    """A benchmark problem paired with its evaluation budget.

    Args:
        name: Label shown in result tables.
        problem: A pymoo ``Problem`` instance.
        n_evals: Total number of true function evaluations allowed per run.
        ref_point: Reference point for hypervolume when no Pareto front exists.
    """

    name: str
    problem: object
    n_evals: int
    ref_point: object = None


@dataclass
class Record:
    """One scored run of a scenario on a problem with a single seed."""

    scenario: str
    problem: str
    seed: int
    score: float
    metric: str
    n_eval: int
    runtime: float
    feasible: bool


@dataclass
class Summary:
    """Aggregated statistics for a (scenario, problem) pair over all seeds."""

    scenario: str
    problem: str
    metric: str
    mean: float
    std: float
    best: float
    mean_runtime: float
    feasible_rate: float
    n_runs: int
    raw: list = field(default_factory=list)


def score_run(problem, res):
    """Score a finished run with a single lower-is-better metric.

    Returns:
        A ``(metric_name, value, feasible)`` tuple. ``value`` is ``inf`` and
        ``feasible`` is ``False`` when the run produced no feasible solution.
    """
    if problem.n_obj == 1:
        F = None if res.F is None else np.atleast_1d(np.asarray(res.F, dtype=float)).ravel()
        if F is None or F.size == 0 or not np.isfinite(F).all():
            return "f_gap", float("inf"), False
        pf = _safe_pf(problem)
        ideal = float(pf.min()) if pf is not None else 0.0
        return "f_gap", float(F.min()) - ideal, True

    # multi-objective
    F = None if res.F is None else np.atleast_2d(np.asarray(res.F, dtype=float))
    if F is None or F.size == 0 or not np.isfinite(F).all():
        return "igd", float("inf"), False
    pf = _safe_pf(problem)
    if pf is not None:
        return "igd", float(IGD(pf)(F)), True
    ref = res_ref_point(problem, F)
    return "hv_gap", float(_hv_gap(F, ref)), True


def _safe_pf(problem):
    try:
        pf = problem.pareto_front()
    except Exception:
        return None
    if pf is None:
        return None
    return np.atleast_2d(np.asarray(pf, dtype=float))


def res_ref_point(problem, F):
    """A reference point worse than every observed objective vector."""
    return F.max(axis=0) + 1.0


def _hv_gap(F, ref):
    """Hypervolume turned into a lower-is-better score (negative hypervolume)."""
    return -float(HV(ref_point=ref)(F))


def run_benchmark(scenarios, problems, n_seeds=5, seed0=1, verbose=True):
    """Run every scenario on every problem across ``n_seeds`` seeds.

    Args:
        scenarios: Iterable of :class:`Scenario`.
        problems: Iterable of :class:`ProblemSpec`.
        n_seeds: Number of independent seeds per (scenario, problem) pair.
        seed0: First seed; subsequent seeds are ``seed0 + i``.
        verbose: When ``True``, print one line per finished run.

    Returns:
        A list of :class:`Record`, one per run.
    """
    records = []
    for spec in problems:
        for scn in scenarios:
            for i in range(n_seeds):
                seed = seed0 + i
                algorithm = scn.factory()
                t0 = time.perf_counter()
                res = minimize(spec.problem, algorithm, ("n_evals", spec.n_evals), seed=seed, verbose=False)
                runtime = time.perf_counter() - t0
                metric, score, feasible = score_run(spec.problem, res)
                n_eval = (
                    int(getattr(getattr(res, "algorithm", None), "evaluator", None).n_eval)
                    if res.algorithm
                    else spec.n_evals
                )
                rec = Record(scn.name, spec.name, seed, score, metric, n_eval, runtime, feasible)
                records.append(rec)
                if verbose:
                    print(
                        f"{spec.name:12s} {scn.name:16s} seed={seed} "
                        f"{metric}={score:.4g} t={runtime:.1f}s feasible={feasible}",
                        flush=True,
                    )
    return records


def summarize(records):
    """Aggregate raw records into one :class:`Summary` per (scenario, problem)."""
    groups = {}
    for r in records:
        groups.setdefault((r.problem, r.scenario), []).append(r)

    summaries = []
    for (problem, scenario), recs in groups.items():
        scores = np.array([r.score for r in recs], dtype=float)
        finite = scores[np.isfinite(scores)]
        runtimes = np.array([r.runtime for r in recs], dtype=float)
        summaries.append(
            Summary(
                scenario=scenario,
                problem=problem,
                metric=recs[0].metric,
                mean=float(finite.mean()) if finite.size else float("inf"),
                std=float(finite.std()) if finite.size else float("nan"),
                best=float(finite.min()) if finite.size else float("inf"),
                mean_runtime=float(runtimes.mean()),
                feasible_rate=float(np.mean([r.feasible for r in recs])),
                n_runs=len(recs),
                raw=recs,
            )
        )
    return summaries


def format_table(summaries):
    """Render summaries as an aligned, sorted text table (best mean first)."""
    rows = sorted(summaries, key=lambda s: (s.problem, s.mean))
    header = f"{'problem':12s} {'scenario':16s} {'metric':7s} {'mean':>10s} {'std':>9s} {'best':>10s} {'time/run':>9s} {'feas':>5s}"
    lines = [header, "-" * len(header)]
    for s in rows:
        lines.append(
            f"{s.problem:12s} {s.scenario:16s} {s.metric:7s} "
            f"{s.mean:10.4g} {s.std:9.3g} {s.best:10.4g} {s.mean_runtime:8.1f}s {s.feasible_rate:5.0%}"
        )
    return "\n".join(lines)
