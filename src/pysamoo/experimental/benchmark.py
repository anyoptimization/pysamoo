"""Fast BBOB-style ECDF benchmark for the BO method search (ORACLE -- do not edit when iterating).

The feedback function the method-search loop optimizes against. It runs the configured
``BayesianOptimization`` on the fixed suite (Sphere / Rosenbrock / Rastrigin / RotatedEllipsoid at
2D and 10D, optimum F*=0) and reports a single **ECDF score** in [0, 1].

Design goal: be a *good gradient* for method development, not just a pass/fail. The targets are
**scale-normalized per function** -- decades below each function's own typical magnitude -- so
*partial progress always registers* (improving 10D Rosenbrock 220->50 moves the score) and all
functions are comparable despite spanning ~6 orders of magnitude. ECDF = fraction of
``(function x target x seed)`` triples reached within the evaluation budget. Wall-clock is part of
the score because the method must stay fast. Tuned to run in ~20s.
"""

import time
import warnings

import numpy as np
from pymoo.optimize import minimize

from pysamoo.experimental.problems import benchmark_problems, heldout_problems

warnings.filterwarnings("ignore")

# targets are R_f * 10^(-k): decades below each function's reference scale R_f. 17 levels over 8
# decades = half-decade granularity, so the ECDF behaves like a smooth normalized log-regret.
TARGET_DECADES = np.linspace(0.0, 8.0, 17)

# 3 seeds, not 2: with only 2 seeds the ECDF was too noisy to reliably reward real improvements
# (a genuine +3% gain on the hard 10D functions landed on seeds 3-5 and was invisible on seeds 1-2).
# 3 seeds is the speed/reliability compromise that keeps the run near the ~20s iteration budget.
SEEDS = (1, 2, 3)


def budget_for(d):
    """Return the function-evaluation budget for a ``d``-dimensional problem."""
    return 30 + 5 * d


def reference_scale(problem, n=2000, seed=0):
    """Reference magnitude R_f for a function: the median objective over random points in its box.

    Targets are taken as decades below this, so every function -- whether its values are ~1 or ~1e6
    -- contributes a meaningful, comparable 0..1 progress signal.

    Args:
        problem: The pymoo problem.
        n: Number of random points to estimate the scale from.
        seed: RNG seed (fixed, so the scale is deterministic).

    Returns:
        Median objective value over the random sample (a positive float).
    """
    rng = np.random.default_rng(seed)
    X = problem.xl + rng.random((n, problem.n_var)) * (problem.xu - problem.xl)
    F = problem.evaluate(X)[:, 0]
    return float(np.median(F))


def make_default_algorithm():
    """Build a fresh BayesianOptimization with the current default configuration (the thing under test)."""
    from pysamoo.experimental.bo import BayesianOptimization

    return BayesianOptimization()


def run_one(problem, seed, make_algorithm):
    """Run one optimization; return the best objective reached (gap to optimum, since F*=0)."""
    res = minimize(problem, make_algorithm(), ("n_evals", budget_for(problem.n_var)), seed=seed, verbose=False)
    return float(res.F[0])


def evaluate_on(problems, make_algorithm=make_default_algorithm, seeds=SEEDS, verbose=True):
    """Run a given problem suite and return its ECDF score, wall-clock, and per-problem breakdown.

    Shared core of :func:`evaluate` (the tuned suite) and :func:`evaluate_heldout` (the held-out
    generalization suite). A method should improve *both*; a gain on the tuned suite that does not
    carry to the held-out suite is overfitting and must not be kept.

    Args:
        problems: List of ``(name, problem)`` to run.
        make_algorithm: Zero-arg factory returning a fresh algorithm to benchmark.
        seeds: Seeds run per problem.
        verbose: Print a per-problem table and the headline score.

    Returns:
        Dict with ``ecdf``, ``time``, ``per_problem``, and ``score`` (ecdf with a speed tie-break).
    """
    t0 = time.time()
    per_problem, solved_flags = {}, []
    for name, problem in problems:
        ref = reference_scale(problem)
        targets = ref * 10.0 ** (-TARGET_DECADES)
        best = [run_one(problem, s, make_algorithm) for s in seeds]
        solved = [[b <= t for t in targets] for b in best]
        solved_flags.extend(np.array(solved).ravel().tolist())
        frac = float(np.mean(solved))
        per_problem[name] = {"best": best, "solved_frac": frac, "ref": ref}
        if verbose:
            print(f"  {name:20} bestF={np.median(best):.2e}  ref={ref:.1e}  solved={frac:5.2f}", flush=True)
    elapsed = time.time() - t0
    ecdf = float(np.mean(solved_flags))
    score = ecdf - 1e-4 * elapsed
    if verbose:
        print(f"\n  ECDF = {ecdf:.4f}   time = {elapsed:.1f}s   score = {score:.4f}", flush=True)
    return {"ecdf": ecdf, "time": elapsed, "per_problem": per_problem, "score": score}


def evaluate_heldout(make_algorithm=make_default_algorithm, seeds=SEEDS, verbose=True):
    """Run the held-out suite (Ackley/Griewank/Zakharov) -- the overfitting guard. See :func:`evaluate_on`."""
    return evaluate_on(heldout_problems(), make_algorithm=make_algorithm, seeds=seeds, verbose=verbose)


def evaluate(make_algorithm=make_default_algorithm, seeds=SEEDS, verbose=True):
    """Run the full suite and return the ECDF score, wall-clock, and per-problem breakdown.

    Args:
        make_algorithm: Zero-arg factory returning a fresh algorithm to benchmark.
        seeds: Seeds run per problem (more = less noisy ECDF, more time).
        verbose: Print a per-problem table and the headline score.

    Returns:
        Dict with ``ecdf`` (scalar in [0,1]), ``time`` (seconds), ``per_problem`` (name -> best F
        and solved-fraction), and ``score`` (ecdf with a tiny speed tie-break).
    """
    return evaluate_on(benchmark_problems(), make_algorithm=make_algorithm, seeds=seeds, verbose=verbose)


def main():
    """Run the tuned and held-out suites on the current default algorithm and print both reports."""
    print(f"BO ECDF benchmark — {len(benchmark_problems())} tuned problems, scale-normalized targets\n")
    tuned = evaluate()
    print(f"\nHeld-out generalization suite ({len(heldout_problems())} problems — never tuned against):\n")
    held = evaluate_heldout()
    print(f"\n  SUMMARY: tuned ECDF = {tuned['ecdf']:.4f}   held-out ECDF = {held['ecdf']:.4f}")


if __name__ == "__main__":
    main()
