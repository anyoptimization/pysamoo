"""Experiment harness for the model-selection research loop.

Measures, for a given *selection strategy*, the three quantities the loop trades
off (see model-selection-loop.md):

  1. cost         — wall-clock seconds to select a model,
  2. generalization — out-of-sample accuracy of the selected model on a held-out
                     test set (RMSE and rank correlation), the property that
                     model selection exists to protect,
  3. determinism  — whether the selection is identical under a perturbed ambient
                     RNG state (the failure mode that makes full runs irreproducible).

Run:  pyclawd python .claude/docs/model_selection_bench.py

A "strategy" is just a callable that, given a training Population and a Target
factory, returns (selected_label, fitted_model, seconds). New loop iterations add
strategies here and append their numbers to the results log in the loop doc.
"""

import random
import time

import numpy as np
from pymoo.core.population import Population
from pymoo.problems import get_problem
from pymoo.util.normalization import NoNormalization

from ezmodel.models.kriging import Kriging
from ezmodel.models.rbf import RBF
from pysamoo.core.defaults import DEFAULT_OBJ_MODELS
from pysamoo.core.target import Target

# --- model pools -----------------------------------------------------------------


def pool_full(**d):
    """The production pool: 38 candidates (32 RBF + 6 Kriging)."""
    return DEFAULT_OBJ_MODELS(**d)


def pool_small(**d):
    """A hand-picked 3-model pool (cheap but narrow)."""
    return {
        "rbf-cubic": RBF(kernel="cubic", **d),
        "rbf-cubic-norm": RBF(kernel="cubic", normalized=True, **d),
        "kriging-lin": Kriging(regr="linear"),
    }


def pool_one_per_family(**d):
    """One representative per kernel family + one Kriging (8 candidates)."""
    p = {}
    for k in ["linear", "cubic", "gaussian", "mq"]:
        p[f"rbf-{k}"] = RBF(kernel=k, **d)
        p[f"rbf-{k}-norm"] = RBF(kernel=k, normalized=True, **d)
    return p


POOLS = {"full(38)": pool_full, "small(3)": pool_small, "family(8)": pool_one_per_family}


# --- strategies ------------------------------------------------------------------


def strat_kfold(pool_factory, n_folds=5, seed_folds=None):
    """Current pysamoo behaviour: k-fold CV over the whole pool, pick best.

    With ``seed_folds`` set, the global RNG is seeded before validation so the
    CV partition (and hence the choice) is reproducible.
    """

    def run(pop):
        if seed_folds is not None:
            random.seed(seed_folds)
            np.random.seed(seed_folds)
        t = Target(("F", 0), pool_factory(norm_X=NoNormalization()), n_folds=n_folds)
        t0 = time.perf_counter()
        t.validate(pop)
        dt = time.perf_counter() - t0
        t.fit(pop)
        return t.best, t.obj, dt

    return run


# --- metrics ---------------------------------------------------------------------


def kendall_tau(a, b):
    """Fraction of concordant pairs minus discordant (rank agreement, in [-1, 1])."""
    n = len(a)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return (c - d) / (c + d) if (c + d) else 0.0


def evaluate(strategy, problem, n_train, n_test=400, seed=0):
    """Run a strategy and score cost + generalization on held-out points."""
    rng = np.random.RandomState(seed)
    xl, xu = problem.bounds()
    Xtr = rng.rand(n_train, problem.n_var) * (xu - xl) + xl
    ytr = problem.evaluate(Xtr).reshape(-1, 1)
    Xte = rng.rand(n_test, problem.n_var) * (xu - xl) + xl
    yte = problem.evaluate(Xte).ravel()

    pop = Population.new(X=Xtr, F=ytr)
    label, model, dt = strategy(pop)
    yhat = model.predict(Xte).ravel()
    rmse = float(np.sqrt(np.mean((yhat - yte) ** 2)))
    tau = kendall_tau(yte[:120], yhat[:120])  # rank fidelity on a subset (cheap)
    return dict(label=label, seconds=dt, rmse=rmse, tau=tau)


def determinism_check(make_strategy, problem, n_train=60, trials=3):
    """A strategy is deterministic if the selected model is identical across trials
    that start from *different* ambient RNG states (mimicking a real run)."""
    labels = []
    for k in range(trials):
        # perturb ambient state the way a real optimization loop would
        np.random.seed(); random.seed()
        [np.random.rand() for _ in range(k * 7 + 1)]
        res = evaluate(make_strategy(), problem, n_train)
        labels.append(res["label"])
    return len(set(labels)) == 1, labels


# --- main ------------------------------------------------------------------------

if __name__ == "__main__":
    problems = [("ackley", get_problem("ackley", n_var=5)), ("rastrigin", get_problem("rastrigin", n_var=5))]

    print("=" * 92)
    print("COST vs GENERALIZATION  (lower rmse = better generalization; tau closer to 1 = better ranking)")
    print("=" * 92)
    header = f"{'problem':10s} {'pool':10s} {'n_train':>7s} {'sel_s':>8s} {'test_rmse':>10s} {'test_tau':>9s}  picked"
    print(header)
    print("-" * 92)
    for pname, prob in problems:
        for n in [40, 80]:
            for poolname, pool in POOLS.items():
                r = evaluate(strat_kfold(pool, n_folds=5), prob, n)
                print(
                    f"{pname:10s} {poolname:10s} {n:7d} {r['seconds']:8.2f} {r['rmse']:10.3f} {r['tau']:9.3f}  {r['label']}"
                )
        print("-" * 92)

    print("\n" + "=" * 92)
    print("DETERMINISM  (selected model identical across perturbed ambient RNG states?)")
    print("=" * 92)
    prob = get_problem("ackley", n_var=5)
    ok_unseeded, labels_u = determinism_check(lambda: strat_kfold(pool_full, n_folds=5), prob)
    ok_seeded, labels_s = determinism_check(lambda: strat_kfold(pool_full, n_folds=5, seed_folds=1), prob)
    print(f"unseeded folds : deterministic={ok_unseeded}  picks={labels_u}")
    print(f"seeded folds   : deterministic={ok_seeded}  picks={labels_s}")
