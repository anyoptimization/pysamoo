"""Empirically check: is LOO-CV a worse model selector than k-fold CV?

For each candidate surrogate we compute three numbers on the same training set:
  - cv5  : 5-fold CV mean-absolute-error (pysamoo's current criterion)
  - loo  : leave-one-out CV mae (== the closed-form GP LOO error, just computed
           the slow way; for GP the closed form gives identical values)
  - test : the TRUE mae on a large held-out set (ground-truth generalization)

A good selection criterion (a) ranks models like `test` does (high rank
correlation) and (b) the model it *picks* has low true `test` error. We compare
cv5 vs loo on both, across seeds and problems.

Run:  pyclawd python .claude/docs/loo_vs_kfold.py
"""

from copy import deepcopy

import numpy as np
from pymoo.problems import get_problem
from pymoo.util.normalization import NoNormalization

from ezmodel.models.kriging import Kriging
from ezmodel.models.rbf import RBF


def make_pool():
    pool = {}
    for kernel in ["cubic", "gaussian", "mq"]:
        for norm in [False, True]:
            pool[f"rbf-{kernel}-{'n' if norm else 'r'}"] = ("rbf", dict(kernel=kernel, normalized=norm))
    for regr in ["constant", "linear", "quadratic"]:
        pool[f"kriging-{regr}"] = ("kriging", dict(regr=regr))
    return pool


def build(spec):
    kind, kw = spec
    return RBF(norm_X=NoNormalization(), **kw) if kind == "rbf" else Kriging(**kw)


def cv_mae(spec, X, y, folds):
    errs = []
    for trn, tst in folds:
        try:
            m = build(spec)
            m.fit(X[trn], y[trn])
            errs.append(np.mean(np.abs(np.asarray(m.predict(X[tst])).ravel() - y[tst])))
        except Exception:
            return np.inf
    return float(np.mean(errs))


def folds_kfold(n, k, rng):
    order = rng.permutation(n)
    pos = np.arange(n)
    return [(order[pos % k != f], order[pos % k == f]) for f in range(k)]


def folds_loo(n):
    idx = np.arange(n)
    return [(np.delete(idx, i), np.array([i])) for i in range(n)]


def spearman(a, b):
    """Rank correlation (no scipy)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return np.nan
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def test_mae(spec, X, y, Xte, yte):
    try:
        m = build(spec)
        m.fit(X, y)
        return float(np.mean(np.abs(np.asarray(m.predict(Xte)).ravel() - yte)))
    except Exception:
        return np.inf


def run(problem_name, n_var=5, n=30, seeds=(0, 1, 2, 3, 4)):
    prob = get_problem(problem_name, n_var=n_var)
    xl, xu = prob.bounds()
    pool = make_pool()

    rho_cv5, rho_loo, pick_gap_cv5, pick_gap_loo, agree = [], [], [], [], []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        X = rng.rand(n, n_var) * (xu - xl) + xl
        y = prob.evaluate(X).ravel()
        Xte = rng.rand(800, n_var) * (xu - xl) + xl
        yte = prob.evaluate(Xte).ravel()

        names = list(pool)
        cv5 = [cv_mae(pool[m], X, y, folds_kfold(n, 5, np.random.RandomState(seed))) for m in names]
        loo = [cv_mae(pool[m], X, y, folds_loo(n)) for m in names]
        tst = [test_mae(pool[m], X, y, Xte, yte) for m in names]

        rho_cv5.append(spearman(cv5, tst))
        rho_loo.append(spearman(loo, tst))

        best_test = min(tst)
        pick_cv5 = names[int(np.argmin(cv5))]
        pick_loo = names[int(np.argmin(loo))]
        pick_gap_cv5.append(tst[names.index(pick_cv5)] - best_test)
        pick_gap_loo.append(tst[names.index(pick_loo)] - best_test)
        agree.append(pick_cv5 == pick_loo)

    print(f"\n{problem_name} (n_var={n_var}, n={n}, {len(seeds)} seeds, pool={len(pool)})")
    print(f"  rank-corr with TRUE test error   : 5-fold={np.nanmean(rho_cv5):+.3f}   LOO={np.nanmean(rho_loo):+.3f}   (higher=better selector)")
    print(f"  test-error gap of PICKED model    : 5-fold={np.mean(pick_gap_cv5):.4f}   LOO={np.mean(pick_gap_loo):.4f}   (lower=better pick)")
    print(f"  5-fold and LOO pick same model    : {np.mean(agree):.0%} of seeds")


if __name__ == "__main__":
    for p in ["ackley", "rastrigin", "sphere"]:
        run(p)
