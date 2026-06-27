"""Prototype: an intelligent model-selection pool that shrinks over time.

Idea (see model-selection-loop.md, hypothesis H8): instead of cross-validating the
whole candidate pool every iteration, maintain an *active* set that shrinks as
evidence accumulates, while protecting generalization with a per-family diversity
floor and periodically re-admitting pruned models to track the (non-stationary)
landscape as the archive grows.

This simulates a run (growing archive) and compares three strategies on
**cost** (cumulative model fits — platform-independent) and **generalization**
(held-out RMSE of the selected model):

  - full      : cross-validate all candidates every iteration (current behaviour)
  - family    : a fixed redundancy-free pool (one model per kernel family)
  - racing    : the adaptive shrink-and-readmit pool proposed here

Run:  pyclawd python .claude/docs/adaptive_pool_prototype.py
"""

import time
from collections import defaultdict, deque
from copy import deepcopy

import numpy as np
from pymoo.problems import get_problem
from pymoo.util.normalization import NoNormalization

from ezmodel.models.kriging import Kriging
from ezmodel.models.rbf import RBF


# --- candidate pool (name -> fresh-model factory) --------------------------------


def make_pool():
    pool = {}
    for kernel in ["linear", "cubic", "gaussian", "mq"]:
        for tail in ["constant", "linear", "quadratic"]:
            for norm in [False, True]:
                name = f"rbf-{kernel}-{tail}-{'norm' if norm else 'raw'}"
                pool[name] = ("rbf", dict(kernel=kernel, tail=tail, normalized=norm))
    for regr in ["constant", "linear", "quadratic"]:
        pool[f"kriging-{regr}"] = ("kriging", dict(regr=regr))
    return pool


def family_of(name):
    """Kernel family used for the diversity floor."""
    return name.split("-")[1] if name.startswith("rbf") else "kriging"


def build(spec):
    kind, kw = spec
    if kind == "rbf":
        return RBF(norm_X=NoNormalization(), **kw)
    return Kriging(**kw)


# --- deterministic CV scoring (mae; lower is better) -----------------------------


def strided_folds(n, k):
    idx = np.arange(n)
    return [(idx[idx % k != f], idx[idx % k == f]) for f in range(k)]


def cv_mae(spec, X, y, folds):
    errs = []
    for trn, tst in folds:
        try:
            m = build(spec)
            m.fit(X[trn], y[trn])
            yh = np.asarray(m.predict(X[tst])).ravel()
            errs.append(np.mean(np.abs(yh - y[tst])))
        except Exception:
            return np.inf
    return float(np.mean(errs))


def fit_full_and_test(spec, X, y, Xte, yte):
    m = build(spec)
    m.fit(X, y)
    yh = np.asarray(m.predict(Xte)).ravel()
    return float(np.sqrt(np.mean((yh - yte) ** 2)))


# --- the adaptive racing pool ----------------------------------------------------


class RacingPool:
    """Active-set model selection that shrinks over time, with a diversity floor
    and periodic re-admission.

    Parameters
    ----------
    warmup : iterations before any pruning (collect evidence first)
    window : rolling-window length for each model's mean CV score
    keep_ratio : fraction of the active set kept at each prune step (halving-ish)
    floor : never prune below this many models, and always keep >=1 per family
    readmit_every : every N iterations, re-admit a round-robin batch of pruned models
    readmit_batch : how many pruned models to re-admit each time
    """

    def __init__(self, pool, warmup=3, window=4, keep_ratio=0.6, floor=8, readmit_every=5, readmit_batch=4, rng=None):
        self.pool = pool
        self.active = list(pool.keys())
        self.pruned = []
        self.hist = defaultdict(lambda: deque(maxlen=window))
        self.warmup = warmup
        self.keep_ratio = keep_ratio
        self.floor = floor
        self.readmit_every = readmit_every
        self.readmit_batch = readmit_batch
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.t = 0

    def _rolling(self, name):
        h = self.hist[name]
        return np.mean(h) if h else np.inf

    def _prune(self):
        # rank active models by rolling mean CV error (lower is better)
        ranked = sorted(self.active, key=self._rolling)
        n_keep = max(self.floor, int(np.ceil(len(ranked) * self.keep_ratio)))
        keep = ranked[:n_keep]
        # diversity floor: ensure >=1 survivor per family represented in the pool
        kept_families = {family_of(m) for m in keep}
        for m in ranked[n_keep:]:
            fam = family_of(m)
            if fam not in kept_families:
                keep.append(m)
                kept_families.add(fam)
        newly_pruned = [m for m in self.active if m not in keep]
        self.pruned = newly_pruned + self.pruned  # round-robin order for re-admission
        self.active = keep

    def _readmit(self):
        batch, self.pruned = self.pruned[: self.readmit_batch], self.pruned[self.readmit_batch :]
        self.active = self.active + batch

    def select(self, X, y, folds):
        """Score the active set, update history, return (best_name, n_fits)."""
        self.t += 1
        if self.readmit_every and self.t % self.readmit_every == 0 and self.pruned:
            self._readmit()

        n_fits = 0
        for name in self.active:
            self.hist[name].append(cv_mae(self.pool[name], X, y, folds))
            n_fits += len(folds)

        best = min(self.active, key=lambda m: self.hist[m][-1])

        if self.t > self.warmup and len(self.active) > self.floor:
            self._prune()
        return best, n_fits


# --- simulation ------------------------------------------------------------------


def simulate(problem_name="ackley", n_var=5, n0=20, step=6, iters=20, k=5, seed=0):
    prob = get_problem(problem_name, n_var=n_var)
    xl, xu = prob.bounds()
    rng = np.random.RandomState(seed)
    Xte = rng.rand(500, n_var) * (xu - xl) + xl
    yte = prob.evaluate(Xte).ravel()

    pool = make_pool()
    family = {name: (kind == "rbf") for name, (kind, _) in pool.items()}  # noqa: F841

    racer = RacingPool(pool, rng=np.random.default_rng(seed))
    fixed_family = [n for n in pool if family_of(n) in {"cubic", "gaussian", "mq", "linear"} and n.endswith("norm")][:8]

    rows = []
    for t in range(iters):
        n = n0 + t * step
        X = rng.rand(n, n_var) * (xu - xl) + xl
        y = prob.evaluate(X).ravel()
        folds = strided_folds(n, k)

        # full pool every iteration
        t0 = time.perf_counter()
        full_scores = {name: cv_mae(spec, X, y, folds) for name, spec in pool.items()}
        full_best = min(full_scores, key=full_scores.get)
        full_t = time.perf_counter() - t0
        full_fits = len(pool) * k
        full_rmse = fit_full_and_test(pool[full_best], X, y, Xte, yte)

        # fixed family pool
        fam_scores = {name: cv_mae(pool[name], X, y, folds) for name in fixed_family}
        fam_best = min(fam_scores, key=fam_scores.get)
        fam_fits = len(fixed_family) * k
        fam_rmse = fit_full_and_test(pool[fam_best], X, y, Xte, yte)

        # racing pool
        race_best, race_fits = racer.select(X, y, folds)
        race_rmse = fit_full_and_test(pool[race_best], X, y, Xte, yte)

        rows.append((n, full_fits, full_rmse, fam_fits, fam_rmse, race_fits, len(racer.active), race_rmse))

    return rows


if __name__ == "__main__":
    for prob in ["ackley", "rastrigin"]:
        print("=" * 96)
        print(f"PROBLEM: {prob}   (fits = #model-fits that iteration; rmse = held-out generalization)")
        print("=" * 96)
        rows = simulate(prob)
        hdr = f"{'n':>4} | {'full_fits':>9} {'full_rmse':>9} | {'fam_fits':>8} {'fam_rmse':>8} | {'race_fits':>9} {'race_act':>8} {'race_rmse':>9}"
        print(hdr)
        print("-" * len(hdr))
        for (n, ff, fr, af, ar, rf, ra, rr) in rows:
            print(f"{n:>4} | {ff:>9} {fr:>9.3f} | {af:>8} {ar:>8.3f} | {rf:>9} {ra:>8} {rr:>9.3f}")
        tot_full = sum(r[1] for r in rows)
        tot_fam = sum(r[3] for r in rows)
        tot_race = sum(r[5] for r in rows)
        mean_full = np.mean([r[2] for r in rows])
        mean_fam = np.mean([r[4] for r in rows])
        mean_race = np.mean([r[7] for r in rows])
        print("-" * len(hdr))
        print(f"TOTAL fits  full={tot_full}  family={tot_fam}  racing={tot_race}   "
              f"(racing/full = {tot_race / tot_full:.0%})")
        print(f"MEAN  rmse  full={mean_full:.3f}  family={mean_fam:.3f}  racing={mean_race:.3f}")
