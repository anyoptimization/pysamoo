"""Adaptive racing model-selection target.

A :class:`~pysamoo.core.target.Target` cross-validates the *whole* candidate pool
every iteration, which dominates runtime (the pool is large and Gaussian-process
fits scale O(n^3) with the growing archive). :class:`RacingTarget` keeps an
*active* subset that shrinks as evidence accumulates, so later (expensive)
iterations score far fewer models:

* **prune** models with consistently worse rolling cross-validation error,
* keep a **per-family diversity floor** (>=1 model per kernel family) so the pool
  never loses the model family that generalizes — the safeguard that makes naive
  pool-shrinking fail,
* periodically **re-admit** pruned models so a model that only becomes good on a
  larger archive can return (handles the non-stationary, clustered data a real
  optimization run produces).

Selection stays reproducible: pruning/re-admission are deterministic functions of
the recorded performances. See .claude/docs/model-selection-loop.md (H8).
"""

from pysamoo.core.target import Target


def family_of(name):
    """Group key for the diversity floor: the RBF kernel, or ``"kriging"``."""
    name = str(name)
    if "kernel=" in name:
        return name.split("kernel=", 1)[1].split(",", 1)[0].rstrip("]")
    if name.lower().startswith("kriging") or name.lower().startswith("krg"):
        return "kriging"
    return name


class RacingTarget(Target):
    """A :class:`Target` whose candidate pool shrinks over time.

    Args:
        warmup: Iterations to score the full pool before any pruning (collect
            evidence first).
        keep_ratio: Fraction of the active set retained at each prune step.
        floor: Never prune below this many models.
        readmit_every: Re-admit a batch of pruned models every this many iterations
            (0 disables re-admission).
        readmit_batch: Number of pruned models re-admitted each time.
    """

    def __init__(
        self,
        label,
        models,
        warmup=3,
        keep_ratio=0.6,
        floor=8,
        readmit_every=5,
        readmit_batch=4,
        **kwargs,
    ):
        super().__init__(label, models, **kwargs)
        self._all_models = dict(models)
        self.active = list(models.keys())
        self._pruned = []  # round-robin queue for re-admission
        self.warmup = warmup
        self.keep_ratio = keep_ratio
        self.floor = max(floor, len({family_of(m) for m in models}))
        self.readmit_every = readmit_every
        self.readmit_batch = readmit_batch
        self._t = 0

    def validate(self, trn, tst=None, find_best=True, random_state=None, **kwargs):
        self._t += 1

        # re-admit a batch of previously pruned models to track a shifting landscape
        if self.readmit_every and self._t % self.readmit_every == 0 and self._pruned:
            batch, self._pruned = self._pruned[: self.readmit_batch], self._pruned[self.readmit_batch :]
            self.active = self.active + batch

        # score only the active subset this iteration
        full = self.models
        self.models = {name: self._all_models[name] for name in self.active}
        try:
            super().validate(trn, tst=tst, find_best=find_best, random_state=random_state, **kwargs)
        finally:
            self.models = full

        if self._t > self.warmup and len(self.active) > self.floor:
            self._prune()

    def _rolling(self, name):
        """Mean recent cross-validation error for ranking (lower is better)."""
        try:
            return float(self.performance("mae", model=name))
        except Exception:
            return float("inf")

    def _prune(self):
        ranked = sorted(self.active, key=self._rolling)
        n_keep = max(self.floor, int(round(len(ranked) * self.keep_ratio)))
        keep = ranked[:n_keep]

        # diversity floor: keep at least one survivor per kernel family
        kept_families = {family_of(m) for m in keep}
        for m in ranked[n_keep:]:
            fam = family_of(m)
            if fam not in kept_families:
                keep.append(m)
                kept_families.add(fam)

        newly_pruned = [m for m in self.active if m not in keep]
        self._pruned = newly_pruned + self._pruned
        self.active = [m for m in self.active if m in keep]
