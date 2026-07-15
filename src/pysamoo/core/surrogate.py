"""Surrogate-model wrapper and surrogate-backed problem definition."""

import numpy as np
from pymoo.core.meta import Meta


class Surrogate:
    def __init__(self, problem, targets=None, **kwargs):
        """Build and update surrogates for the components of a Population.

        Objectives and inequality/equality constraints may each need their own model, and different model
        combinations are possible; this object wraps that bookkeeping behind fit/validate/predict.

        Args:
            problem: The optimization problem, used only for its metadata (it is never evaluated here).
            targets: Target objects describing how each population component is modeled (model type and
                hyper-parameters). The modular definition allows a different surrogate per target.
        """

        super().__init__(**kwargs)
        self._problem = problem
        self.targets = targets if targets is not None else []

    def validate(self, trn=None, tst=None, random_state=None, **kwargs):
        for target in self.targets:
            target.validate(trn=trn, tst=tst, random_state=random_state, **kwargs)

    def fit(self, sols):
        for target in self.targets:
            target.fit(sols)

    def performance(self, indicator, **kwargs):
        ret = {}
        for target in self.targets:
            ret[target.label] = target.performance(indicator, **kwargs)
        return ret

    def problem(self):
        return ProblemFromTargets(self._problem, self.targets)


class ProblemFromTargets(Meta):
    def __init__(self, problem, targets, **kwargs):
        super().__init__(problem, **kwargs)
        self.targets = targets

    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X)

        out["F"] = np.full((n, self.n_obj), np.nan, dtype=float)
        out["G"] = np.full((n, self.n_ieq_constr), np.nan, dtype=float)
        out["H"] = np.full((n, self.n_eq_constr), np.nan, dtype=float)

        for target in self.targets:
            target.predict(X, out)

        for v in ["F", "G", "H"]:
            if np.any(np.isnan(out[v])):
                raise RuntimeError(f"Surrogate prediction produced NaN values in '{v}'; the run was terminated.")
