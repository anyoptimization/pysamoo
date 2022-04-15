import numpy as np

from pymoo.core.problem import Problem
from pymoo.problems.meta import MetaProblem


class Surrogate:

    def __init__(self,
                 problem,
                 targets=None,
                 **kwargs):

        """

        This surrogate object allows to conveniently build and update surrogates for a Population object.
        This can be a rather complicated task because surrogates for objectives and ieq. and eq. constraints might
        need to be build and different combinations of doing that are possible.

        Parameters
        ----------
        problem : Problem
            The optimization problem to access the meta data (it will never be called for an evaluation)

        targets : list
            A list of target objects describing how each of the components of a population should be modeled.
            This also includes the type of model and other hyper-parameters. This modular definition is necessary
            because different types of surrogate might be used for different _targets.

        """

        super().__init__(**kwargs)
        self._problem = problem
        self.targets = targets if targets is not None else []

    def validate(self, trn=None, tst=None, **kwargs):
        for target in self.targets:
            target.validate(trn=trn, tst=tst, **kwargs)

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


class ProblemFromTargets(MetaProblem):

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
                raise Exception("Building Surrogate has failed (nan values were predict). The run has been terminated.")

        out["F_estm_error"] = np.full((n, self.n_obj), np.nan, dtype=float)
        out["G_estm_error"] = np.full((n, self.n_ieq_constr), np.nan, dtype=float)
        out["H_estm_error"] = np.full((n, self.n_eq_constr), np.nan, dtype=float)

        for target in self.targets:
            label, k = target.label
            out.get(label + "_estm_error")[:, k] = target.performance("mae")
