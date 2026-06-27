"""Helpers for plugging custom surrogate models into the benchmark.

pysamoo's surrogate-assisted algorithms are *generic*: they only need a
:class:`~pysamoo.core.surrogate.Surrogate` describing how to model each problem
output. :func:`make_surrogate` builds that object from per-output model-set
factories, so swapping the model family is a one-liner. Each model-set factory
takes keyword defaults (e.g. ``norm_X``) and returns a ``{name: model}`` dict;
the surrogate cross-validates and selects among them automatically.

Example:
    >>> from ezmodel.models.rbf import RBF
    >>> def only_rbf(**defaults):
    ...     return {"rbf-cubic": RBF(kernel="cubic", **defaults)}
    >>> surrogate = make_surrogate(problem, obj_models=only_rbf)
    >>> algorithm = GPSAF(NSGA2(), surrogate=surrogate)
"""

from pysamoo.core.algorithm import MyNormalization
from pysamoo.core.defaults import (
    DEFAULT_EQ_CONSTR_MODELS,
    DEFAULT_IEQ_CONSTR_MODELS,
    DEFAULT_OBJ_MODELS,
)
from pysamoo.core.surrogate import Surrogate
from pysamoo.core.target import Target


def make_surrogate(
    problem,
    obj_models=DEFAULT_OBJ_MODELS,
    ieq_models=DEFAULT_IEQ_CONSTR_MODELS,
    eq_models=DEFAULT_EQ_CONSTR_MODELS,
):
    """Build a :class:`Surrogate` for ``problem`` from per-output model factories.

    Args:
        problem: The optimization problem (used only for metadata and bounds).
        obj_models: Factory returning a ``{name: model}`` dict for each objective.
        ieq_models: Factory for each inequality constraint.
        eq_models: Factory for each equality constraint.

    Returns:
        A :class:`~pysamoo.core.surrogate.Surrogate` ready to pass as the
        ``surrogate=`` argument of a surrogate-assisted algorithm.
    """
    xl, xu = problem.bounds()
    defaults = dict(norm_X=MyNormalization(xl, xu))

    targets = []
    for m in range(problem.n_obj):
        targets.append(Target(("F", m), obj_models(**defaults)))
    for g in range(problem.n_ieq_constr):
        targets.append(Target(("G", g), ieq_models(**defaults)))
    for h in range(problem.n_eq_constr):
        targets.append(Target(("H", h), eq_models(**defaults)))

    return Surrogate(problem, targets)
