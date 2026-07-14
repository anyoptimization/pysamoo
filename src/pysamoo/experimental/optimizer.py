"""Pluggable acquisition-function optimizers for Bayesian optimization.

An *acquisition optimizer* maximizes an acquisition function (e.g. Expected
Improvement) over a problem's box to choose the next point to evaluate. The base
:class:`Optimizer` defines a single ``optimize`` method; concrete strategies
implement it, and :class:`~pysamoo.experimental.bo.BayesianOptimization` accepts any
:class:`Optimizer` instance.

Built-in strategies:

* :class:`VectorizedGradientDescent` -- the default. Maximizes EI with pysurrogate's
  generic :class:`~pysurrogate.optimizer.Adam` over an
  :class:`~pysamoo.experimental.acquisition.EIProblem`: a population climbs ``-EI`` by the
  surrogate's analytic mean/variance gradients, one batched ``predict(grad=True)`` per step.
  Fast and seed-independent on a well-fit surrogate -- no maximin seed pool needed.
* :class:`GeneticAlgorithm` -- derivative-free niching GA over a (plain) LHS seed. Robust on
  any model and any acquisition; the fallback when the acquisition is not EI.

``optimize`` returns the chosen point and the acquisition value there, in the *minimization*
convention that matches :class:`~pysamoo.experimental.acquisition.AcquisitionProblem` (lower is
better, i.e. ``-EI``).
"""

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pysampling.algorithms.lhs import LatinHypercubeSampling
from pysurrogate.core.sampling import Sampling
from pysurrogate.optimizer import Adam

from pysamoo.experimental.acquisition import EI, AcquisitionProblem, EIProblem, LogEI


def _seed_pool(sampling, problem, n, random_state=None):
    """Draw a seed pool in the problem's box with a pysampling ``Sampling`` object.

    The ``Sampling`` instance is the pluggable hook: the default
    ``LatinHypercubeSampling(criterion=None)`` is a single plain Latin-hypercube draw in
    ``[0, 1]^d`` -- cheap, because these are only *starting points* for the optimizer and
    do not need an optimized space-filling layout. A maximin-optimized draw
    (``LatinHypercubeSampling(criterion="maxmin")``, the pysampling default) re-runs a 20-sweep
    swap search every call, which dominated the BO runtime for no quality gain; pass it (or
    ``RieszEnergySampling``) only when seed spacing specifically matters. The unit draw is
    scaled to the box.

    Args:
        sampling: A pysampling ``Sampling`` instance producing points in ``[0, 1]^d``.
        problem: The problem, used for its dimensionality and box bounds.
        n: Number of seed points to draw.
        random_state: Optional seed/generator threaded into the draw for reproducibility;
            set on the sampler so successive infills advance a shared generator.

    Returns:
        The seed points, shape ``(n, d)``, within ``[xl, xu]``.
    """
    xl, xu = problem.bounds()
    sampling.random_state = random_state
    unit = sampling.sample(n, problem.n_var)
    return xl + unit * (xu - xl)


def _seed_from(random_state):
    """Coerce a pymoo ``random_state`` (int, numpy ``Generator``, or ``None``) to an int seed."""
    if random_state is None:
        return 0
    if isinstance(random_state, (int, np.integer)):
        return int(random_state)
    # a numpy Generator / RandomState: draw a reproducible integer seed from it
    return int(random_state.integers(0, 2**31 - 1))


class Optimizer:
    """Base class for acquisition optimizers: maximize an acquisition over the box."""

    def optimize(self, problem, model, acq_func, f_min, elites=None, random_state=None):
        """Choose the next point by maximizing the acquisition over the problem's box.

        Args:
            problem: The (real) problem, used only for its box bounds.
            model: The surrogate, exposing ``predict`` (and ``predict(grad=True)`` for the
                gradient strategies).
            acq_func: The acquisition function (e.g. ``EI()``).
            f_min: The incumbent target the acquisition improves over.
            elites: Optional best evaluated points ``(k, d)``; the search adds a small *perturbed*
                cloud around them to its seeds, so it starts in the (well-sampled, low-uncertainty)
                basin where the EI peak lives but a global draw never lands. ``None`` seeds globally
                only.
            random_state: Optional generator threaded into any sampling.

        Returns:
            Tuple ``(x_best, acq_value)`` -- the chosen point ``(d,)`` and the
            acquisition value there in minimization convention (``-EI`` for EI).
        """
        raise NotImplementedError


class VectorizedGradientDescent(Optimizer):
    """Maximize (log)EI with pysurrogate's population :class:`~pysurrogate.optimizer.Adam`.

    Builds an :class:`~pysamoo.experimental.acquisition.EIProblem` (``-log EI`` over the unit cube,
    with the analytic gradient from the surrogate) and hands it to pysurrogate's generic Adam --
    the same optimizer layer that fits the kriging hyper-parameters. A population of ``pop_size``
    points climbs EI together via one batched ``predict(grad=True)`` per step.

    EI is multimodal (a sharp exploit peak in the well-sampled incumbent basin plus broad explore
    bumps far from data), so the population must be *seeded* in the promising basins or Adam gets
    lost on a shallow far-away gradient. The seeding is a cheap screen: draw ``n_pool`` points from
    a plain pysampling LHS (``criterion=None`` -- no maximin, so it stays fast), **plus a small
    perturbed cloud around the best evaluated points** (``elites``) -- because the EI peak sits next
    to the incumbent, in a basin so small a global draw never lands there (seeding the incumbent
    *exactly* is useless: sigma=0 there so EI and its gradient vanish, hence the perturbation). The
    combined pool is ranked by a value-only ``EIProblem.screen`` and Adam starts from the best
    ``pop_size``. No gradient is spent on the pool, only on the population that climbs.

    Args:
        pop_size: Number of points climbing EI in parallel (the Adam population).
        n_iter: Number of Adam steps.
        lr: Adam learning rate, a fraction of each dimension's box width (the EIProblem searches
            the unit cube, so the rate is scale-free).
        n_pool: Size of the plain-LHS global screen pool the starts are picked from.
        n_elite: Number of best evaluated points to seed a local cloud around.
        n_local: Points per elite cloud (split across a few perturbation scales).
        sampling: pysampling ``Sampling`` for the global pool (default
            ``LatinHypercubeSampling(criterion=None)``, a cheap plain draw).
    """

    # unit-cube std-devs of the local clouds: multi-scale so at least one band lands in the basin
    # whatever its size (the improvement basin shrinks as the incumbent improves).
    _LOCAL_SCALES = (0.01, 0.03, 0.1)

    def __init__(self, pop_size=32, n_iter=30, lr=0.1, n_pool=256, n_elite=3, n_local=30, sampling=None):
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.lr = lr
        self.n_pool = n_pool
        self.n_elite = n_elite
        self.n_local = n_local
        self.sampling = sampling if sampling is not None else LatinHypercubeSampling(criterion=None)

    def optimize(self, problem, model, acq_func, f_min, elites=None, random_state=None):
        """Maximize (log)EI with screened-seed pysurrogate Adam; see :meth:`Optimizer.optimize`."""
        if not isinstance(acq_func, (EI, LogEI)):
            raise TypeError("VectorizedGradientDescent supports only the EI / LogEI acquisition")
        xl, xu = problem.bounds()
        span = np.where(xu - xl == 0.0, 1.0, xu - xl)
        ei = EIProblem(model, f_min, xl, xu, log=isinstance(acq_func, LogEI))

        # plain LHS pool in the unit cube (no maximin), scored by the CURRENT model's value-only
        # (log)EI -- recomputed every infill since the surrogate (and so EI) changes each time.
        self.sampling.random_state = random_state
        pool = self.sampling.sample(self.n_pool, problem.n_var)

        # local clouds around the best evaluated points: the EI peak is in the (densely sampled)
        # incumbent basin, which the global LHS misses; perturbed seeds put Adam inside it.
        if elites is not None and self.n_elite > 0 and self.n_local > 0:
            rng = np.random.default_rng(_seed_from(random_state))
            ue = np.clip((np.atleast_2d(elites)[: self.n_elite] - xl) / span, 0.0, 1.0)
            per = max(1, self.n_local // len(self._LOCAL_SCALES))
            clouds = [
                np.clip(u + rng.normal(0.0, s, (per, problem.n_var)), 0.0, 1.0) for u in ue for s in self._LOCAL_SCALES
            ]
            pool = np.vstack([pool, *clouds])

        starts = pool[np.argsort(ei.screen(pool))[: self.pop_size]]
        seed = Sampling(self.pop_size, include=list(starts))
        adam = Adam(steps=self.n_iter, lr=self.lr, sampling=seed, random_state=_seed_from(random_state))
        res = adam.minimize(ei)
        u_best = res.x if res.x is not None else starts[0]
        return ei.to_x(u_best)[0], float(res.f)


class GeneticAlgorithm(Optimizer):
    """Niching genetic algorithm over a plain Latin-hypercube seed.

    Derivative-free and batched; works with any surrogate and any acquisition. The fallback for
    a non-EI acquisition (POI, UCB), where the gradient strategy does not apply -- at the cost of
    not pinpointing sharp optima as precisely as the gradient path.

    Args:
        pop_size: Niching-GA population size.
        n_sample: Size of the Latin-hypercube seed the GA is initialized from.
        sampling: pysampling ``Sampling`` instance for the seed pool (default
            ``LatinHypercubeSampling(criterion=None)``, a cheap plain draw); the pluggable hook.
    """

    def __init__(self, pop_size=50, n_sample=500, sampling=None):
        self.pop_size = pop_size
        self.n_sample = n_sample
        self.sampling = sampling if sampling is not None else LatinHypercubeSampling(criterion=None)

    def optimize(self, problem, model, acq_func, f_min, elites=None, random_state=None):
        """Maximize the acquisition with a niching GA; see :meth:`Optimizer.optimize`."""
        acq = AcquisitionProblem(problem, model, acq_func, f_min=f_min)
        seeds = _seed_pool(self.sampling, problem, self.n_sample, random_state=random_state)
        if elites is not None:
            # let the niching GA also start from the best evaluated points
            seeds = np.vstack([np.atleast_2d(elites), seeds])
        algorithm = NicheGA(pop_size=self.pop_size, sampling=seeds)
        res = pymoo_minimize(acq, algorithm, DefaultSingleObjectiveTermination(period=1), verbose=False)
        return res.opt.get("X")[0], float(res.opt.get("F")[0, 0])
