"""Pluggable infill strategies: where Bayesian optimization samples next (global / local / hybrid)."""

import numpy as np

from pysamoo.experimental.optimizer import VectorizedGradientDescent


class Infill:
    """Decide the next point to evaluate, given the archive and the fitted surrogate.

    The single seam between the BO loop and *how* the next point is chosen. A strategy reads the
    evaluated points ``X``/``F`` (and, if it wants, the surrogate ``model``) and returns the next
    point plus a scalar acquisition value (for display). Concrete strategies: :class:`GlobalEI`
    (explore the whole box), :class:`LocalQuadratic` (refine locally), :class:`Hybrid` (switch).
    """

    def do(self, problem, get_model, X, F, acq_func, random_state=None):
        """Return ``(x, acq_value)`` -- the next point and its acquisition value (minimization).

        Args:
            problem: The problem (used for its box bounds and dimensionality).
            get_model: Zero-arg callable returning the fitted surrogate, fitting it lazily on first
                call. Global strategies call it; purely local ones do not -- so the (expensive) GP
                fit is skipped entirely on local steps.
            X: Evaluated inputs, shape ``(n, d)``.
            F: Evaluated objective values, shape ``(n,)``.
            acq_func: The acquisition function (e.g. ``LogEI()``).
            random_state: Optional generator threaded into any sampling.

        Returns:
            Tuple ``(x, acq_value)`` -- the chosen point ``(d,)`` and its value (lower is better).
        """
        raise NotImplementedError


class GlobalEI(Infill):
    """Global exploration: maximize the acquisition over the whole box (the standard BO step).

    Thin wrapper around an acquisition :class:`~pysamoo.experimental.optimizer.Optimizer` that
    supplies the incumbent ``f_min`` and the best-evaluated ``elites`` (for basin-local seeding).

    Args:
        optimizer: The acquisition optimizer (default ``VectorizedGradientDescent``).
        n_elite: How many best points to pass as seeding elites.
    """

    def __init__(self, optimizer=None, n_elite=5):
        self.optimizer = optimizer if optimizer is not None else VectorizedGradientDescent()
        self.n_elite = n_elite

    def do(self, problem, get_model, X, F, acq_func, random_state=None):
        """Maximize the acquisition globally; see :meth:`Infill.do`."""
        model = get_model()  # global EI needs the surrogate -> fit it
        f_min = float(F.min())
        elites = X[np.argsort(F)[: self.n_elite]]
        return self.optimizer.optimize(problem, model, acq_func, f_min, elites=elites, random_state=random_state)


def _quadratic_step(X, F, xc, f_min, lo, hi):
    """Fit a quadratic to ``(X, F)`` around ``xc``; return its LM-Newton minimizer in ``[lo, hi]``.

    Least-squares fits ``f ~ a + g.s + 0.5 s'H s`` (``s = x - xc``) to the given points, then steps
    to the model minimizer ``-H^-1 g`` with Levenberg-Marquardt regularization so an indefinite or
    near-singular ``H`` still yields a descent step, clipped to the trust-region box.

    The model **degree falls back with the point budget** -- a full quadratic has
    ``1 + d + d(d+1)/2`` (~d**2/2) coefficients, which a small archive cannot determine. When fewer
    points than that are available it drops to a **diagonal** quadratic (squared terms only,
    ``2d+1`` coefficients): still curvature-aware (so it keeps the Newton behavior, unlike a linear
    model) and exact on separable objectives, just without cross-terms. It never drops below
    quadratic.

    Args:
        X: Nearby evaluated inputs, shape ``(m, d)``.
        F: Their objective values, shape ``(m,)``.
        xc: Center of the local model (the incumbent), shape ``(d,)``.
        f_min: Objective at ``xc`` (for the predicted-decrease return).
        lo: Lower bound of the trust-region box, shape ``(d,)``.
        hi: Upper bound of the trust-region box, shape ``(d,)``.

    Returns:
        Tuple ``(x_star, predicted_decrease)`` -- the step target and ``f_min - model(x_star)``.
    """
    d = X.shape[1]
    diff = X - xc
    # full quadratic when the points can determine it; otherwise a diagonal quadratic (curvature
    # without cross-terms, 2d+1 dof) -- never below quadratic.
    full_pairs = [(i, j) for i in range(d) for j in range(i, d)]
    pairs = full_pairs if len(X) >= 1 + d + len(full_pairs) else [(i, i) for i in range(d)]
    design = np.column_stack(
        [np.ones(len(X))] + [diff[:, i] for i in range(d)] + [diff[:, i] * diff[:, j] for i, j in pairs]
    )
    coef, *_ = np.linalg.lstsq(design, F, rcond=None)
    g = coef[1 : 1 + d]
    hess = np.zeros((d, d))
    for k, (i, j) in enumerate(pairs):
        v = coef[1 + d + k]
        if i == j:
            hess[i, i] = 2.0 * v
        else:
            hess[i, j] = hess[j, i] = v
    # robust LM-Newton: nan-guard a degenerate lstsq fit, and use lstsq (pseudo-inverse) rather than
    # solve so a singular/ill-conditioned Hessian never raises (seen on Griewank/Zakharov).
    g, hess = np.nan_to_num(g), np.nan_to_num(hess)
    lam = max(0.0, -np.linalg.eigvalsh(hess).min()) + 1e-6
    s = np.linalg.lstsq(hess + lam * np.eye(d), -g, rcond=None)[0]
    x_star = np.clip(xc + s, lo, hi)
    ds = x_star - xc
    model_val = coef[0] + g @ ds + 0.5 * ds @ hess @ ds
    return x_star, f_min - model_val


class LocalQuadratic(Infill):
    """Local refinement: fit a quadratic to nearby points and take an LM-Newton trust-region step.

    Independent of the surrogate -- it approximates the objective *directly* from the nearest
    evaluated points (a quadratic least-squares fit), which is exactly the dense local data the
    biased BO sample provides near the optimum. It steps to that model's minimizer clipped to a box
    of side ``L*span`` around the incumbent, and adapts ``L`` by the standard trust-region ratio
    ``rho = actual decrease / predicted decrease``: shrink when the step under-delivers, grow when it
    over-delivers.

    Args:
        L0: Initial trust-region side as a fraction of the box width.
        L_bounds: ``(min, max)`` clamp on the trust-region side.
        eta: ``rho`` below this shrinks the region (a poor step).
        eta_grow: ``rho`` above this grows it (a good step).
        shrink: Multiplicative shrink factor.
        expand: Multiplicative grow factor.
    """

    def __init__(self, L0=0.1, L_bounds=(1e-4, 0.5), eta=0.1, eta_grow=0.5, shrink=0.5, expand=2.0):
        self.L0 = L0
        self.L_bounds = L_bounds
        self.eta = eta
        self.eta_grow = eta_grow
        self.shrink = shrink
        self.expand = expand
        self.reset()

    def reset(self):
        """Reset the trust region to its initial size and forget the last step (on a fresh local phase)."""
        self._L = self.L0
        self._center = None
        self._f_center = None
        self._pred_dec = None

    def do(self, problem, get_model, X, F, acq_func, random_state=None):
        """Take one local quadratic-Newton trust-region step; see :meth:`Infill.do` (ignores the surrogate)."""
        xl, xu = problem.bounds()
        span = xu - xl
        d = problem.n_var
        i = int(F.argmin())
        xc, f_min = X[i], float(F[i])

        # adapt L from how the PREVIOUS local step actually performed vs. its prediction
        if self._center is not None and self._pred_dec is not None:
            rho = (self._f_center - f_min) / self._pred_dec if self._pred_dec > 1e-300 else -1.0
            lo_b, hi_b = self.L_bounds
            if rho < self.eta:
                self._L = max(self._L * self.shrink, lo_b)
            elif rho > self.eta_grow:
                self._L = min(self._L * self.expand, hi_b)

        half = 0.5 * self._L * span
        lo, hi = np.maximum(xc - half, xl), np.minimum(xc + half, xu)
        dof = 1 + d + d * (d + 1) // 2  # quadratic degrees of freedom
        inside = np.where(np.all((X >= lo) & (X <= hi), axis=1))[0]
        idx = inside if len(inside) >= dof + 1 else np.argsort(((X - xc) ** 2).sum(1))[: dof + 1]

        x, pred_dec = _quadratic_step(X[idx], F[idx], xc, f_min, lo, hi)
        self._center, self._f_center, self._pred_dec = xc, f_min, pred_dec
        return x, -pred_dec


class Hybrid(Infill):
    """Switch between :class:`GlobalEI` (explore) and :class:`LocalQuadratic` (refine) on a stall.

    Runs the global strategy until it fails to improve the incumbent for ``patience_global``
    consecutive infills (EI exploration is spent), then hands off to the local strategy; when the
    local strategy stalls for ``patience_local`` (its trust region has collapsed), it switches back
    to global to explore elsewhere. The mode + stall counter live here; each sub-strategy owns its
    own state (surrogate/EI for global, trust region for local).

    Args:
        global_strategy: The exploration strategy (default :class:`GlobalEI`).
        local_strategy: The refinement strategy (default :class:`LocalQuadratic`).
        patience_global: Consecutive non-improving global infills before switching to local.
        patience_local: Consecutive non-improving local infills before switching back to global.
    """

    def __init__(self, global_strategy=None, local_strategy=None, patience_global=4, patience_local=5):
        self.g = global_strategy if global_strategy is not None else GlobalEI()
        self.l = local_strategy if local_strategy is not None else LocalQuadratic()
        self.patience_global = patience_global
        self.patience_local = patience_local
        self.mode = "global"
        self.stall = 0
        self._prev_best = None

    def do(self, problem, get_model, X, F, acq_func, random_state=None):
        """Pick the active strategy (flipping on a stall) and delegate; see :meth:`Infill.do`."""
        best = float(F.min())
        if self._prev_best is not None:
            improved = best < self._prev_best - 1e-12 * max(1.0, abs(best))
            self.stall = 0 if improved else self.stall + 1
            patience = self.patience_global if self.mode == "global" else self.patience_local
            if self.stall >= patience:
                self.mode = "local" if self.mode == "global" else "global"
                self.stall = 0
                if self.mode == "local":
                    self.l.reset()
        self._prev_best = best

        active = self.g if self.mode == "global" else self.l
        return active.do(problem, get_model, X, F, acq_func, random_state)


def _weighted_pca(X, F, xc):
    """Weighted-PCA rotation of points around ``xc``, weighted toward better objective values.

    Computes the rotation ``R`` whose columns are the principal axes of the quality-weighted local
    point cloud -- the LABCAT idea: rotate so the (rotated) valley becomes axis-aligned, letting a
    cheap *diagonal* model capture an ill-conditioned, rotated valley with only ``2d+1`` parameters
    instead of a full quadratic's ``~d^2/2``.

    Args:
        X: Nearby evaluated inputs, shape ``(m, d)``.
        F: Their objective values, shape ``(m,)``.
        xc: Center (the incumbent), shape ``(d,)``.

    Returns:
        Orthonormal rotation ``R``, shape ``(d, d)`` (so ``z = R^T (x - xc)``).
    """
    spread = float(F.max() - F.min())
    w = np.exp(-(F - F.min()) / (spread + 1e-12)) if spread > 0 else np.ones(len(F))
    w = w / w.sum()
    diff = X - xc
    cov = (diff * w[:, None]).T @ diff
    cov = cov + 1e-12 * np.eye(X.shape[1])
    _, R = np.linalg.eigh(cov)
    return R


def _rotated_diag_step(X, F, xc, f_min, R, lo, hi):
    """Fit a diagonal quadratic in the PCA-rotated frame and return its LM-Newton minimizer.

    In the rotated frame ``z = R^T (x - xc)`` the model is ``f ~ a + g.z + 0.5 sum_j h_j z_j^2``
    (``2d+1`` coefficients): curvature-aware but cheap, and -- because the frame is aligned with the
    local principal axes -- able to represent a *rotated* ellipsoidal valley that an axis-aligned
    diagonal model cannot. The step is the Levenberg-Marquardt-regularized Newton minimizer, mapped
    back to the original space and clipped to the box.

    Args:
        X: Nearby evaluated inputs, shape ``(m, d)``.
        F: Their objective values, shape ``(m,)``.
        xc: Center (the incumbent), shape ``(d,)``.
        f_min: Objective at ``xc`` (for the predicted-decrease return).
        R: Rotation from :func:`_weighted_pca`, shape ``(d, d)``.
        lo: Lower bound of the trust-region box, shape ``(d,)``.
        hi: Upper bound of the trust-region box, shape ``(d,)``.

    Returns:
        Tuple ``(x_star, predicted_decrease)`` -- the step target and ``f_min - model(x_star)``.
    """
    d = X.shape[1]
    z = (X - xc) @ R
    design = np.column_stack([np.ones(len(z))] + [z[:, j] for j in range(d)] + [z[:, j] ** 2 for j in range(d)])
    coef, *_ = np.linalg.lstsq(design, F, rcond=None)
    g = np.nan_to_num(coef[1 : 1 + d])
    h = np.nan_to_num(2.0 * coef[1 + d : 1 + 2 * d])
    lam = max(0.0, -h.min()) + 1e-6
    s = -g / (h + lam)
    x_star = np.clip(xc + R @ s, lo, hi)
    s_eff = R.T @ (x_star - xc)
    model_val = coef[0] + g @ s_eff + 0.5 * np.sum(h * s_eff**2)
    return x_star, f_min - model_val


class LocalPCAQuadratic(Infill):
    """Local refinement for ROTATED, ill-conditioned valleys (LABCAT-style PCA rotation).

    Like :class:`LocalQuadratic` but fits its quadratic in the quality-weighted PCA frame of the
    nearby points, where a cheap *diagonal* quadratic (``2d+1`` dof) can model a rotated ellipsoidal
    valley -- so it fits far earlier than a full quadratic (``~d^2/2`` dof) and stays
    well-conditioned. Surrogate-free (never calls ``get_model``), so it is ~free. The trust region
    adapts by the standard ``rho`` ratio.

    Args:
        L0: Initial trust-region side as a fraction of the box width.
        L_bounds: ``(min, max)`` clamp on the trust-region side.
        eta: ``rho`` below this shrinks the region.
        eta_grow: ``rho`` above this grows it.
        shrink: Multiplicative shrink factor.
        expand: Multiplicative grow factor.
    """

    def __init__(self, L0=0.1, L_bounds=(1e-4, 0.5), eta=0.1, eta_grow=0.5, shrink=0.5, expand=2.0):
        self.L0 = L0
        self.L_bounds = L_bounds
        self.eta = eta
        self.eta_grow = eta_grow
        self.shrink = shrink
        self.expand = expand
        self.reset()

    def reset(self):
        """Reset the trust region and forget the last step (on a fresh local phase)."""
        self._L = self.L0
        self._center = None
        self._f_center = None
        self._pred_dec = None

    def do(self, problem, get_model, X, F, acq_func, random_state=None):
        """Take one PCA-rotated diagonal quadratic-Newton trust-region step; see :meth:`Infill.do`."""
        xl, xu = problem.bounds()
        span = xu - xl
        d = problem.n_var
        i = int(F.argmin())
        xc, f_min = X[i], float(F[i])

        if self._center is not None and self._pred_dec is not None:
            rho = (self._f_center - f_min) / self._pred_dec if self._pred_dec > 1e-300 else -1.0
            lo_b, hi_b = self.L_bounds
            if rho < self.eta:
                self._L = max(self._L * self.shrink, lo_b)
            elif rho > self.eta_grow:
                self._L = min(self._L * self.expand, hi_b)

        half = 0.5 * self._L * span
        lo, hi = np.maximum(xc - half, xl), np.minimum(xc + half, xu)
        dof = 2 * d + 1  # diagonal-quadratic degrees of freedom
        inside = np.where(np.all((X >= lo) & (X <= hi), axis=1))[0]
        idx = inside if len(inside) >= dof + 1 else np.argsort(((X - xc) ** 2).sum(1))[: dof + 1]

        R = _weighted_pca(X[idx], F[idx], xc)
        x, pred_dec = _rotated_diag_step(X[idx], F[idx], xc, f_min, R, lo, hi)
        self._center, self._f_center, self._pred_dec = xc, f_min, pred_dec
        return x, -pred_dec


class RestartingLocal(Infill):
    """Multi-start local search for MULTIMODAL functions (escape basins, e.g. Rastrigin).

    Runs a cheap local refiner until it stalls (no incumbent improvement for ``patience`` calls),
    then restarts it from a space-filling point chosen FAR from all evaluated points (farthest-point
    / max-min over a random candidate pool). Many cheap local dives at diverse locations find the
    global basin among exponentially many local minima. Surrogate-free.

    Args:
        local_strategy: The local refiner to restart (default :class:`LocalQuadratic`).
        patience: Consecutive non-improving calls before a restart.
        pool: Candidate pool size for the farthest-point restart pick.
    """

    def __init__(self, local_strategy=None, patience=4, pool=200):
        self.local = local_strategy if local_strategy is not None else LocalQuadratic()
        self.patience = patience
        self.pool = pool
        self.stall = 0
        self._prev_best = None

    def reset(self):
        """Reset the stall counter and the wrapped local refiner (on a fresh local phase)."""
        self.stall = 0
        self._prev_best = None
        self.local.reset()

    def do(self, problem, get_model, X, F, acq_func, random_state=None):
        """Refine locally, restarting from a far point on a stall; see :meth:`Infill.do`."""
        xl, xu = problem.bounds()
        rng = random_state if random_state is not None else np.random.default_rng(0)
        best = float(F.min())
        if self._prev_best is not None:
            improved = best < self._prev_best - 1e-12 * max(1.0, abs(best))
            self.stall = 0 if improved else self.stall + 1
        self._prev_best = best

        if self.stall >= self.patience:
            self.stall = 0
            self.local.reset()
            cand = xl + rng.random((self.pool, problem.n_var)) * (xu - xl)
            # farthest-point: maximize min distance to all evaluated points
            d2 = ((cand[:, None, :] - X[None, :, :]) ** 2).sum(-1).min(axis=1)
            x = cand[int(np.argmax(d2))]
            return x, 0.0
        return self.local.do(problem, get_model, X, F, acq_func, random_state)


class Explorer(Infill):
    """Pure space-filling exploration -- never exploits, just keeps covering the box.

    Each infill draws a fresh Latin-Hypercube batch that is spaced *away from the current
    population* via pysampling's ``LatinHypercubeSampling(Xp=...)`` -- the ``Xp`` argument folds the
    distance to the already-evaluated points into the maximin energy, so the new design fills the
    emptiest region ("LHS conditioned on ``Xp``"). It returns the most novel of the batch (largest
    nearest-neighbour gap). There is no incumbent, no acquisition, no surrogate call -- it is the
    exploration extreme (the opposite of :class:`LocalQuadratic`).

    Two uses. (1) A baseline / diversification component. (2) A *measurement* tool: because the
    design is unbiased space-filling (not the self-selected, clustered sample an EI loop produces),
    the surrogate's prequential refit score under Explorer is an honest learning curve for *how
    fast a model learns the function* -- the quantity a normal BO run confounds with where it chose
    to look (see ``findings.md`` §8).

    Args:
        batch: Number of LHS points drawn per infill (spaced against ``Xp``); the most novel is returned.
        criterion: pysampling LHS criterion (``"maxmin"`` optimizes spacing incl. distance to ``Xp``).
    """

    def __init__(self, batch=16, criterion="maxmin"):
        self.batch = batch
        self.criterion = criterion

    def do(self, problem, get_model, X, F, acq_func, random_state=None):
        """Return the most-novel point of an LHS batch drawn away from ``X`` (via ``Xp``); see :meth:`Infill.do`."""
        from pysampling.algorithms.lhs import LatinHypercubeSampling

        xl, xu = problem.bounds()
        span = xu - xl
        rng = random_state if random_state is not None else np.random.default_rng(0)
        # existing population in the unit cube -> pysampling samples the new batch AWAY from it
        Xp = (X - xl) / span
        U = LatinHypercubeSampling(criterion=self.criterion, Xp=Xp, random_state=rng).sample(self.batch, problem.n_var)
        cand = xl + U * span
        # return the candidate whose nearest already-evaluated neighbour is farthest away
        d2 = ((cand[:, None, :] - X[None, :, :]) ** 2).sum(-1).min(axis=1)
        j = int(np.argmax(d2))
        # acq value (lower = better, for display): negative fill distance -- bigger gap = more novel
        return cand[j], -float(np.sqrt(d2[j]))


class LocalCMAES(Infill):
    """Sequential CMA-ES local strategy: learns the metric (covariance ~ inverse Hessian).

    The derivative-free method of choice for ill-conditioned, non-separable, curved-valley problems
    (Rosenbrock, RotatedEllipsoid): it adapts a full covariance ``C`` so the sampling distribution
    aligns with the local valley, learning the conditioning/rotation a diagonal model cannot. Runs
    one sample per infill -- it draws ``x = mean + sigma * B D z`` (``z ~ N(0, I)``), and once a full
    generation of ``lambda`` points has been evaluated, applies the standard CMA-ES rank-mu / rank-one
    update (Hansen) to ``mean``, ``sigma``, and ``C``. Surrogate-free (never calls ``get_model``).

    Designed for *pure* use (every infill is its own); inside :class:`Hybrid` the generation
    bookkeeping (which archive rows are this generation's) is only approximate.

    Args:
        sigma0: Initial step size as a fraction of the mean box width.
    """

    def __init__(self, sigma0=0.3):
        self.sigma0 = sigma0
        self.reset()

    def reset(self):
        """Forget the CMA state so the next call re-initializes from the current incumbent."""
        self._init = False
        self._gen = 0
        self._returned = []

    def _setup(self, d, xl, xu, xc):
        """Initialize CMA-ES constants and state for dimension ``d`` centered at ``xc``."""
        self.d = d
        self.lam = 4 + int(3 * np.log(d))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w**2)
        self.cs = (self.mueff + 2) / (d + self.mueff + 5)
        self.ds = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (d + 1)) - 1) + self.cs
        self.cc = (4 + self.mueff / d) / (d + 4 + 2 * self.mueff / d)
        self.c1 = 2 / ((d + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((d + 2) ** 2 + self.mueff))
        self.chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        self.ps = np.zeros(d)
        self.pc = np.zeros(d)
        self.C = np.eye(d)
        self.B = np.eye(d)
        self.D = np.ones(d)
        self.mean = xc.copy().astype(float)
        self.sigma = self.sigma0 * float(np.mean(xu - xl))
        self._init = True

    def _update(self, Xg, Fg):
        """Apply the CMA-ES update from a finished generation's points ``Xg`` and values ``Fg``."""
        order = np.argsort(Fg)
        ys = (Xg - self.mean) / self.sigma
        ysel = ys[order[: self.mu]]
        yw = self.w @ ysel
        self.mean = self.mean + self.sigma * yw
        cinv = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (cinv @ yw)
        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1))
        denom = np.sqrt(1 - (1 - self.cs) ** (2 * (self._gen + 1)))
        hs = 1.0 if np.linalg.norm(self.ps) / denom < (1.4 + 2 / (self.d + 1)) * self.chiN else 0.0
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = sum(self.w[k] * np.outer(ysel[k], ysel[k]) for k in range(self.mu))
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
            + self.cmu * rank_mu
        )
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        vals, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.clip(vals, 1e-20, None))

    def do(self, problem, get_model, X, F, acq_func, random_state=None):
        """Draw the next CMA-ES sample, updating the distribution after each full generation."""
        xl, xu = problem.bounds()
        d = problem.n_var
        rng = random_state if random_state is not None else np.random.default_rng(0)
        if not self._init:
            self._setup(d, xl, xu, X[int(F.argmin())])
        # once a full generation of lambda points has been returned (and so evaluated -- they are the
        # last lambda archive rows in pure use), apply the CMA-ES update and start a new generation.
        if len(self._returned) == self.lam:
            self._update(np.array(self._returned), F[-self.lam :])
            self._returned = []
            self._gen += 1
        z = rng.standard_normal(d)
        x = np.clip(self.mean + self.sigma * (self.B @ (self.D * z)), xl, xu)
        self._returned.append(x)
        return x, 0.0
