import numpy as np
from pymoo.core.meta import Meta
from pysurrogate.core.optimizer import Evaluation, Problem
from scipy.special import erfcx, log_ndtr, ndtr
from scipy.stats import norm

_SQRT_2 = np.sqrt(2.0)
_SQRT_2PI = np.sqrt(2.0 * np.pi)
_SQRT_HALF_PI = np.sqrt(np.pi / 2.0)
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _log_h(z):
    """Numerically stable ``log h(z)`` where ``h(z) = z*Phi(z) + phi(z)`` (so ``EI = sigma*h(z)``).

    ``h`` is the standardized expected improvement; it is positive but underflows to ``0`` in
    float64 once ``z`` is very negative (the incumbent is already good), which is what makes plain
    EI -- and its gradient -- vanish. This computes ``log h`` directly so it stays finite for any
    ``z``:

    - ``z > -1``: the direct ``log(z*Phi + phi)`` is accurate and free of underflow.
    - ``z <= -1``: the Mills-ratio form ``h = phi(z) * (1 + z * Phi(z)/phi(z))`` with
      ``Phi/phi = sqrt(pi/2)*erfcx(-z/sqrt2)`` -- ``erfcx`` is stable for large arguments, so the
      bracket is evaluated without the cancellation that kills the naive expression.

    Args:
        z: Standardized improvement ``(f_min - mu) / sigma``, any shape.

    Returns:
        ``log h(z)``, same shape as ``z``.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)

    upper = z > -1.0
    zu = z[upper]
    out[upper] = np.log(zu * ndtr(zu) + np.exp(-0.5 * zu**2) / _SQRT_2PI)

    zl = z[~upper]
    log_phi = -0.5 * zl**2 - _LOG_SQRT_2PI
    bracket = 1.0 + zl * _SQRT_HALF_PI * erfcx(-zl / _SQRT_2)
    out[~upper] = log_phi + np.log(np.clip(bracket, 1e-300, None))
    return out


# =========================================================================================================
# Acquisition Functions
# =========================================================================================================


class AcquisitionFunction:
    def calc(self, mu, sigma, **kwargs):
        pass


class EI(AcquisitionFunction):
    def calc(self, mu, sigma, f_min=None, **kwargs):
        if f_min is None:
            raise Exception("Estimation of minimum function value needs to be provided!")

        f = -sigma

        # through precision error sigma can be negative - this should actually never be the case
        pos_sigma = sigma > 0
        mu, sigma = mu[pos_sigma], sigma[pos_sigma]

        # minimization version of EI
        impr = f_min - mu

        # calculate expected improvement
        z = impr / sigma

        ei = impr * norm.cdf(z) + sigma * norm.pdf(z)

        # because we are minimizing take the negative expected improvement
        f[pos_sigma] = -ei

        return f


class LogEI(AcquisitionFunction):
    """Logarithm of Expected Improvement -- the numerically stable acquisition.

    Plain EI underflows to ``0`` (and so does its gradient) once the incumbent is good, because
    ``z = (f_min - mu)/sigma`` is very negative everywhere; a gradient optimizer then has no
    signal and the search stalls. ``LogEI`` returns ``log(EI) = log(sigma) + log h(z)`` computed
    in a stable closed form, which stays finite and keeps a usable gradient even when EI itself is
    ~1e-300 (Ament et al., NeurIPS 2023). The argmax is identical to EI's, since ``log`` is
    monotonic. ``calc`` returns ``-log(EI)`` to match the minimization convention.
    """

    def calc(self, mu, sigma, f_min=None, **kwargs):
        if f_min is None:
            raise Exception("Estimation of minimum function value needs to be provided!")
        sigma = np.maximum(sigma, 1e-300)
        z = (f_min - mu) / sigma
        return -(np.log(sigma) + _log_h(z))


class POI(AcquisitionFunction):
    def calc(self, mu, sigma, f_min=None, **kwargs):
        if f_min is None:
            raise Exception("Estimation of minimum function value needs to be provided!")

        pos_sigma = sigma > 0
        mu, sigma = mu[pos_sigma], sigma[pos_sigma]
        f = sigma

        # minimization version of PI
        impr = f_min - mu

        # calculate pi
        z = impr / sigma
        pi = norm.pdf(z)

        f[pos_sigma] = -pi

        return f


class UCB(AcquisitionFunction):
    def __init__(self, beta=3.0) -> None:
        super().__init__()
        self.beta = beta

    def calc(self, mu, sigma, **kwargs):
        ucb = mu - self.beta * sigma
        return ucb


# =========================================================================================================
# Acquisition Problem
# =========================================================================================================


class AcquisitionProblem(Meta):
    def __init__(self, problem, model, acquisition_func, **kwargs):
        super().__init__(problem)
        self.model = model
        self.acquisition_func = acquisition_func
        self.kwargs = kwargs

    def _evaluate(self, x, out, *args, **kwargs):
        if self.model is None:
            raise Exception("Please set the model for the problem to be defined.")

        # calculate the metric using the implementation (Dace returns a Prediction)
        pred = self.model.predict(x, var=True)
        mu, sigma = pred.y[:, 0], pred.sigma[:, 0]

        # calculate the value of the acquisition function
        acq = self.acquisition_func.calc(mu, sigma, x=x, **self.kwargs)

        out["F"], out["acq"] = acq, acq


# =========================================================================================================
# Expected Improvement as a pysurrogate optimization Problem
# =========================================================================================================


class EIProblem(Problem):
    """(Log) Expected-Improvement maximization as a pysurrogate minimization ``Problem``.

    Lets a generic ``pysurrogate.optimizer`` strategy (``Adam``, ``LBFGS``, ``Restart``) maximize
    EI over the box -- the same layer pysurrogate uses to fit theta -- instead of a bespoke
    acquisition optimizer. EI is maximized by *minimizing* ``-EI`` (or ``-log EI``); the analytic
    gradient comes from the surrogate's mean/variance gradients (``predict(mse=True, grad=True)``),
    so the search is gradient-driven and needs no seed pool.

    With ``log=True`` (the default) it optimizes ``log EI`` instead of ``EI``. They share an
    argmax (``log`` is monotonic), but ``log EI`` does not underflow to ``0`` once the incumbent is
    good, so its gradient stays finite and the search keeps converging instead of stalling.

    The search runs in the unit cube ``[0, 1]^d`` (``__call__`` maps each candidate back to the
    real box) so an optimizer's step size is a fraction of each dimension's width, independent of
    the problem's actual scale.

    Args:
        model: The surrogate exposing ``predict(mse=True, grad=True)`` (a Dace model).
        f_min: The incumbent objective EI improves over (the best observed value).
        xl: Lower bounds of the real problem box, shape ``(d,)``.
        xu: Upper bounds of the real problem box, shape ``(d,)``.
        log: Optimize ``log EI`` (stable, default) rather than ``EI``.
    """

    def __init__(self, model, f_min, xl, xu, log=True):
        self.model = model
        self.f_min = f_min
        self.log = log
        self._xl = np.asarray(xl, dtype=float)
        self._span = np.asarray(xu, dtype=float) - self._xl

    @property
    def bounds(self):
        """The unit-cube search box ``(0, 1)^d`` the optimizer works in."""
        p = len(self._xl)
        return np.zeros(p), np.ones(p)

    def to_x(self, U):
        """Map unit-cube candidates ``U`` back to the real problem box."""
        return self._xl + np.atleast_2d(U) * self._span

    def screen(self, U):
        """Cheap value-only ``-(log)EI`` for ranking a seed pool (no gradient computed)."""
        X = self.to_x(U)
        pred = self.model.predict(X, var=True)
        mu = pred.y[:, 0]
        sigma = np.maximum(pred.sigma[:, 0], 1e-300 if self.log else 1e-12)
        z = (self.f_min - mu) / sigma
        if self.log:
            return -(np.log(sigma) + _log_h(z))
        return -((self.f_min - mu) * norm.cdf(z) + sigma * norm.pdf(z))

    def __call__(self, U):
        """Return ``-(log)EI`` and its gradient (w.r.t. the unit cube) at the candidates ``U``."""
        X = self.to_x(U)
        pred = self.model.predict(X, var=True, grad=True)
        mu = pred.y[:, 0]
        sigma = np.maximum(pred.sigma[:, 0], 1e-300 if self.log else 1e-12)

        z = (self.f_min - mu) / sigma
        # chain-rule pieces shared by both forms: dmu = pred.grad, dsigma = grad(var)/(2 sigma)
        dmu = pred.grad
        dsigma = pred.var_grad / (2.0 * sigma[:, None])

        if self.log:
            # log EI = log(sigma) + log h(z); h'(z) = Phi(z), so d logEI/dx =
            # (1/sigma)[ -r dmu + dsigma (1 - z r) ] with r = Phi(z)/h(z) (computed in log-space,
            # so it stays finite where EI itself underflows).
            log_h = _log_h(z)
            r = np.exp(log_ndtr(z) - log_h)
            f = -(np.log(sigma) + log_h)
            d_log_ei = (-r[:, None] * dmu + dsigma * (1.0 - z * r)[:, None]) / sigma[:, None]
            grad_x = -d_log_ei
        else:
            Phi, phi = norm.cdf(z), norm.pdf(z)
            ei = (self.f_min - mu) * Phi + sigma * phi
            f = -ei
            # gradient of -EI w.r.t. x: -(-Phi*dmu + phi*dsigma)
            grad_x = Phi[:, None] * dmu - phi[:, None] * dsigma

        grad = grad_x * self._span[None, :]
        feasible = np.isfinite(f)
        return Evaluation(f=np.where(feasible, f, np.inf), feasible=feasible, grad=grad)
