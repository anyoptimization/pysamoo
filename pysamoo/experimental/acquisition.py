from scipy.stats import norm

# =========================================================================================================
# Acquisition Functions
# =========================================================================================================
from pymoo.problems.meta import MetaProblem


class AcquisitionFunction:

    def calc(self, mu, sigma, **kwargs):
        pass


class EI(AcquisitionFunction):

    def calc(self, mu, sigma, f_min=None, **kwargs):
        if f_min is None:
            raise Exception("Estimation of minimum function value needs to be provided!")

        f = - sigma

        # through precision error sigma can be negative - this should actually never be the case
        pos_sigma = sigma > 0
        mu, sigma = mu[pos_sigma], sigma[pos_sigma]

        # minimization version of EI
        impr = f_min - mu

        # calculate expected improvement
        z = impr / sigma

        ei = impr * norm.cdf(z) + sigma * norm.pdf(z)

        # because we are minimizing take the negative expected improvement
        f[pos_sigma] = - ei

        return f


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

        f[pos_sigma] = - pi

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


class AcquisitionProblem(MetaProblem):

    def __init__(self,
                 problem,
                 model,
                 acquisition_func,
                 **kwargs):
        super().__init__(problem)
        self.model = model
        self.acquisition_func = acquisition_func
        self.kwargs = kwargs

    def _evaluate(self, x, out, *args, **kwargs):
        if self.model is None:
            raise Exception("Please set the model for the problem to be defined.")

        # calculate the metric using the implementation
        mu, sigma = self.model.predict(x, return_values_of=["y", "sigma"])
        mu, sigma = mu[:, 0], sigma[:, 0]

        # calculate the value of the acquisition function
        acq = self.acquisition_func.calc(mu, sigma, x=x, **self.kwargs)

        out["F"], out["acq"] = acq, acq
