import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.util.normalization import normalize, denormalize
from pymoo.util.ref_dirs.energy import squared_dist, calc_potential_energy_with_grad
from pymoo.util.ref_dirs.optimizer import Adam
from pysamoo.sampling.niching import NichingConstrainedSampling
from pysamoo.sampling.rejection import RejectionConstrainedSampling


def calc_potential_energy(A, d):
    i, j = np.triu_indices(len(A), 1)
    D = np.sqrt(squared_dist(A, A)[i, j])
    energy = np.log((1 / D ** d).mean())
    return energy


class EnergyConstrainedSampling(Sampling):

    def __init__(self,
                 func_eval_constr,
                 n_max_iter=10000):
        super().__init__()
        self.func_eval_constr = func_eval_constr
        self.n_max_iter = n_max_iter

    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        constr = self.func_eval_constr
        d = problem.n_var ** 2

        X = RejectionConstrainedSampling(constr).do(problem, n_samples).get("X")
        if len(X) < n_samples:
            X = NichingConstrainedSampling(constr).do(problem, n_samples).get("X")

        if len(X) == 0:
            raise Exception("No feasible solution could be found!")
        elif len(X) < n_samples:
            print("WARNING: Less feasible solutions than requested could be found.")

        X = normalize(X, xl, xu)

        optimizer = Adam(alpha=0.005)

        obj, grad = calc_potential_energy_with_grad(X, d)
        hist = [obj]

        done = False

        for i in range(self.n_max_iter):

            if done:
                break

            _X = optimizer.next(X, grad)
            _CV = constr(denormalize(_X, xl, xu))
            feasible = np.logical_and(_CV <= 0, np.all(np.logical_and(_X >= 0, _X <= 1), axis=1))

            X[feasible] = _X[feasible]

            obj, grad = calc_potential_energy_with_grad(X, d)
            hist.append(obj)

            hist = hist[-100:]

            avg_impr = (- np.diff(hist[-100:])).mean()

            if len(hist) > 100:
                if avg_impr < 1e-3:
                    optimizer = Adam(alpha=optimizer.alpha / 2)
                elif avg_impr < 1e-6:
                    done = True

        X = denormalize(X, xl, xu)

        return X
