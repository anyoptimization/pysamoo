from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.problems.single import Ackley

from pysamoo.algorithms.psaf import PSAF

if __name__ == "__main__":

    problem = Ackley(n_var=10)

    algorithm = DE(pop_size=20, n_offsprings=10)

    algorithm = PSAF(algorithm, alpha=10, beta=30)

    res = minimize(
        problem,
        algorithm,
        ('n_evals', 300),
        seed=1,
        verbose=True)

    print("Best solution found: \nX = %s\nF = %s\nCV=%s" % (res.X, res.F, res.CV))
