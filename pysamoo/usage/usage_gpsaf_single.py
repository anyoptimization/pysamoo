from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems.single import Ackley
from pysamoo.algorithms.gpsaf import GPSAF


problem = Ackley(n_var=10)

algorithm = GA(pop_size=20, n_offsprings=10)

algorithm = GPSAF(algorithm, n_initial_doe=30, alpha=10, beta=30, n_max_infills=10, n_max_doe=500)

res = minimize(
    problem,
    algorithm,
    ('n_evals', 300),
    seed=2,
    verbose=True)

print("Best solution found: \nX = %s\nF = %s\nCV=%s" % (res.X, res.F, res.CV))
