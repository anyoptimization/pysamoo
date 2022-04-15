from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pysamoo.algorithms.gpsaf import GPSAF

problem = get_problem("g1")

algorithm = ISRES()

algorithm = GPSAF(algorithm,
                  alpha=3,
                  beta=30,
                  n_max_infills=1)

res = minimize(
    problem,
    algorithm,
    ('n_evals', 50),
    seed=1,
    verbose=True)

print("Best solution found: \nX = %s\nF = %s\nCV=%s" % (res.X, res.F, res.CV))


