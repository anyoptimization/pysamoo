import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1
from pymoo.visualization.scatter import Scatter
from pysamoo.algorithms.gpsaf import GPSAF

problem = ZDT1(n_var=10)

algorithm = NSGA2(pop_size=20, n_offsprings=10)

algorithm = GPSAF(algorithm,
                  alpha=5,
                  beta=50,
                  n_max_doe=100,
                  n_max_infills=np.inf,
                  )

res = minimize(
    problem,
    algorithm,
    ('n_evals', 200),
    seed=2,
    verbose=True)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()

