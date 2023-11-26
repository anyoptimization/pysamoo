from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

from pysamoo.algorithms.gpsaf import GPSAF

problem = get_problem("dtlz2", n_var=10)

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=5)
algorithm = NSGA3(ref_dirs, n_offsprings=10)

algorithm = GPSAF(algorithm, alpha=5, beta=30)

res = minimize(
    problem,
    algorithm,
    ('n_evals', 300),
    seed=1,
    verbose=True)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()

