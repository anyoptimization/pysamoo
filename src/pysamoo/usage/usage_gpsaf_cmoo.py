from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.problems.many import C3DTLZ4
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pysamoo.algorithms.gpsaf import GPSAF

if __name__ == "__main__":

    problem = C3DTLZ4()

    ref_dirs = get_reference_directions("das-dennis", 3, n_points=91)
    algorithm = NSGA3(ref_dirs)

    algorithm = GPSAF(algorithm,
                      alpha=2,
                      beta=30,
                      n_max_infills=10,
                      n_max_doe=200)

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
