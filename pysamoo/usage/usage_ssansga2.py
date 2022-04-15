from pymoo.optimize import minimize
from pymoo.problems.multi.zdt import ZDT2
from pymoo.visualization.scatter import Scatter
from pysamoo.algorithms.ssansga2 import SSANSGA2

problem = ZDT2(n_var=10)

algorithm = SSANSGA2(n_initial_doe=50,
                     n_infills=10,
                     surr_pop_size=100,
                     surr_n_gen=50)

res = minimize(
    problem,
    algorithm,
    ('n_evals', 150),
    seed=1,
    verbose=True)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()


