from pymoo.optimize import minimize
from pymoo.problems.single import Sphere

from pysamoo.experimental.bo import BayesianOptimization

if __name__ == "__main__":
    problem = Sphere(n_var=10)

    # one persistent DACE (pysurrogate) surrogate: cold-fit on the initial design,
    # then warm-refit only the newly evaluated point each generation.
    algorithm = BayesianOptimization()

    # 50 sequential GP infills is already a generous Bayesian-optimization budget;
    # it also keeps the archive small, capping the O(n^3) GP fit cost.
    res = minimize(problem, algorithm, ("n_gen", 50), seed=1, verbose=True)

    print("Best solution found: \nX = %s\nF = %s\nCV=%s" % (res.X, res.F, res.CV))
