from pymoo.optimize import minimize
from pymoo.problems.single import Sphere
from pysamoo.experimental.bo import BayesianOptimization

if __name__ == "__main__":
    problem = Sphere(n_var=10)

    # model_selection=False avoids re-running k-fold CV over the full Kriging
    # hyperparameter grid every generation (~121 model fits/gen), which is the
    # dominant cost of Bayesian optimization here. See docs/PERFORMANCE.md.
    algorithm = BayesianOptimization(model_selection=False, adaptive_fmin=True)

    # 50 sequential GP infills is already a generous Bayesian-optimization budget;
    # it also keeps the archive small, capping the O(n^3) GP fit cost.
    res = minimize(problem,
                   algorithm,
                   ("n_gen", 50),
                   seed=1,
                   verbose=True)

    print("Best solution found: \nX = %s\nF = %s\nCV=%s" % (res.X, res.F, res.CV))


