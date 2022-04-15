from pymoo.optimize import minimize
from pymoo.problems.single import Sphere
from pysamoo.experimental.bo import BayesianOptimization

if __name__ == "__main__":
    problem = Sphere(n_var=10)

    algorithm = BayesianOptimization(model_selection=True, adaptive_fmin=True)

    res = minimize(problem,
                   algorithm,
                   ("n_gen", 150),
                   seed=1,
                   verbose=True)

    print("Best solution found: \nX = %s\nF = %s\nCV=%s" % (res.X, res.F, res.CV))


