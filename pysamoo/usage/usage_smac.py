from pymoo.optimize import minimize
from pymoo.problems.single import Sphere
from pysamoo.vendor.smac import SMAC

if __name__ == "__main__":

    problem = Sphere(n_var=10)

    algorithm = SMAC()

    res = minimize(
        problem,
        algorithm,
        ('n_evals', 60),
        seed=2,
        verbose=True)

    print("Best solution found: \nX = %s\nF = %s\nCV=%s" % (res.X, res.F, res.CV))


