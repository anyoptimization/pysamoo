from pymoo.optimize import minimize
from pymoo.problems.single import Sphere
from pysamoo.vendor.lqcmaes import lqCMAES

if __name__ == "__main__":

    problem = Sphere(n_var=10)

    algorithm = lqCMAES()

    res = minimize(
        problem,
        algorithm,
        ('n_evals', 200),
        seed=2,
        verbose=True)

    print("Best solution found: \nX = %s\nF = %s\nCV=%s" % (res.X, res.F, res.CV))

