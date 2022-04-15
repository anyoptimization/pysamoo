import matplotlib.pyplot as plt
import numpy as np

from pymoo.problems.multi import SRN
from pysamoo.sampling.energy import EnergyConstrainedSampling

problem = SRN()

n_points = 50


def func_constr(X):
    G = problem.evaluate(X, return_values_of=["G"])
    return np.maximum(G, 0.0).sum(axis=1)


X = EnergyConstrainedSampling(func_constr).do(problem, n_points).get("X")

xl, xu = problem.bounds()


def circle(x=0, y=0, r=1):
    theta = np.linspace(0, 2 * np.pi, 100)
    return x + r * np.cos(theta), y + r * np.sin(theta)


fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')

x, y = circle(r=15)
ax.plot(x, y, color="black", alpha=0.6)

x = np.linspace(-20, 20)
y = 1 / 3 * x + 10 / 3
ax.plot(x, y, color="black", alpha=0.6)

ax.set_aspect(1)

ax.set_xlim(xl[0], xu[0])
ax.set_ylim(xl[1], xu[1])

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")

plt.show()
