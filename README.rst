pysamoo - Surrogate-Assisted Multi-objective Optimization
====================================================================


|python| |license|


.. |python| image:: https://img.shields.io/badge/python-3.6-blue.svg
   :alt: python 3.6

.. |license| image:: https://img.shields.io/badge/license-apache-orange.svg
   :alt: license apache
   :target: https://www.apache.org/licenses/LICENSE-2.0

The software documentation is available here: https://anyoptimization.com/projects/pysamoo/

Installation
====================================================================

The official release is always available at PyPi:

.. code:: bash

    pip install -U pysamoo



.. _Usage:

Usage
********************************************************************************

We refer here to our documentation for all the details.
However, for instance, executing NSGA2:

.. code:: python

    from pymoo.optimize import minimize
    from pymoo.problems.multi.zdt import ZDT1
    from pymoo.visualization.scatter import Scatter
    from pysamoo.algorithms.ssansga2 import SSANSGA2

    problem = ZDT1(n_var=10)

    algorithm = SSANSGA2(n_initial_doe=50,
                         n_infills=10,
                         surr_pop_size=100,
                         surr_n_gen=50)

    res = minimize(
        problem,
        algorithm,
        ('n_evals', 200),
        seed=1,
        verbose=True)

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()



.. _Citation:

Citation
********************************************************************************

If you use this framework, we kindly ask you to cite the following paper:

| `Julian Blank, & Kalyanmoy Deb. (2022). pysamoo: Surrogate-Assisted Multi-Objective Optimization in Python. <https://arxiv.org/abs/2204.05855>`_
|
| BibTex:

::

    @misc{pysamoo,
      title={pysamoo: Surrogate-Assisted Multi-Objective Optimization in Python},
      author={Julian Blank and Kalyanmoy Deb},
      year={2022},
      eprint={2204.05855},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
    }

.. _Contact:

Contact
********************************************************************************

Feel free to contact me if you have any questions:

| `Julian Blank <http://julianblank.com>`_  (blankjul [at] msu.edu)
| Michigan State University
| Computational Optimization and Innovation Laboratory (COIN)
| East Lansing, MI 48824, USA

