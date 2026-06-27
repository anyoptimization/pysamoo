"""Test session setup: force a headless matplotlib backend.

pysamoo depends on ``pymoo>=0.6.1.5`` (see setup.py), which is compatible with
the installed numpy 2.x / matplotlib 3.11. Earlier pymoo releases crashed on
``numpy.math`` and ``matplotlib.cm.get_cmap`` removals — see
docs/source/performance.rst for that history.
"""

import matplotlib

matplotlib.use("Agg")
