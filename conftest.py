"""Root pytest configuration.

Golden behavior-regression tests (``@pytest.mark.golden``) are enforced by a
pytest plugin. Two environments must both work:

* Local dev with pyclawd installed — pyclawd auto-registers its own golden plugin
  via an entry point, so we must NOT also register the vendored copy (double
  registration errors).
* CI / contributors without pyclawd — there is no entry-point plugin, so we load
  the self-contained vendored copy (``tests/_golden_plugin.py``) instead.

Regenerate the vendored plugin with ``pyclawd golden vendor tests/_golden_plugin.py``.
"""

import importlib.util
import os

_force_vendored = os.environ.get("PYSAMOO_FORCE_VENDORED_GOLDEN") == "1"

if _force_vendored or importlib.util.find_spec("pyclawd") is None:
    pytest_plugins = ["tests._golden_plugin"]
