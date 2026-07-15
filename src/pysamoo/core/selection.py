"""Pluggable model-selection strategies.

A model-selection strategy is just a **Target factory**: a callable
``(label, models) -> Target`` that decides how a pool of candidate surrogates is
cross-validated and reduced to the one that is used.

Built-in strategies:

* ``"full"``   — :class:`~pysamoo.core.target.Target`: cross-validate the whole
  pool every iteration (most accurate, slowest).

Add a new strategy by registering another :class:`~pysamoo.core.target.Target`
subclass in :data:`STRATEGIES`, or pass any factory directly to an algorithm's
``selection`` argument. This is the single extension point — algorithms do
not hard-code which strategy they use.
"""

from pysamoo.core.target import Target

STRATEGIES = {
    "full": Target,
}


def resolve(strategy):
    """Resolve a ``selection`` value to a Target factory.

    Args:
        strategy: Either a registered name (a key of :data:`STRATEGIES`) or a
            callable ``(label, models) -> Target`` (e.g. a ``Target`` subclass or a
            ``lambda`` that pre-configures one).

    Returns:
        A callable ``(label, models) -> Target``.
    """
    if callable(strategy):
        return strategy
    if strategy in STRATEGIES:
        return STRATEGIES[strategy]
    raise ValueError(f"unknown selection {strategy!r}; choose from {sorted(STRATEGIES)} or pass a Target factory")
