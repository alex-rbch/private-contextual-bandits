"""Differentially private contextual bandits.

Layering, from the bottom up:

* ``privacy``     - mechanisms and accounting, with no knowledge of policies.
* ``oracles``     - the two batch-local oracles, ``quadratic`` and
                    ``logistic``, each fitting only the batch it is handed.
* ``policies``    - action-selection rules and their pooled confidence geometry.
* ``protocol``    - batch chronology, features, and the experiment driver.
* ``run``         - one seeded benchmark setting.
* ``run_config``  - fixed settings or tuning grids through the same runner.
* ``plot`` - plots of saved config results by algorithm or privacy.
"""

__all__ = [
    "oracles",
    "policies",
    "privacy",
    "protocol",
    "run",
    "run_config",
    "plot",
]
