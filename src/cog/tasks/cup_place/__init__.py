"""Cup-place task family for the Cost of Generality study.

Importing this package registers all gym env IDs (state / visuomotor /
mimic variants for every generality level defined in ``levels.py``).
"""

from .config import franka  # noqa: F401  (triggers gym.register calls)
from .mimic import *  # noqa: F401,F403  (registers -Mimic-v0 IDs)
