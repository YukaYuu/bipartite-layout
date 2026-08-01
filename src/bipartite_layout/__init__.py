"""Bipartite graph layout optimization (MovieLens / DBLP).

Sets up the matplotlib backend before any submodule can import pyplot —
this must stay the only place `matplotlib.use(...)` is called.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False

__version__ = "0.1.0"
