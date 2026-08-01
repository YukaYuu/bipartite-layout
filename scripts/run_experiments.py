import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bipartite_layout.experiments.batteries import main_compare_movielens_and_dblp

if __name__ == "__main__":
    main_compare_movielens_and_dblp()
