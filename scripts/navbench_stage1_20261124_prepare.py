"""Entry point for the frozen, CPU-only navigation benchmark preparation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.completion.navbench_stage1_20261124 import main


if __name__ == '__main__':
    main()
