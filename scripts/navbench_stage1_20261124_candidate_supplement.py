#!/usr/bin/env python3
"""Create six-scene fixed-frame read-only browsing supplement."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conceptgraph.completion.navbench_stage1_20261124_candidate_supplement import main

if __name__ == '__main__':
    main()
