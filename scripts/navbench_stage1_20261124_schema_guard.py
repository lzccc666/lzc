#!/usr/bin/env python3
"""Append stage-1 schema-v2 checks and source binding without changing v1."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conceptgraph.completion.navbench_stage1_20261124_schema_guard import main

if __name__ == '__main__':
    main()
