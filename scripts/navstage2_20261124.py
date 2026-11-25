"""新入口：prepare冻结81任务；run仅方法输入；evaluate独立末端评价。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conceptgraph.navigation.navstage2_runner_20261124 import main

if __name__ == '__main__':
    main()
