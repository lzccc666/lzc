"""本机全nuScenes冻结认证与后续统一实验的新命令入口。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from conceptgraph.completion.nuscenes_full_20261125 import main

if __name__=='__main__': main()
