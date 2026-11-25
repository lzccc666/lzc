from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from conceptgraph.completion.nuscenes_full_20261125_single8_reedge import run
if __name__=='__main__':run()
