from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from conceptgraph.completion.nuscenes_full_20261125_resource_runtime_v8 import main
if __name__=='__main__':main()
