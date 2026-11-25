"""技术运行修订：cv2导入会前置lib64，导入后恢复冻结CUDA运行环境断言。"""
from pathlib import Path
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from conceptgraph.completion.nuscenes_full_20261125_resource_pilot import main
if __name__=='__main__':
    prior=os.environ.get('LD_LIBRARY_PATH','');required='/root/miniconda3/envs/conceptgraph/lib'
    os.environ['LD_LIBRARY_PATH']=':'.join([required,*[p for p in prior.split(':') if p and p!=required]])
    main()
