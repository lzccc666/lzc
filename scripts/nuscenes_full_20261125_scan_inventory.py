"""原sensor元数据的scan来源盘点；不打开图像、GT或模型分数。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
from collections import Counter
from conceptgraph.completion import nuscenes_full_20261125 as base

def main():
    manifest=json.loads((base.DATA/'manifest.json').read_text());c=Counter();bysetting={k:Counter() for k in ['front','front3','six']}
    for scene in manifest['scenes']:
        request=next(base.rows(scene['request']))
        for group in request['groups']:
            c['groups']+=1;front={group['views']['CAM_FRONT']['lidar']}-{None}
            for label,cameras in [('front',manifest['cameras'][:1]),('front3',manifest['cameras'][:3]),('six',manifest['cameras'])]:
                ids={group['views'][cam]['lidar'] for cam in cameras}-{None};extra=ids-front;b=bysetting[label]
                b['groups']+=1;b['unique_scan_count:'+str(len(ids))]+=1;b['extra_scan_groups']+=int(bool(extra));b['extra_unique_scan_total']+=len(extra)
                b['current_front_missing_LiDAR_groups']+=int(not front);b['any_missing_view_LiDAR_groups']+=int(any(group['views'][cam]['lidar'] is None for cam in cameras))
    base.dump(base.REPORT/'declared_scan_support_inventory.json',dict(created_utc=base.now(),metadata_only=True,source_sha256=base.sha(__file__),
        manifest_sha256=base.sha(base.DATA/'manifest.json'),counts=dict(c),settings={k:dict(v) for k,v in bysetting.items()},
        limits='metadata requests only, original source membership still separately certified; not model performance or pure camera causal effect',GPU_calls=0,GT_accessed=False))

if __name__=='__main__':main()
