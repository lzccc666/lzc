"""制作一个连续场景的离线任务标注包：全部现有关键帧与因果对象图。"""
import base64
import gzip
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image

from conceptgraph.completion.stage9 import PROJECT, OUTPUT, digest


def main():
    scene='scene-0012'
    root=PROJECT/'dataset/annotations/natural_graph_v004/episode_scene0012_20261120_001'
    if shutil.disk_usage(PROJECT).free<20*1024**3:raise OSError('磁盘低于保护线')
    root.mkdir(parents=True,exist_ok=False)
    cache=PROJECT/'outputs/results/nuscenes/validation/iterations/vehicle_frontend_full_20261118_v1'/scene/'frames.jsonl.gz'
    sensor=OUTPUT/'mapping_evaluation/causal_mapping_full_001'/scene/'frames.jsonl'
    with gzip.open(cache,'rt') as f:rows=[json.loads(x) for x in f]
    sensors={r['frame_idx']:r for r in map(json.loads,sensor.open())}
    frames=[]
    names=dict(car='小汽车',truck='卡车',bus='公交车',bicycle='自行车',motorcycle='摩托车',pedestrian='行人')
    for idx,row in enumerate(rows):
        assert row['frame_idx']==idx
        source=sensors[idx]
        pose,k=np.asarray(source['camera_pose']),np.asarray(source['intrinsic'])
        nodes=[]
        for obj in row['predictions']['frontend_v2_evidence']:
            camera=pose[:3,:3].T@(np.asarray(obj['center'])-pose[:3,3])
            uv=k@camera
            node=dict(node_id=obj['node_id'],short=obj['node_id'].replace('track-','N'),category=obj['category'],
                zh=names[obj['category']],world_center=obj['center'],right=float(camera[0]),forward=float(camera[2]),
                uv=[float(uv[0]/uv[2]/1600),float(uv[1]/uv[2]/900)] if camera[2]>0 else None,
                last_seen_frame=obj['last_seen_frame'],num_detections=obj['num_detections'])
            nodes.append(node)
        edges=[[a,b] for a in range(len(nodes)) for b in range(a+1,len(nodes))
               if np.linalg.norm(np.asarray(nodes[a]['world_center'][:2])-nodes[b]['world_center'][:2])<=8]
        imagepath=Path(row['rgb'])
        with Image.open(imagepath) as image:
            image=image.convert('RGB');image.thumbnail((1280,900));out=io.BytesIO();image.save(out,format='JPEG',quality=88)
        frames.append(dict(frame_idx=idx,camera_timestamp_s=source['camera_timestamp_s'],
            relative_s=source['camera_timestamp_s']-sensors[0]['camera_timestamp_s'],rgb_sha256=digest(imagepath),
            nodes=nodes,edges=edges,image='data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode()))
    with (root/'causal_graphs.jsonl').open('x') as f:
        for r in frames:f.write(json.dumps({k:v for k,v in r.items() if k!='image'},ensure_ascii=False)+'\n')
    manifest=dict(campaign='episode_scene0012_20261120_001',scene=scene,frames=len(frames),
        duration_s=frames[-1]['relative_s'],cache_sha256=digest(cache),sensor_manifest_sha256=digest(sensor),
        graphs_sha256=digest(root/'causal_graphs.jsonl'),page_source_sha256=digest(PROJECT/'conceptgraph/completion/episode_review.html'),
        generator_sha256=digest(__file__),selection='first_sorted_validation_scene_all_existing_keyframes',
        split='workflow_pilot_training_validation_not_new_test',GT_or_future_graph_in_display=False,
        graph_scope='当前可见预测交通节点＋同帧坐标计算的8米内接近边；非原模型完整持久关系边模块',
        image_scope='现有nuScenes关键帧全部显示，不等于原始相机全部高频帧',human_annotations_created=0)
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    payload=dict(campaign=manifest['campaign'],scene=scene,manifest_sha256=digest(root/'manifest.json'),frames=frames)
    template=(PROJECT/'conceptgraph/completion/episode_review.html').read_text()
    (root/'index.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False)))
    (root/'填写说明.txt').write_text('解压后双击index.html；无需服务器或网络。先播放整个场景，写一条整段任务指令，注明开始帧。按帧或区间标目标可见状态，在首次出现、遮挡后再出现、歧义等关键帧点目标。可勾选查看当前预测对象图，模型图不是标准答案。导出JSON发回聊天。整段只有一条指令，不要求每帧重复写。该回放不能证明车辆按指令实际到达。\n')
    with zipfile.ZipFile(root/'连续场景审核_scene0012_离线包.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['index.html','manifest.json','causal_graphs.jsonl','填写说明.txt']:z.write(root/name,name)
    assert sum(p.stat().st_size for p in root.iterdir())<50*1024**2
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
