"""固定官方PyPI工具的RGB无损时域压缩；静止段/移动段均逐字节回读。"""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
import zipfile

import cv2
import numpy as np

from conceptgraph.completion.stage9 import digest, dump

R = Path(__file__).resolve().parents[1]
EXPECTED = 'c7e46fcec401dd990405049d2e2f475e2b397779df2519b544b8aab515195282'


def main():
    out=R/'storage_audit/temporal_lossless_probe_002'
    out.mkdir(exist_ok=False)
    policy=json.loads((R/'json/resource_policy_002.json').read_text())
    size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
    assert size+256*2**20<policy['new_project_artifact_budget_bytes']
    assert shutil.disk_usage(R).free>policy['minimum_physical_free_GiB']*2**30
    vendor=out/'ffmpeg_tool'
    vendor.mkdir()
    metadata_url='https://pypi.org/pypi/imageio-ffmpeg/0.6.0/json'
    raw=urllib.request.urlopen(metadata_url,timeout=30).read()
    metadata=json.loads(raw)
    files=[x for x in metadata['urls'] if x['filename']=='imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl']
    assert len(files)==1
    entry=files[0]
    assert entry['digests']['sha256']==EXPECTED
    wheel=vendor/entry['filename']
    with urllib.request.urlopen(entry['url'],timeout=60) as incoming, wheel.open('xb') as stream:
        shutil.copyfileobj(incoming,stream)
    assert digest(wheel)==EXPECTED
    with zipfile.ZipFile(wheel) as archive:
        selected=[n for n in archive.namelist() if '/binaries/ffmpeg-' in n and not n.endswith('/')]
        assert len(selected)==1
        binary=vendor/Path(selected[0]).name
        binary.open('xb').write(archive.read(selected[0]))
        for name in archive.namelist():
            if 'LICENSE' in name.upper() and not name.endswith('/'):
                (vendor/Path(name).name).open('xb').write(archive.read(name))
    binary.chmod(0o755)
    version=subprocess.run([str(binary),'-version'],capture_output=True,text=True,check=True).stdout
    license_text=subprocess.run([str(binary),'-L'],capture_output=True,text=True,check=True)
    (vendor/'ffmpeg_version.txt').open('x').write(version)
    (vendor/'ffmpeg_license.txt').open('x').write(license_text.stdout+license_text.stderr)
    dump(vendor/'download_receipt.json',dict(metadata_url=metadata_url,wheel_url=entry['url'],
        version='imageio-ffmpeg==0.6.0',wheel_bytes=wheel.stat().st_size,wheel_sha256=EXPECTED,
        binary_sha256=digest(binary),no_python_install_or_import=True,goal_complete=False))
    source=R/'carla_official/prius_controller_navigation_005/car20/full_rule_geometry/frames'
    allfolders=sorted(source.iterdir())
    reports=[]
    manifest=[]
    for regime,folders in [('initial',allfolders[:40]),('moving',allfolders[160:200])]:
        assert len(folders)==40
        for filename in ['rgb.png','official_depth_raw_bgr.png']:
            video=out/f'{regime}_{Path(filename).stem}.mkv'
            log=out/f'{video.stem}_encode.log'
            first=cv2.imread(str(folders[0]/filename))
            h,w=first.shape[:2]
            command=[str(binary),'-nostdin','-n','-hide_banner','-loglevel','error',
                '-f','rawvideo','-pix_fmt','bgr24','-video_size',f'{w}x{h}',
                '-framerate','5','-i','pipe:0','-an','-c:v','libx264rgb','-crf','0',
                '-preset','medium','-pix_fmt','bgr24',str(video)]
            started=time.monotonic()
            hashes=[]
            original_bytes=0
            with log.open('xb') as errors:
                process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=errors)
                try:
                    for folder in folders:
                        p=folder/filename
                        frame=cv2.imread(str(p))
                        assert frame.shape==first.shape
                        raw_frame=frame.tobytes()
                        sha=hashlib.sha256(raw_frame).hexdigest()
                        process.stdin.write(raw_frame)
                        hashes.append(sha)
                        original_bytes+=p.stat().st_size
                        manifest.append(dict(regime=regime,path=str(p),file_sha256=digest(p),
                            raw_BGR_sha256=sha,encoded_video=video.name,video_frame=len(hashes)-1))
                    process.stdin.close()
                    assert process.wait(timeout=120)==0
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            encode_s=time.monotonic()-started
            decoded=subprocess.run([str(binary),'-nostdin','-hide_banner','-loglevel','error',
                '-i',str(video),'-f','rawvideo','-pix_fmt','bgr24','pipe:1'],capture_output=True,check=True).stdout
            count=h*w*3
            assert len(decoded)==len(folders)*count
            for i,(folder,sha) in enumerate(zip(folders,hashes)):
                actual=decoded[i*count:(i+1)*count]
                assert hashlib.sha256(actual).hexdigest()==sha
                assert np.array_equal(np.frombuffer(actual,np.uint8).reshape(h,w,3),cv2.imread(str(folder/filename)))
            reports.append(dict(regime=regime,kind=filename,frames=40,byte_exact_frames=40,
                original_PNG_bytes=original_bytes,lossless_video_bytes=video.stat().st_size,
                reduction_fraction=1-video.stat().st_size/original_bytes,encoding_wall_s=encode_s,
                video_sha256=digest(video),encode_command=command))
    assert all(digest(x['path'])==x['file_sha256'] for x in manifest)
    dump(out/'input_manifest.json',manifest)
    dump(out/'report.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        results=reports,distinct_existing_sensor_frames=80,actual_new_sensor_frames=0,
        actual_GPU_calls=0,all_original_files_unchanged=True,source_sha256=digest(__file__),
        existing_experiment_storage_unchanged=True,goal_complete=False,
        scope='既有任务80帧的两路传感器无损压缩测试，保留全部失败与旧数据；未在当前冻结导航中启用。'))
    print(json.dumps(reports,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
