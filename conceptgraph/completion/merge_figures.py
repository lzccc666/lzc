"""校验后合并已通过检查的Python科研图包，不重新绘图或覆盖源包。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--source',action='append',required=True)
    args=parser.parse_args()
    root=OUTPUT/'thesis'
    destination=root/args.run_id
    destination.mkdir(exist_ok=False)
    sources=[]
    captions=[]
    for name in args.source:
        source=root/name
        manifest=json.loads((source/'manifest.json').read_text())
        for filename,sha in manifest['files'].items():
            if digest(source/filename)!=sha:
                raise ValueError('source figure package changed')
        for qa in ['qa_automatic.json','qa_visual_pdf_002.json']:
            if not json.loads((source/qa).read_text())['passed']:
                raise ValueError('source figure QA failed')
        sources.append(dict(package=name,manifest_sha256=digest(source/'manifest.json'),
            visual_qa_sha256=digest(source/'qa_visual_pdf_002.json')))
        for file in source.iterdir():
            if file.suffix not in {'.svg','.pdf','.png','.tiff','.csv'} or file.name.startswith('pdf_preview_'):
                continue
            target=destination/file.name
            if target.exists():
                if file.suffix!='.csv':
                    raise ValueError('duplicate figure name')
                target=destination/(name+'_'+file.name)
            shutil.copy2(file,target)
        captions.append((source/'图注与统计说明.md').read_text())
    (destination/'图注与统计说明.md').write_text('\n\n'.join(captions))
    for name in ['qa_automatic.json','qa_visual_pdf_002.json']:
        dump(destination/name,dict(passed=True,merged_packages=sources,
            note='逐包核验原始SHA与实际QA，复制已通过的图，不重新绘制',goal_complete=False))
    shutil.copy2(__file__,destination/'generator_source.py')
    dump(destination/'manifest.json',dict(sources=sources,backend='Python/matplotlib',
        source_sha256=digest(Path(__file__)),
        files={file.name:digest(file) for file in destination.iterdir() if file.is_file()},goal_complete=False))


if __name__=='__main__':
    main()
