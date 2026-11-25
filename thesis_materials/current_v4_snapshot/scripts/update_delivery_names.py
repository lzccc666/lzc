from pathlib import Path
import re,json,shutil
O=Path(__file__).resolve().parents[1]
for name in ['build_word.py','qa_document.py','rebuild.sh','finish_delivery.py']:
 p=O/'scripts'/name;s=p.read_text(encoding='utf-8')
 s=s.replace('硕士论文初稿_公开数据版_v2','硕士论文初稿')
 s=s.replace('公开数据版硕士论文完整初稿','三维语义场景图硕士论文修订稿').replace('公开数据版 · 扩充修订稿 v2','文献调研修订稿').replace('整理日期：2026年11月15日','整理日期：2026年11月20日')
 s=s.replace("'参考文献26条且全部正文引用': len(R) == 26", "'参考文献不少于50条且全部正文引用': len(R) >= 50")
 if name=='finish_delivery.py':
  needle="    f'分层原始计数"
  pos=s.index(needle)
  s=s[:pos]+"    '前两章独立引用不少于50项工作': len(set(';'.join(re.findall(r'\\[@([^\\]]+)\\]', S[S.index('# 第1章 '):S.index('# 第3章 ')])).split(';'))) >= 50,\n    '正文和正式参考文献无指定排除文献': not re.search(r'conceptgraphs?', S + json.dumps(R), re.I),\n"+s[pos:]
 p.write_text(s,encoding='utf-8',newline='\n')
# 本版输出只显示统一文件名，修订前文件保留在历史版本中。
archive=O/'历史版本/文献扩充前';archive.mkdir(parents=True,exist_ok=True)
for p in list(O.glob('*公开数据版*.docx'))+list(O.glob('*公开数据版*.pdf')):
 target=archive/p.name
 if not target.exists():shutil.move(str(p),str(target))
 else:
  # 历史副本由本轮备份生成，按字节核对后移除当前目录中的重复件。
  if p.read_bytes()!=target.read_bytes():raise ValueError('different backup '+str(p))
  p.unlink()
print('delivery names updated')
