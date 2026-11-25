from pathlib import Path
import re,json,csv,hashlib
O=Path(__file__).resolve().parents[1];text=(O/'sources/论文正文.md').read_text();metadata=json.loads((O/'references/metadata.json').read_text());by={r['id']:r for r in metadata};order=[]
for group in re.findall(r'\[@([^\]]+)\]',text):
 for key in group.split(';'):
  if key not in order:order.append(key)
notes={'1604.07446':'区分规划与控制、车辆模型及环境假设；原文综述，不引入其外部实验分数','1903.11027':'nuScenes传感器、官方对象标注和数据基准背景','KITTI2013':'KITTI平台与数据类别；本文抽样规模来自项目报告','1609.05130':'CNN类别预测在ElasticFusion几何对应下概率融合','1910.02490':'VIO、位姿图、网格、语义重建模块化','1808.08378':'每对象TSDF与6DoF对象图','1804.09194':'独立运动对象识别、跟踪和重建','2002.06289':'层次化动态场景图与空间关系','2004.03967':'PointNet/GCN预测对象与关系及3DSSG数据','2211.15654':'点级开放词汇特征与文本图像空间对齐','2303.09553':'辐射场内多尺度语言特征蒸馏','1711.07280':'R2R与Matterport图像导航任务，非本文录制车辆执行','2207.04429':'组合ViNG、CLIP、语言模型进行地标导航','2210.05714':'原法使用LSeg稠密像素嵌入，本项目仅稀疏crop网格近似','1909.10838':'道路语言对象指代任务，不等于车辆控制','2103.00020':'图文对比学习、共享特征空间与零样本分类','2304.02643':'可提示分割模型，作为已有前端归属','2401.17270':'RepVL-PAN与区域文本对比；HUST/Tencent作者单位已核对','2409.12191':'动态分辨率、M-RoPE；实际模型为Qwen2-VL','2106.09685':'冻结权重与低秩增量W+BA','2305.14314':'量化低秩适配；明确本文实际bf16 LoRA','Dijkstra1959':'非负图最短路径基础；元数据核验，全文读取受限','Hart1968':'启发式最小代价路径与可采纳启发条件；作者机构PDF已检索，下载/全文读取受限','Dolgov2010':'半结构环境车辆路径搜索；SAGE作者摘要核验，全文下载失败','1609.01775':'全序列身份匹配与ID指标公式','2009.07736':'检测、关联、定位的评价维度'}
refs=[];bib=[];ris=[];check=[]
for num,key in enumerate(order,1):
 r=by[key];r['number']=num;authors=r['authors'];year=r.get('date','')[:4];doi=r.get('doi','');url=r['url'];author_display=', '.join(authors[:3])+(', et al.' if len(authors)>3 else '')
 if key[0].isdigit():
  formatted=f"{author_display}. {r['title']}[EB/OL]. arXiv:{key}, {year}. {url}.";venue='arXiv'
 else:
  formatted=f"{author_display}. {r['title']}[J]. {r.get('venue','')}, {year}, {r.get('volume','')}({r.get('issue','')}): {r.get('pages','')}. DOI: {doi}.";venue=r.get('venue','')
 claim=notes.get(key,r.get('claim',''))
 if not claim:raise ValueError('missing claim mapping: '+key)
 formatted=formatted.replace('et al..','et al.');r.update(formatted=formatted,claim=claim);refs.append(r)
 bib.append('@misc{'+key.replace('.','_')+',\n  title={'+r['title']+'},\n  author={'+' and '.join(authors)+'},\n  year={'+year+'},\n  doi={'+doi+'},\n  url={'+url+'}\n}')
 ris.append('TY  - '+('JOUR' if not key[0].isdigit() else 'GEN')+'\nTI  - '+r['title']+'\n'+''.join('AU  - '+a+'\n' for a in authors)+'PY  - '+year+'\nDO  - '+doi+'\nUR  - '+url+'\nER  - \n')
 files=list((O/'references'/key).glob('*fulltext.txt'));full=max([len(p.read_text()) for p in files],default=0)
 reading=O/'references'/key/'阅读核对.md'
 status=('作者元数据与摘要已核验；全文已获取，相关方法核对见阅读记录' if reading.exists() else '作者元数据与摘要已核验；全文缓存已获取，既有阅读范围见交接记录') if full>1000 else '元数据与作者摘要核验；本轮按摘要支撑范围引用，未声称通读全文'
 check.append({'序号':num,'ID':key,'作者':'; '.join(authors),'题名':r['title'],'年份':year,'DOI或稳定链接':doi or url,'原文支撑':claim,'核验状态':status,'全文字符数':full,'文件来源':'; '.join(str(p.resolve()) for p in files),'访问日期':r.get('access_date','2026-11-15')})
(O/'references/verified_references.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2));(O/'references/参考文献.bib').write_text('\n\n'.join(bib));(O/'references/参考文献.ris').write_text('\n'.join(ris))
with (O/'引用核验清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(check[0]));w.writeheader();w.writerows(check)
(O/'references/参考文献及阅读依据.md').write_text('# 参考文献及阅读依据\n\n'+ '\n\n'.join(f"[{r['number']}] {r['formatted']}\n\n支撑范围：{r['claim']}" for r in refs)+'\n\n核验备注：学术MCP工具在本会话不可用，使用arXiv作者提交全文、CVF/作者机构页面和Crossref核验。Crossref部分请求限流；Dolgov正确DOI尾号为9210。R2R误定位编号1802.02179已排除，改为1711.07280。SAM的arXiv空页面PDF不作为阅读依据，已读取CVF正式PDF。Dijkstra全文受限；Hart作者PDF已检索但打开超时，不声称已读全文；Dolgov通过出版者摘要核验，未声称已读其全文。\n')
print('verified cited references',len(refs))
