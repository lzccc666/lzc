"""保存作者提交页面并提取引文信息；仅补充文献，不运行实验。"""
from pathlib import Path
import concurrent.futures as cf, subprocess, json, re, sys
from html.parser import HTMLParser
class SourceParser(HTMLParser):
 def __init__(self):super().__init__();self.meta={};self.abstract=[];self.in_abstract=False
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='meta' and 'name' in a:self.meta.setdefault(a['name'],[]).append(a.get('content',''))
  if tag=='blockquote' and 'abstract' in a.get('class',''):self.in_abstract=True
 def handle_endtag(self,tag):
  if tag=='blockquote':self.in_abstract=False
 def handle_data(self,data):
  if self.in_abstract:self.abstract.append(data)
O=Path(__file__).resolve().parents[1];D=O/'references'
claims={
'1502.00956':'ORB特征、关键帧与回环组织视觉定位和地图；为几何参考提供背景。',
'1610.06475':'单目、双目和RGB-D共享的SLAM系统；解释深度输入与尺度的联系。',
'2007.11898':'视觉惯性和多地图融合；解释位姿估计与本文已知位姿输入的分工。',
'1806.05620':'语义检测与多视图几何识别动态区域，构建静态背景地图。',
'1903.01177':'体积地图融合stuff语义和thing实例，维护跨帧全景标签。',
'2201.13360':'在线建立并优化层次化三维场景图，组织对象、位置和房间。',
'2106.14386':'多机器人分布式度量语义建图与鲁棒回环；作为系统组织背景。',
'2402.12259':'二维基础模型特征与三维表征对齐，支持开放对象和关系查询。',
'2403.17846':'楼层、房间和对象的分层开放词汇图及语言目标导航。',
'2307.06135':'通过层次场景图检索与规划反馈，缩小大语言模型任务规划上下文。',
'2310.03923':'区域视觉语言特征与TSDF融合，构建可查询三维语义地图。',
'2302.07241':'多视图像素特征融合为支持多模态查询的三维表示。',
'2306.13631':'类别无关三维实例掩膜与多视图语言特征聚合。',
'2201.03546':'像素特征和文本标签对齐，提供语言驱动的稠密分割表征。',
'2112.03857':'联合目标检测与短语定位预训练，建立区域和语言的对应。',
'2201.02605':'利用图像级标签扩展目标检测类别词表。',
'2303.05499':'语言引导查询和跨模态交互实现开放集对象检测。',
'2304.07193':'自监督学习通用视觉特征，作为与图文对齐不同的表征路线。',
'2408.00714':'流式记忆支持图像及视频中的提示分割。',
'2205.13542':'在统一鸟瞰图中融合相机与激光雷达特征。',
'2006.11275':'以三维中心及尺寸、方向和速度回归支持检测与跟踪。',
'1812.05784':'柱状点云编码连接稀疏三维输入与二维卷积检测。',
'1602.00763':'卡尔曼预测与匈牙利分配组成在线多对象跟踪。',
'1703.07402':'在运动匹配中引入深度外观度量以维持遮挡对象身份。',
'2110.06864':'分阶段关联高低分检测，利用低分观测补充轨迹。',
'2008.08063':'三维卡尔曼滤波及一对一匹配构成三维跟踪基线。',
'2203.17270':'空间交叉注意力和时间注意力构建多相机鸟瞰图特征。',
'2212.10156':'以规划为目标串联感知、预测等任务的查询式系统组织。',
'2307.05973':'把语言任务转为三维价值和约束地图，连接机器人操作规划。',
'2502.13923':'后续视觉语言模型技术报告；与本文实际Qwen2-VL版本区分。',
'2312.03275':'视觉语言语义价值与前沿地图结合，支持未知环境目标探索。',
'1105.1186':'采样规划渐近最优性及RRT*重连机制。',
'1404.2334':'首次解后在可改进路径的椭球子集中采样，提高规划搜索效率。',
'2410.08189':'在线三维场景图组织对象、组和房间，以层次提示及再感知支持零样本对象导航。',
'2410.11989':'开放词汇对象与空间关系形成动态图，通过局部更新响应移动操作中的场景变化。',
'2503.08474':'多主体相机和激光雷达观测结合位姿图，构建城市驾驶场景的层次化开放词汇动态图。',
}
def fetch(item):
 key,claim=item;p=D/key;p.mkdir(exist_ok=True);dest=p/'landing.html'
 if not dest.exists():
  subprocess.run(['curl.exe' if sys.platform=='win32' else 'curl','-L','--fail','--retry','2','--max-time','45','-sS','https://arxiv.org/abs/'+key,'-o',str(dest)],check=True)
 s=SourceParser();s.feed(dest.read_text(encoding='utf-8'))
 def meta(n):return s.meta.get(n,[])
 r={'id':key,'url':'https://arxiv.org/abs/'+key,'access_date':'2026-11-20','claim':claim}
 for k,n in [('title','citation_title'),('authors','citation_author'),('date','citation_date'),('doi','citation_doi')]:
  a=meta(n);r[k]=a if k=='authors' else (a[0] if a else '')
 r['abstract']=' '.join(' '.join(s.abstract).split())
 if not r['title'] or not r['authors'] or not r['abstract']:raise ValueError((key,'incomplete source page'))
 r['status']='作者提交页面、完整作者表、首次提交日期及摘要已核验；全文状态另列'
 return r
if __name__=='__main__':
 previous=json.loads((D/'metadata.json').read_text(encoding='utf-8'));by={r['id']:r for r in previous}
 with cf.ThreadPoolExecutor(max_workers=4) as pool:
  for r in pool.map(fetch,claims.items()):
   by[r['id']]=r;print(r['id'],r['title'],flush=True)
 (D/'metadata.json').write_text(json.dumps(list(by.values()),ensure_ascii=False,indent=2),encoding='utf-8')
 print('metadata records',len(by))
