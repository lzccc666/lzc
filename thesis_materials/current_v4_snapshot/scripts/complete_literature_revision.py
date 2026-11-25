"""整合补检文献；保持实验结果及表图数据。"""
from pathlib import Path
O=Path(__file__).resolve().parents[1]
for name in ['论文正文.md','前两章文献综述修订.md']:
 p=O/'sources'/name;s=p.read_text(encoding='utf-8')
 if '[@2410.11989;2503.08474]' not in s:
  marker='语言导航研究将上述表示用于目标与路线推断。'
  extra='''动态开放词汇图开始面向长期交互及城市道路扩展。DovSG通过局部更新调整对象及空间关系，使地图能够响应移动操作中的场景变化；CURB-OSG将多个主体的相机、激光雷达观测与协同位姿图结合，构建城市驾驶场景的层次化语义图[@2410.11989;2503.08474]。前者强调交互后的状态更新，后者强调多视角覆盖及共同空间参考。这些研究使动态维护与道路语义图之间的联系更加明确。本文在单车已知位姿条件下研究稀疏对象维护，重点检查观测支持、持续身份及关系时效。

'''
  if marker not in s:raise ValueError('insertion marker absent')
  s=s.replace(marker,extra+marker,1)
 if '[@2410.08189]' not in s:
  marker='语言特征地图适合把自然表达转换为空间查询。'
  extra='''SG-Nav在线组织对象、组和房间之间的三维关系，利用层次提示推断目标位置，并通过再感知机制修正对象导航中的感知错误[@2410.08189]。其图结构为语言推理提供了当前场景上下文，也体现了目标选择与观测更新之间的相互作用。本文采用已录制观测评价图上目标选择，并保留无匹配及证据不足输出，以分析当前对象支持对语言判断的影响。

'''
  if marker not in s:raise ValueError('insertion marker absent')
  s=s.replace(marker,extra+marker,1)
 s=s.replace('模型装载代码为conceptgraph/scripts/streamlined_detections.py；检测输出文件与代码位置一并纳入索引。','视觉前端加载脚本、实际权重和检测输出文件一并纳入代码与结果索引。')
 p.write_text(s,encoding='utf-8')
print('three focused studies integrated')
