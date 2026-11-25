from pathlib import Path
import json
O=Path(__file__).resolve().parents[1];D=O/'references'
notes={
'2201.13360':('III Incremental 3D Scene Graph Construction；IV Loop Closure Detection and Optimization','局部TSDF/ESDF构建地点拓扑，聚类得到对象；高层房间组织及回环后的多层修正。本文引用其层次表示和增量维护机制。'),
'2402.12259':('Method：Object feature computation；Relationship feature computation','按可见性选择对象/对象对所在视图，聚合基础模型特征；对齐三维对象及关系特征。正文仅概述开放词汇对象与关系路线。'),
'2307.06135':('方法总体流程及Semantic Search、Iterative Replanning说明','折叠图通过expand/contract检索任务相关子图，经典路径规划连接节点，场景图反馈支持修正方案。本文引用模块分工和子图组织。'),
'2310.03923':('Feature matching；Inverse Rendering and Feature Update','匹配区域嵌入后更新语义TSDF和置信信息。具体分配使用匈牙利扩展算法。正文保留区域匹配与融合的概要。'),
'2306.13631':('3.1 Class-agnostic mask proposal；3.2 Mask-feature computation','冻结三维实例提案，使用类别无关掩膜；按可见性选择视图，形成多尺度裁剪并聚合CLIP特征。用于解释特征空间归属。'),
'2205.13542':('III Method：Unified Representation和BEV pooling','相机与激光特征变换到统一BEV，再经卷积编码用于任务头。本文引用多模态统一空间表示，未移入其原始成绩。'),
'2006.11275':('3 Preliminaries及中心检测、速度与跟踪方法','中心热图配合尺寸、朝向及速度回归，速度外推后按距离进行贪心关联。本文区分检测中心与可见点云中心。'),
'1602.00763':('3 Methodology；3.2 Estimation Model；3.3 Data Association','卡尔曼预测轨迹，IoU代价及门限约束下采用匈牙利分配。正文引用运动预测与一对一匹配的基本结构。'),
'2110.06864':('3 BYTE；Algorithm 1及两阶段关联解释','先关联高分检测，再将剩余轨迹与低分检测匹配，未匹配低分区域作背景处理。本文讨论弱观测的使用条件。'),
'1105.1186':('RRT*算法、Parent/Cost定义及rewiring步骤','通过邻近连接与重连维护较低代价路径；渐近最优性具有相应假设。正文仅用于比较采样搜索机制与状态定义。'),
'1404.2334':('Informed RRT*总体说明及可采样子集定义','获得首次解后，直接在启发函数限定的可改善状态子集中采样；路径长度问题对应长旋转椭球。正文引用搜索区域收缩机制。'),
}
for key,(scope,note) in notes.items():
 assert (D/key/'fulltext.txt').exists()
 (D/key/'阅读核对.md').write_text(f'# 方法相关内容阅读核对\n\n日期：2026-11-20。\n\n原始来源：https://arxiv.org/abs/{key}\n\n阅读位置：{scope}。\n\n核对结论：{note}\n\n阅读范围为正文所引方法内容；不声称逐字通读全部正文与补充材料。\n',encoding='utf-8')
# 下载未完成的PDF不作为阅读缓存。
bad=D/'2403.17846/paper.pdf'
if bad.exists() and not (D/'2403.17846/fulltext.txt').exists():bad.unlink()
print('method reading records',len(notes))
