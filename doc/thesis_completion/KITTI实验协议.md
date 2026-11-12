# KITTI 外部跨数据集诊断协议（在正式64帧推理前冻结）

KITTI 是真实车载数据，覆盖城市、乡村与道路交通场景。本地 KITTI Object 的 RGB、Velodyne、相机标定和官方三维框均可读取；SemanticKITTI 另有序列点云、实例标签和标注用位姿。原始压缩包保持只读。此实验不下载完整数据、不使用公开排行榜数值冒充项目实测。

## 选样与用途

- 在读取样本标签与运行模型前，先写 `data_manifest/kitti_object_uniform64_001/selection.json`。
- 固定选择 KITTI Object training 的 `round(linspace(0,7480,64))`，不按类别、难度、支持量或实验成败挑选。实际对应33个原始采集drive。
- drive 使用官方 devkit 的 `train_rand.txt`（从1开始）和 `train_mapping.txt` 联合映射；不是把64帧称作64个独立场景。
- 前3个固定ID用于CPU smoke，之后64帧全部运行，正式记录保留这3帧，不把smoke与正式重复帧累加样本量。
- 用途是已访问的外部数据开发诊断。YOLO/CLIP上游预训练与KITTI的重叠未知，不能称为严格训练分布外泛化或封存测试。不会因KITTI结果调整冻结主矩阵或新测试阈值。

## 隔离输入与真实前端

`agent_inputs/` 只含RGB、calib与实测LiDAR投影。`evaluation_only/` 含官方目标框、类别和原始成员哈希；前端入口只读 `agent_manifest.json`，不读取评价标签。

直接复用 `streamlined_detections.detect_rgb_file`：YOLO-World、MobileSAM、CLIP ViT-H/14，14个现有类别，置信度0.1，不将真值框喂入SAM。不做人工补框或缓存结果代替真实推理。保存掩膜、框、crop、特征和RGB哈希。

仅在隔离CPU子进程运行，明确 `CUDA_VISIBLE_DEVICES=''`、2个Torch线程，YOLO/SAM/CLIP设备均为CPU；不另占正在运行主矩阵的GPU。共享主机资源，因此不计独占性能信用。记录真实墙钟但不与CUDA速度横比。

## 标定与两种固定条件

LiDAR先经 `Tr_velo_to_cam` 与 `R0_rect` 转入矫正相机0，以完整3×4 `P2` 投影到image_2。保留第四列和非零第三行偏置；像素用floor，同像素只保留最近实测点。不把image_2近似内参逆投影的相机2坐标直接比较相机0标签。保存每个有效像素对应的原始相机0三维点，中心不依赖量化后的像素逆投影。

两种条件共享全部真实检测、掩膜和LiDAR：

1. `mask_points_mean`：掩膜内1–65m实测点，至少5点，求可见点均值。
2. `mask_points_robust_mean`：使用项目已冻结 `completion.depth.depth_support(robust=True)`，取固定深度间隙分簇的最大支持簇（并列取更近），至少5点后求均值。

这些是单帧可见表面中心诊断，不等同于原完整融合建图、官方三维框估计或物体真实体积中心。稀疏表面与完整框中心有系统偏差；必须保留负结果。

## 独立评价

- KITTI标签尺寸顺序h,w,l，location是底面中心；转几何中心时y减h/2。坐标单位m。
- 主表仅car/Truck/Pedestrian映射的car/truck/pedestrian。Cyclist与bicycle检测框语义不同，仅单列代理诊断。
- 真值几何中心在1–65m范围内的目标全部保留，不按未检出、遮挡、低支持或无深度删除正样本。
- 同类别最大基数一对一匹配：二维IoU≥0.5；三维中心距离≤3m。**不是KITTI官方AP、3D IoU、R40或官方难度评价**。
- 未匹配预测与DontCare、对应类别的超范围标签及Van/Person_sitting忽略框IoA≥0.5时可忽略；有效目标上的重复预测不能被DontCare隐藏。
- 同时报三维召回、三维正确/全部有效二维检测、三维正确/有深度检测、深度未知数，以及正确二维匹配且有深度的条件中心误差和对应分母。不能通过全部不定位提高精度而不披露覆盖。
- 方法差异做4000次、种子17的配对drive聚类bootstrap。原始每帧每类每方法记录与所有未知保留。

## 预算与完成边界

prepare：1800秒、1GiB；3帧smoke：1800秒、0.3GiB；64帧正式：10800秒、2GiB；项目可用空间低于6GiB停止。独立run-id，保留失败，不覆盖挑优；每帧输入、输出、模型和源码SHA校验。

KITTI Object这批输入未提供连续世界位姿、独立语言指令/安全接近标签、车道拓扑与控制执行证据，因此本实验中的“生成路线”“选对语言对象”“正确锁定”“实际到达”均无信用，实际完整导航到达次数为0。单帧感知定位结果不填入新的封存导航主表；不宣布Goal完成。

## 来源

数据与许可：https://www.cvlibs.net/datasets/kitti/ （CC BY-NC-SA 3.0）。

三维检测任务：https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d 。

序列标签说明：https://semantic-kitti.org/dataset （低16位语义，高16位跨帧实例；位姿来自标注所用SuMa估计，并非本项目独立SLAM结果）。

引用KITTI：Geiger A, Lenz P, Urtasun R. *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*. CVPR, 2012。

若后续使用SemanticKITTI，再引用Behley J, et al. *SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences*. ICCV, 2019。本轮不将其真值分割冒充RGB感知输出。
