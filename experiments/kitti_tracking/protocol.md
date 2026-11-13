# KITTI连续序列跨帧对象图开发诊断001

补充已有KITTI Object单帧实验，不改nuScenes/主450/缺一/故障/性能方法和阈值，不开新封存，不写论文正文。官方tracking训练分片只有开发诊断信用，不提交排行榜，不宣称官方HOTA/AP或未见分布；上游模型预训练重合未知。

在读取选中标签及推理前固定序列0000、0005、0013，各取按官方image_02成员总数确定的中心连续30帧（start=(N-30)//2），共90帧、3序列；不按类别、遮挡、返回或成败挑选。分别保存物理采集序列关系与已有Object64的图像精确重复审计，不能重复计独立场景。缺输入或标定则保留失败，不换选序列。

官方公开S3端点与官方任务页提供RGB、Velodyne、GPS/IMU、标定、训练轨迹标签。仅使用HTTP Range读取所选大ZIP成员及中心目录，小标签/标定/OXTS ZIP可完整下载；校验Content-Range、ETag、ZIP CRC、长度和SHA。新增文件只写指定outputs目录，不将原始图像/点云/标签提交Git。数据遵守KITTI的CC BY-NC-SA 3.0。

输入端仅读RGB、实测LiDAR、标定与当前/过去IMU位姿；评价端另存官方instance/三维框和类别。前端为当前CPU YOLO-World/MobileSAM/CLIP，不用GT框切图或GT mask。先预定首序列前3帧smoke，通过真实输入/输出后再扩展全部90；同SHA下复用这3帧前端，新增调用只算87，总不同前端90。

几何使用rectified camera0、P2完整第四列及Tr_imu_velo。把P2=K[I|t]的t明确吸收到实际camera2坐标/位姿，使通用K逆投影与GPS/IMU世界坐标一致，深度取实际camera2的z。官方标签location为底面中心，y减h/2后经参考相机0到世界转换；独立重投影检查。IMU从官方经纬高/roll/pitch/yaw转Mercator，原始首帧作原点。tracking无逐帧传感器原始时间戳时使用声明10Hz名义时间，不能称测得精确同步或零扫描运动畸变。

同一实际前端输入比较single_frame、original_fusion、one_to_one_depth_robust_motion；原融合按原默认融合/关联，改进版本使用已开发一对一、深度质量、稳健XY、动态TTL和真实0.1秒名义间隔。单帧不复用跨帧ID。参数不依据本KITTI结果重选，三条件变化是整体开发配置，不单独归因某模块。

主类别Car/Truck/Pedestrian映射car/truck/pedestrian。有效3D标签与image2重叠且实际相机2几何中心1–65m者全部保留，不按检出、深度、遮挡排除。Van/Person_sitting/Cyclist等不强行混进主类；对不支持类别及DontCare重叠另报未知/忽略敏感性，不能把未关联预测都叫确证误检。匹配采用同类别、世界XY3m最大基数一对一，另报共同匹配误差。IDF1沿用本项目固定TrackEval版本在同一3m关联矩阵上计算，非KITTI官方2D HOTA。报告全部GT观测、预测、匹配、重复、中心误差、相邻身份切换、返回证据和单序列差异。

3序列配对bootstrap（4000次/seed17）仅作有限开发不确定性描述；三个序列不足以形成充分跨道路泛化结论，帧不能作独立样本。无语言对象、正确锁定、路径或车辆执行/到达信用。

预算：数据准备1800秒/网络500MB/输出1GB；3帧smoke600秒/0.5GB；全90帧3600秒/3GB；评价600秒/0.5GB。全程CUDA隐藏、CPU各1线程、最低项目余量6GB，不删除历史。任何失败原样保留、独立run-id，不自动重试已尝试的模型帧来挑优。

来源：https://www.cvlibs.net/datasets/kitti/eval_tracking.php ，https://www.cvlibs.net/datasets/kitti/raw_data.php 。TrackEval使用该任务页直接链接的官方评估仓库及本地既有固定版本，不引用未核验地址。
