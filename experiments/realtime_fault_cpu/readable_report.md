# CPU真实前端六故障条件逐帧复核

本批全部六条件、同一任务config00-positive0/seed17。CPU/llvmpipe、规则前端、15秒TTL；不新增独立任务/配置覆盖，不是正式GPU故障或独占性能。
逐帧原图哈希相同、零深度未加密、位姿与物理车态隔离、真实检测七字段保留索引全部核对。复跑既有独立车身执行审计与保存结果一致；本次不是第二套独立几何实现。

|条件|帧/真实前端调用|注入帧|控制tick|终止|实际到达|
|---|---:|---:|---:|---|---|
|baseline|10|0|391|ARRIVED|True|
|sensor_blackout_4s|9|1|377|ARRIVED|True|
|lidar_dropout_80pct|12|3|382|ARRIVED|True|
|depth_noise_02m|11|3|379|ARRIVED|True|
|pose_noise_025m_2deg|11|3|369|ARRIVED|True|
|detection_dropout_30pct|10|3|372|ARRIVED|True|

实际断流4.003093523904681秒；期间40控制tick，过期制动0tick，断流期停止=False，随后新观测下运动=True。15秒TTL下未触发过期制动的负结果保留，不能称该CPU条件证明过期安全停止。
全部原始成功/失败按实际记录保留，真实人工0，正式GPU矩阵仍待前置主矩阵及缺一任务完整结束。
