# CPU分阶段计时试跑与独立统计复核

12个真实RGB的YOLO/MobileSAM/CLIP重新执行；10帧真实保存观测重算建图/规划，391个保存控制状态逐字段重算。没有新的车辆执行或到达信用，Qwen/GPU调用0。

|阶段/条件|调用|中位秒|P95秒|函数异常|
|---|---:|---:|---:|---:|
|CLIP_preprocess_image_text/cold_first_call|1|2.603607|2.603607|0|
|CLIP_preprocess_image_text/warm|11|3.090007|5.877267|0|
|SAM_segmentation/cold_first_call|1|0.620858|0.620858|0|
|SAM_segmentation/warm|11|0.720202|1.004902|0|
|YOLO_detection/cold_first_call|1|0.561571|0.561571|0|
|YOLO_detection/warm|11|0.226716|0.280886|0|
|YOLO_vocabulary_setup/cold_setup|1|3.442117|3.442117|0|
|frontend_model_load/cold_process|1|9.672860|9.672860|0|
|frontend_total/cold_first_frame|1|7.261731|7.261731|0|
|frontend_total/warm|11|4.155363|7.191361|0|
|Hybrid_Astar_planning/warm|2|3.774370|4.758587|0|
|bicycle_state_update/warm|391|0.000026|0.000043|0|
|incremental_relations/warm|10|0.000817|0.002048|0|
|mapping_update/warm|10|0.010918|0.018906|0|
|observed_LiDAR_obstacles/warm|10|0.005116|0.006163|0|
|perception_mapping_planning_total/warm|10|0.192834|4.058486|0|
|rule_candidate_ranking/warm|10|0.000458|0.001497|0|
|tracking_control/warm|10|0.000952|0.002845|0|
|verified_candidate_chain/warm|10|0.120883|2.833800|0|

CPU共享主机，以上不能作为GPU实时性能或与后台GPU主矩阵混成一张性能表。规划只有2调用、控制10调用，不能将P95解释为总体上界。外部阶段包含内部阶段；CLIP含预处理、图像与文本编码，不是纯编码核函数。模型加载和首次调用分开，未清OS缓存，不宣称冷机。

正式GPU队列等待既有三个作业真实退出，原Qwen/旧Adapter/新Adapter使用相同72条不同缓存键请求，各执行72无缓存+72命中复跑；新调用共216。GPU前端12输入独立子进程，采样监测其它CUDA PID，污染则取消独占信用。当前全部GPU性能调用0；queue存在不是执行证据。

原协议001有一组重复排序输入，语义未执行；002去重后216个真实模型配置缓存键预检通过，在cache.get处截获并抛专用异常，未加载模型、未写语义响应。不能计入模型准确率或缓存推理复跑。
