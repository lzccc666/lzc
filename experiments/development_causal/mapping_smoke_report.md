# 开发集过去LiDAR重建统计

1场景、1log、3帧；缺测帧留在全部GT分母。

|条件|匹配数/GT|预测数|召回|匹配精确率|中心误差(m)|身份变化/连续匹配|
|---|---:|---:|---:|---:|---:|---:|
|one_to_one_depth_quality|0/2|0|0.0000|unknown|unknown|0/0|
|one_to_one_depth_robust_motion|0/2|0|0.0000|unknown|unknown|0/0|
|one_to_one_no_expiry|0/2|0|0.0000|unknown|unknown|0/0|
|one_to_one_timestamp|0/2|0|0.0000|unknown|unknown|0/0|
|original_fusion|0/2|0|0.0000|unknown|unknown|0/0|
|single_frame|0/2|0|0.0000|unknown|unknown|0/0|
|timestamp_only|0/2|0|0.0000|unknown|unknown|0/0|

官方GT沿用原sample时刻，较相机晚约33–42ms；动态误差作为时间近似单列，不称完整严格同步GT

仅开发集对象建图，不提供检索、锁定、路径、正确到达或独占实时性能信用。配对log聚类区间、缺测分组及全部原始计数见report.json。goal_complete=false。
