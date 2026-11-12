# 相机时刻中心插值：固定开发集合敏感性

38场景11log1508帧，原5723个GT观测全部保留；5311可插值、412缺前驱仍用原中心。尺寸、姿态、可见集合不变。这是中心参考模型敏感性，不是精确相机时刻真值或封存主表。

|评价中心|方法|召回|匹配精确率|中心误差(m)|
|---|---|---:|---:|---:|
|official_sample_proxy|original_fusion|17.229%|16.488%|1.6956|
|official_sample_proxy|one_to_one_depth_quality|30.107%|43.676%|1.3026|
|official_sample_proxy|one_to_one_depth_robust_motion|30.054%|44.046%|1.2824|
|camera_center_interpolation_proxy|original_fusion|17.351%|16.605%|1.6440|
|camera_center_interpolation_proxy|one_to_one_depth_quality|30.334%|44.005%|1.2731|
|camera_center_interpolation_proxy|one_to_one_depth_robust_motion|30.281%|44.379%|1.2529|

深度质量相对原融合在插值参考下召回差+12.983pp（11log配对95%CI[8.205,18.200]pp）、中心误差差−0.37094m（CI[−0.52945,−0.24141]m）。该定位差异在这一敏感性条件仍保留；身份变化率差区间跨0。

同一深度质量方法改评价中心后召回只+0.227pp（CI[−0.023,0.727]pp），中心误差−0.02953m（CI[−0.05285,0.00374]m），区间跨0，不声称实测提高。不能据此把412fallback或非线性运动近似消除。

全部21112个方法×帧×评价条件、逐对象中心/来源、指派/身份和区间独立重算通过。未新调用模型或改变冻结标签，真实人工仍不足，goal_complete=false。
