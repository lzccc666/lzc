# 当前工程核验范围（相机时间与地图补充）

项目标准命令`python -m pytest -q tests`实际321通过、5依赖警告，记录tests_full_028.log。原机两个真实crop集成测试通过；不携带数据的公开环境须明确skip。相机时间新增10项边界测试只验证插值规则与隔离性，不增加真实人工记录。

completion/navigation/tests范围Ruff全部通过（ruff_completion_043.log），git diff --check通过（git_diff_check_029.log）。广域042额外包含未改动scenegraph/build_scenegraph_cfslam.py，仍10项旧问题；前次更广扫描还含slam上游共17项，历史证据保留，不能将范围检查称全仓库Ruff通过。

门户008真实桌面和手机42状态行、82证据链接均可读取、无溢出或JS错误，所有请求GET，人工记录0。相机时间38场景1508帧21112方法帧条件独立重算；地图1272条件/84指标/210区间复算，三处浮点边界明细差异和首次严格失败保留，路线级车身结果一致，未改判原违规。全部是开发评价，不是车辆执行或封存验收。

原无路径pytest误收集第三方/历史复制测试导致104收集错误的tests_full_025.log仍保留；未修改历史文件凑通过。所有运行结束后的最终验收仍缺，goal_complete=false。
