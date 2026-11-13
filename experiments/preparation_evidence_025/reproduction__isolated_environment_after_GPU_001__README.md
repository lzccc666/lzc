# 原GPU队列之后的新Python环境复演

此队列当前仅等待。准备清单、CPU导入预检和Ruff通过，不代表依赖安装成功或新环境已实际导航。

队列原进程PID：425561；/proc启动时刻：1383255247。先核对启动身份，再读取结果；不要重复启动，也不要因观察工具超时而重启。

## 固定顺序

1. 等待原消融、兼容审核、故障、性能和增量执行审核六个进程全部退出，并确认GPU没有计算进程。原实验失败不改写成成功，前置终态单独记录。
2. 新Python环境复制29494文件，原始内容合计1516087988字节；CPU版torch2.0.1、torchvision0.15.2从官方轮子下载并核对官方索引给出的SHA-256。安装前至少11GB空闲，安装中至少6GB；环境与轮子最多5GB。
3. 使用新环境独立的Python包目录和复制解释器，实际执行原390文件发布包的config00-positive0、seed17完整CPU导航。四项结果、实际传感器变化、整车连续安全、方向、身份和至少2秒停车保持分别独立核验。
4. 推理结束后才为统计添加旧450主实验的只读数据入口，以新解释器重算450记录主表，并逐字节比较report.json、audited_trials.jsonl、trials.csv与生成源码。

只有上述实际完成且通过，才生成本目录complete.json。installed.json仅证明依赖安装和基础解释器检查；不能代替导航结果。失败保留interrupted.json或incomplete.json，不自动重试。

## 隔离范围与限制

源码与Python第三方包独立复制，模型、场景资产只读引用；基础Python标准库和原生图形/C++系统库共享并明确声明。这是新Python环境，不能称为新操作系统、新GPU训练环境或新增独立场景。

PyOpenGL采用已实际使用的图形运行时3.1.7并复制到新环境；初版清单001保留但未安装。分割包从原可编辑安装源单独复制，排除.pth和editable finder；推理时核对14个关键依赖的真实路径。

沿用现有可工作的OpenCV4.13.0.92、NumPy1.26.4组合，其包元数据的NumPy>=2声明存在不满足项，清单显式记录；pyrender元数据固定PyOpenGL3.1.0，但实际运行采用声明的3.1.7。不能声称pip依赖检查全部通过，能否运行以新环境实际执行结果为准。

小任务沿用已声明的停车保持开发扩展，不回填主450，不增加新任务计数，不作为GPU性能证据。正式封存测试仍未开启，论文正文暂缓。

## 文件与后续处理

脚本位于source_bundle：isolated_environment_after_GPU_001.py、isolated_cpu_environment_pack_002.py、prepare_isolated_environment_navigation_001.py、standalone_cpu_runner_003.py、isolated_environment_navigation_audit_001.py。

安装产物：reproduction/isolated_cpu_environment_002/；实际任务：reproduction/isolated_cpu_executed_001/。此目录保留等待release、逐子进程日志和终态。新辅助脚本尚未加入最终Git导出，待实际核验后统一同步。

准备核验：source_bundle/isolated_environment_preparation_verification_001.json。全局仍保持goal_complete=false。
