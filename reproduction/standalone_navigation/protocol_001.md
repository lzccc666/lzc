# 隔离源码完整任务复演001

目标是核验当前交付源码能在独立目录完成一个真实感知—建图—规划—控制任务。固定选择既有冻结50任务中的config00-positive0、seed17；选择在运行前写入本协议，不依运行成败更换。此处验证交付可运行性，不增加独立场景、主450样本、新封存测试、人工记录或GPU性能信用。全部失败和中断保留。

从source_checkpoint_033/checkpoint.json逐项复制390业务源码及配置，不复制原工作区对象图、图像、缓存、车辆轨迹或推理结果。在新目录复制冻结任务JSON（仅环境/隔离评价使用），通过独立入口运行相同live_navigation_worker。代码包与操作入口冻结后才能启动。

外部依赖均显式声明：复用现有conceptgraph Python环境和隔离OSMesa/PyOpenGL运行库，不声称全新操作系统或全新虚拟环境；三个真实前端权重通过只读使用的checkpoint路径链接合法本地模型，逐项核对SHA；ToyCar资产复制到新依赖目录并核对原SHA。启动入口只替换实际资产文件路径变量，内容SHA与原资产完全一致；不改原业务源码、任务、车辆尺寸、感知、选择或安全阈值。

执行进程移除原项目根目录的import搜索项，将隔离源码根置首，运行前后核验conceptgraph所有已加载模块的实际__file__均来自隔离根。CUDA_VISIBLE_DEVICES为空，torch与BLAS/OSMesa单线程，真实YOLO/MobileSAM/CLIP、真实软件渲染，禁止读取旧帧代替本次观测。固定0.1秒自行车积分，每0.5仿真秒重新渲染感知；不是实际墙钟异步Qwen实验。

预算：准备600秒、执行900秒墙钟/30仿真秒/80前端/500MB、最低6GB余量。提交一条自动run命令，运行至任务终态或预算，保存命令、原始初态、实际传感器、检测、crop、逐帧对象/关系/候选/路径和控制记录。自动操作不称人工标注。独立核验源码导入、全输入SHA、整车连续包络、控制积分、方向及正确对象/区域/速度/航向/保持；不能只据运行端ARRIVED或自报audit判断通过。

所有新增文件均位于outputs/project_v2/thesis_completion_v1/reproduction/standalone_cpu_delivery_001/及本轮source_bundle/task_protocol中。原工作区Git索引和既有GPU冻结依赖不变。仍缺GPU消融/故障/性能、人工标签和合格新封存时保持goal_complete=false；本阶段不写论文正文。
