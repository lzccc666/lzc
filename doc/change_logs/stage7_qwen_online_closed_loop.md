# 阶段7：Qwen在线闭环接入

- 实施日期：2026-11-10
- 项目根目录：`/root/autodl-tmp/lizichong`
- 在线输出目录：`outputs/project_v2/qwen_online_closed_loop`
- 模型：`checkpoint/Qwen2-VL-2B-Instruct`
- 适配器：`outputs/project_v2/qwen_eval/finetune_v2/best_adapter`
- 状态：代码闭环与CPU可验证回归完成；当前执行容器无可见CUDA，双场景真实模型回放尚未执行

## 1. 闭环范围

本阶段把微调Qwen接入逐帧导航主循环的三个语义环节：

```text
自然语言指令
  -> 异步Qwen query_parse
当前稳定3D对象 + 当前帧真实2D裁剪
  -> 异步Qwen object_attributes
Stage2已完成A*和碰撞验证的候选
  -> 异步Qwen candidate_rerank
  -> 连续帧锁定状态机
  -> 重新执行Stage2目标点搜索、A*和碰撞检查
```

Qwen不生成占据图、不声明可达性、不直接写路径，也不覆盖检测器类别或几何运动判定。空间边仍由Stage3结构化几何规则逐帧更新，Qwen只读取关系元数据辅助候选重排。

## 2. 对象属性在线回写

端到端回放从 `StreamlinedMapper` 对象保存的 `image_idx`、`color_path` 和 `xyxy` 提取当前帧真实裁剪，写入独立的 `qwen_object_crops/<scene>`，不修改源图。对象至少累计3次观测才可提交；每帧最多提交1个对象，最多2个属性任务并发。

有效Qwen输出按对象轨迹身份缓存并在后续帧合并：

- 检测器 `object_tag` 保持权威，Qwen类别只扩展 `possible_tags`；
- `summary` 更新caption；`colors` 和 `body_type` 写入 `metadata.attributes`；
- 几何判定为dynamic时，Qwen不能降级为static；Qwen只可把非dynamic对象升级为dynamic，或补全unknown；
- 置信度低于0.60、schema/fallback、对象消失、检测类别变化或轨迹身份变化时不采纳；
- 默认同一节点最短刷新间隔30帧，失败后10帧再试，且必须存在新的可读裁剪。

## 3. 候选安全重排

原逻辑在锁定目标规划后只把该单一目标交给Qwen，无法形成真实Top-K重排。本阶段增加独立的Stage2候选验证通道：不指定锁定节点，先对Top-K执行目标点生成、连通性、A*与碰撞检查，再只把可达候选提交给Qwen。

Qwen推荐必须满足：原始schema合法、无fallback、置信度至少0.60、ranking完整覆盖已验证集合、推荐节点仍在当前候选中，且提交后候选集合变化比例小于0.34。有效推荐只提升该候选的语义分数；换目标仍需通过 `better_target_score_margin` 和默认连续3帧持久化。切换后重新调用Stage2规划，旧路径不会直接复用。

异步任务改为任务级有效期：query parse只绑定query版本；对象属性绑定对象身份；候选重排绑定query与已验证候选集合。普通对象位置变化不会使约1.5秒返回的有效结果必然作废，但候选消失或集合显著变化会拒绝迟到结果。

## 4. 模型部署配置

`incremental_navigation.yaml` 新增默认关闭的 `qwen_online` 配置。专用入口显式开启后使用：

- `prompt_profile=finetune_v2`；
- `torch_dtype=bfloat16`，GPU不支持BF16时显式回退FP16并写入报告；
- `max_image_side=448`；
- 确定性生成、最长192个新token；
- 独立cache、audit、crop和双场景输出目录。

旧Stage5入口仍默认 `online.qwen_enabled=false`，不会加载模型，也不会改变既有规则回放结果。

## 5. 运行与验收

专用入口固定执行 `scene-0916`、`scene-0003`，只消费已有RGB-D、位姿和缓存检测，不重新运行YOLO、SAM或深度生成：

```bash
conda run -n conceptgraph python -m conceptgraph.scripts.run_qwen_online_closed_loop
```

入口强制至少一个可见CUDA设备，检查基础模型与adapter文件，结束前排空异步任务，并要求query parse、对象属性和候选重排三类结果均至少实际采纳一次。最终写入：

```text
outputs/project_v2/qwen_online_closed_loop/qwen_online_closed_loop_report.json
```

当前容器执行入口得到明确错误 `Qwen online closed loop requires at least one visible CUDA GPU`，因此没有伪造双场景真实Qwen吞吐、延迟或通过报告。代码与配置需在用户已配置GPU的目标服务器运行上述命令完成最终实机验收。

## 6. 已完成验证

- 对象裁剪隔离、当前帧证据选择和节点元数据挂接测试通过；
- 异步属性回写使“红色小汽车”从无候选变为可检索候选；
- 已验证Top-K推荐经状态机切换后重新规划；
- 推荐节点消失时迟到结果被拒绝；
- Qwen关闭、超时、异常、缓存命中、关闭排空与事务恢复兼容；
- 新增闭环相关测试15项通过；
- 项目 `tests/` 全量为148 passed、2条既有torchvision Beta警告；
- 目标Python文件Ruff与 `git diff --check` 通过。

无路径限定的仓库根 `pytest` 会额外收集 `gradslam`、`nature-skills` 的第三方测试，并因多个顶层 `tests` 包冲突在收集期失败，不属于项目测试门禁。
