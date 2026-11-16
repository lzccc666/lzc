# 复现命令（当前交付）

以下命令在当前环境执行，均指向已有证据，不会读取最终 GT 进入模型推理。

```bash
cd /root/autodl-tmp/lizichong
PYTHONPATH=$PWD:/autodl-fs/data/third_party/carla_0.9.16/python_site \
  /root/miniconda3/envs/conceptgraph/bin/python -m pytest tests -q

# 查看封存结果
python -m json.tool outputs/project_v2/thesis_completion_v1/public_experiment_completion_20261115/json/delivery_index_101.json

# 查看三线表和图表
ls outputs/project_v2/thesis_completion_v1/public_experiment_completion_20261115/{figures/sealed_public_078,tables/sealed_office_097}
```

GitHub 推送（需在已配置 GitHub 凭据的机器执行）：

```bash
cd /root/autodl-tmp/lzc_sync_101/repo
git push origin HEAD:refs/heads/codex/thesis-completion-20261116
```
