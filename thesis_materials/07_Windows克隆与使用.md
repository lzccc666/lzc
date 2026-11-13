# Windows克隆与独立资料目录

目标布局：

```text
D:\study\Python.Project\lzc\
├── code\                  # Git代码仓库（含小型实验依据）
└── thesis_materials\      # 独立写作资料副本，可另存笔记
```

运行本包附带的`setup_windows.ps1`会克隆公开仓库、验证代码哈希并复制资料。脚本不安装训练环境、不启动模型、不恢复Goal；已有非空陌生目录会报错并保留，不清空、不强制覆盖。

在Windows PowerShell执行：

```powershell
git -c core.autocrlf=false clone --config core.autocrlf=false https://github.com/lzccc666/lzc.git "D:\study\Python.Project\lzc\code"
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\study\Python.Project\lzc\code\thesis_materials\setup_windows.ps1"
```

`core.autocrlf=false`仅用于本仓库，使带哈希的源码与索引保持一致。若目录已有仓库，先查看其状态；脚本不会自动reset或覆盖已有修改。已有资料目录仅在受管文件内容与包完全一致时复用，额外笔记不删除。

脚本成功后在独立资料目录产生`WINDOWS_SETUP_RECEIPT.json`，记录本机Git版本、实际commit和校验数量。没有这份实际执行回执，不能声称D盘已经克隆成功。

用VS Code打开代码目录；从独立资料目录的00开始阅读。CSV可用Excel按UTF-8打开；固定版本GitHub链接可精确跳转行号。Windows原生CUDA/图形仿真未验证，如要重跑实验先读服务器复现说明与依赖边界。
