# Agentic GRPO Retail 项目规则

本项目固定在 `/home/liuchenyang/agentic-grpo-retail`，唯一研究场景为
τ-bench retail。目标是在统一预算下训练和评测 SFT、Vanilla GRPO、
State-GRPO、Constraint-GRPO 与 CS-GRPO。

## 1. 执行原则

- 默认使用简体中文；命令、路径、配置键和报错保持原文。
- 从仓库当前状态继续，不重复已通过门禁的训练或评测。
- 每项工作按“检查—最小修改—验证—记录”闭环推进。
- 只修改当前任务必需内容，不重构无关模块。
- 严格区分 smoke、正式实验、已验证结果和未验证推断。
- 未经明确要求，不提交 commit、创建 branch、推送远程或改写历史。

## 2. 目录边界

- Retail 主代码：`agentic-grpo-longhorizon/`
- τ-bench retail 环境：`tau-bench/`
- veRL：`verl/`
- 本地模型：`models/`
- Airline 历史归档：`airline/`

`airline/` 不参与当前安装、训练、测试或评测，不得从 Retail 配置中引用。
新的代码、配置、数据和实验产物必须使用 retail 命名并写入 Retail 主目录。

严禁把 token、密码、`auth.json`、私钥或其他凭证写入代码、日志或文档。

## 3. GPU 共享规则

每次启动模型、Ray、vLLM、SFT、GRPO 或评测前必须重新检查：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
```

- 不终止、不迁移、不干扰其他用户或未知来源进程。
- 禁止为抢卡执行 `kill`、`pkill` 或 GPU reset。
- 只使用确认空闲且显存满足要求的 GPU，并显式设置 `CUDA_VISIBLE_DEVICES`。
- 在运行元数据中记录物理 GPU、职责、显存状态、端口和启动时间。
- 正式对比必须固定 GPU 数、训练预算、采样参数、任务集合和用户模拟器。
- 资源不足时优先转做离线验证，不得把缩放 smoke 当成性能结果。

## 4. 研究定义

CS-GRPO（Constraint-aware State-transition GRPO）由以下信号组成：

1. τ-bench 原生终局 outcome reward。
2. 工具执行前后结构化状态的进展信号。
3. 基于 Retail policy、工具 schema 和环境状态的约束成本。
4. 仅分配给对应 tool-call token 的步骤级 advantage。

不得修改 ground truth、放宽成功条件、把隐藏目标暴露给策略，或针对具体
task ID 编写奖励规则。dev/test 数据不得用于 SFT 标签或训练奖励构造。

## 5. 实验与门禁

正式矩阵至少包含：

```text
SFT
Vanilla GRPO
State-GRPO
Constraint-GRPO
CS-GRPO
```

长训练前必须确认：Retail baseline、SFT merge/reload、Vanilla 1-step、步骤
信用单测、约束单测、checkpoint merge/reload 和 GPU 分配均已通过。

统一评测至少报告：`pass^1`、任务覆盖率、seen/unseen success、约束违例率、
高风险工具错误率、恢复率、系统错误率、turns、tool calls、tokens 和延迟。
所有方法必须固定 task IDs、seeds、temperature、top_p、max turns、context 和
用户模拟器。

## 6. 验证与安全

- 能运行相关测试、配置解析、shell 语法和 Python 编译检查时必须实际运行。
- 不覆盖有价值的 checkpoint、报告或原始轨迹。
- 输出采用独立目录或 append-only 策略。
- 发现失败时保留原因和验证边界，不把部分结果包装成完整结论。
- 开始项目工作时先阅读 `RETAIL_EXPERIMENT_GUIDE.md`。
