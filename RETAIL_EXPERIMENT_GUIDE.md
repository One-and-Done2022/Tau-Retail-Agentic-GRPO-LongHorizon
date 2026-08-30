# τ-bench Retail GRPO 实施指南

## 1. 项目目标

本项目实现 τ-bench retail 的完整后训练闭环：

```text
Retail 环境与工具
→ Retail SFT
→ Vanilla GRPO
→ State-GRPO
→ Constraint-GRPO
→ CS-GRPO
→ 固定协议评测
```

当前正式实验以 `Qwen2.5-7B-Instruct` 策略和
`Qwen2.5-72B-Instruct-AWQ` 用户模拟器为主。Airline 历史材料仅保存在
`airline/`，不属于当前运行路径。

## 2. 主要入口

```text
agentic-grpo-longhorizon/configs/       Retail 配置
agentic-grpo-longhorizon/scripts/       数据、训练和评测脚本
agentic-grpo-longhorizon/src/           Retail 环境适配与评测代码
agentic-grpo-longhorizon/experiments/   Retail 数据、checkpoint 和报告
tau-bench/                              Retail benchmark 环境
verl/                                   GRPO 训练框架
models/                                 本地模型
```

进入项目后先执行：

```bash
cd /home/liuchenyang/agentic-grpo-retail
git status --short --branch
nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
```

## 3. 数据纪律

Retail SFT 数据必须来自环境验证成功的完整轨迹：

```text
retail task
→ 教师策略与用户模拟器交互
→ τ-bench 执行工具并评分
→ 过滤失败、截断和污染轨迹
→ 固定 seen/unseen 划分
→ Render-Twice-Diff loss mask
```

- SFT 和 GRPO 训练只使用 train split。
- dev/test 不得用于训练标签、奖励构造或规则定制。
- 超长轨迹不得静默截断。
- system、user 和 tool observation token 不参与 policy loss。

## 4. 方法定义

### Vanilla GRPO

只使用 τ-bench 原生终局成功奖励和轨迹级组内 advantage。

### State-GRPO

在 Vanilla 基础上，计算工具执行前后的状态进展：

```text
r_progress(t) = progress(s_t, goal) - progress(s_(t-1), goal)
```

进展信号仅用于环境侧训练信用，不进入策略输入。

### Constraint-GRPO

对身份认证、订单所有权、订单状态、商品资格、支付方式、用户确认及重复写
操作等 Retail 前置条件计算约束成本。

### CS-GRPO

联合轨迹级 outcome advantage 与状态/约束步骤 advantage：

```text
A_total(t) = A_traj + beta * A_step(t)
```

步骤项只分配给对应 tool-call token；无效、重叠或越界 span 必须硬失败。

## 5. 固定实验协议

正式 72B-user 确认性实验：

- SFT 共同起点：`experiments/retail_sft/lora_smoke_merged`
- GRPO 方法：Vanilla、State、Constraint、CS
- train tasks：Retail train `0–7`
- 训练预算：8 updates，group size 4
- checkpoints：step 4、step 8
- seen eval：train `0–7`，每任务 4 次
- unseen eval：全部 20 个 dev tasks，每任务 4 次
- 用户模拟器：`Qwen2.5-72B-Instruct-AWQ`

最终结果：

```text
agentic-grpo-longhorizon/experiments/retail_confirmatory_72b_user/final_summary/
```

不要重复已经有 `COMPLETED` 且门禁 `passed=true` 的运行。

## 6. 阶段门禁

1. Retail reset、tool、termination 和 native reward 测试通过。
2. tokenizer、chat template、tool call 和 vLLM smoke 通过。
3. SFT 数据验证、LoRA、merge、HF/vLLM reload 通过。
4. Vanilla 1-step 的 rollout、reward、loss、gradient、checkpoint 和部署通过。
5. 状态进展、约束规则和 tool span 单测通过。
6. State、Constraint、CS 1-step 通过。
7. 固定预算多步训练与统一 checkpoint 评测通过。

每阶段记录命令、配置、GPU、输入输出、退出码、关键指标、失败原因和验证
边界。smoke 只证明工程闭环，不作为正式性能结论。

## 7. 结果解释

- `pass^1` 是单条轨迹平均成功率，作为主要成功指标。
- 当前报告中的任务覆盖率表示每任务 4 次采样中至少成功一次。
- 约束违例率的分母是高风险写操作数，不是轨迹数。
- seen 结果用于判断训练是否生效；泛化判断以固定 unseen dev 为主。
- 污染轨迹、系统错误和置信区间必须与主指标一起披露。
