# Agentic GRPO Retail

面向 τ-bench retail 的长链路工具智能体后训练项目，包含 Retail SFT、
Vanilla GRPO、State-GRPO、Constraint-GRPO、CS-GRPO 和固定协议独立评测。

## 当前结果

72B-user 确认性实验已经完成：

- 用户模拟器：`Qwen2.5-72B-Instruct-AWQ`，4 卡 TP
- 策略：`Qwen2.5-7B-Instruct`，2 卡 FSDP/rollout
- 四种 GRPO 方法各 8 updates，评测 step 4/8
- SFT 加 8 个 RL checkpoint，共 1008 条评测轨迹
- unseen `pass^1` 最佳：State step 4/8、Constraint step 8，均为 `0.375`
- unseen 任务覆盖率最佳：State step 8，`0.750`

完整报告：

- `agentic-grpo-longhorizon/experiments/retail_confirmatory_72b_user/final_summary/report.md`
- `agentic-grpo-longhorizon/experiments/retail_confirmatory_72b_user/final_summary/summary.json`

这些结果是固定本地任务集上的工程证据，不是官方 τ-bench 排名。

## 目录

```text
agentic-grpo-longhorizon/   Retail 代码、配置、脚本和实验
tau-bench/                  Retail 环境
verl/                       GRPO 训练框架
models/                     7B 策略和 72B-AWQ 用户模拟器
airline/                    只读 Airline 历史归档
```

## 环境

```bash
bash setup.sh
conda activate agentrl
```

核心版本为 Python 3.10、PyTorch 2.7、Transformers 4.51、vLLM 0.9.2、
veRL 0.6.1。

## 验证

```bash
cd /home/liuchenyang/agentic-grpo-retail/agentic-grpo-longhorizon
conda run -n agentrl pytest -q \
  src/envs/tests/test_retail_support.py \
  src/envs/tests/test_retail_cs_grpo.py \
  src/evaluation/tests/test_retail_eval.py \
  src/evaluation/tests/test_retail_confirmatory_summary.py
```

训练与评测入口见 `RETAIL_EXPERIMENT_GUIDE.md`。运行模型或训练前必须先检查
GPU 占用，并显式设置 `CUDA_VISIBLE_DEVICES`。
