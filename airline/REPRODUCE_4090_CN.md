# 4×RTX 4090 复现与启动手册

本文档记录在 4×RTX 4090 24GB 机器上已经实际验证的启动流程。它用于完成项目理解和工程闭环，不等同于 README 中 2×A800 80GB、24K context、72B-AWQ 用户模拟器的原始实验规模。

## 1. 这个项目位于后训练流程的哪一段

项目覆盖两个环节：

1. 冷启动 SFT：用成功的多轮工具调用轨迹训练 Qwen2.5-7B-Instruct 的 LoRA。
2. 领域强化学习：在 τ-bench airline 环境中使用 GRPO，并比较 Vanilla、Turn-Discount、PRM-Lite、LATA 和 PRM-Lite+LATA。

项目不包含 Base 预训练、多教师在线蒸馏、DPO/SimPO 等最终偏好对齐阶段。

## 2. 仓库与运行资产

仓库根目录：

```bash
cd /home/liuchenyang/agentic-grpo-longhorizon
```

训练代码目录：

```bash
cd /home/liuchenyang/agentic-grpo-longhorizon/agentic-grpo-longhorizon
```

主要运行资产：

- Base 模型：`../models/Qwen2.5-7B-Instruct`
- SFT 数据：`experiments/sft_collect_airline/train.jsonl`
- 4090 SFT adapter：`experiments/sft_lora_4090_smoke`
- 合并模型：`experiments/sft_lora_4090_smoke_merged`
- GRPO 数据：`experiments/vanilla/train.parquet`、`val.parquet`
- GRPO 日志：`experiments/4090_smoke/training.log`
- GRPO rollout：`experiments/4090_smoke/rollouts/1.jsonl`
- GRPO checkpoint：`experiments/4090_smoke/checkpoints/global_step_1`
- GRPO 合并模型：`experiments/4090_smoke/hf_step_1`
- 7B 用户模拟器评测：`experiments/4090_smoke/eval/eval_report.json`
- 72B-AWQ 用户模拟器评测：`experiments/4090_smoke/eval_72b_user/eval_report.json`

`models/`、Hydra `outputs/`、checkpoint 和 safetensors 已加入 Git 忽略规则，避免误提交几十 GB 权重。

## 3. 环境

首次安装：

```bash
cd /home/liuchenyang/agentic-grpo-longhorizon
bash setup.sh
conda activate agentrl
```

已验证的核心版本：Python 3.10、PyTorch 2.7.0+cu126、Transformers 4.51.3、vLLM 0.9.2、veRL 0.6.1、Ray 2.56.1。

当前训练使用 FSDP 和 PyTorch SDPA。DeepSpeed 与 FlashAttention 不是 4090 smoke 的必需依赖；原 A800 配置仍需要单独评估 FlashAttention 2。

如果 7B 模型尚不存在，公开模型可直接下载，不要在聊天或脚本中写 token：

```bash
cd /home/liuchenyang/agentic-grpo-longhorizon
conda activate agentrl
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct \
  --max-workers 16
```

正式对齐原实验的用户模拟器时，将模型 ID 和目录替换为 `Qwen/Qwen2.5-72B-Instruct-AWQ`。本机已下载并校验 11 个权重分片，总权重约 38.7 GiB。

## 4. SFT：建立工具调用冷启动能力

### 4.1 检查数据和 loss mask

```bash
cd /home/liuchenyang/agentic-grpo-longhorizon/agentic-grpo-longhorizon
conda activate agentrl
python scripts/train/sft/inspect_sft_dataset.py \
  --train-jsonl experiments/sft_collect_airline/train.jsonl \
  --tokenizer ../models/Qwen2.5-7B-Instruct \
  --max-length 4096 \
  --num-show 2
```

为什么：SFT 只能对 assistant 文本和 tool call 计算 loss，system/user/tool observation 必须 mask。错误的 mask 会让模型学习复述用户或工具返回值。

4090 smoke 实测：45 条原始轨迹保留 23 条，22 条因超过 4096 tokens 被过滤；平均 2252 tokens，其中 assistant label 占 44.4%。

### 4.2 训练 LoRA

```bash
bash scripts/train/sft/run_4090_smoke.sh
```

为什么：先用成功轨迹给策略提供对话格式、工具 schema 和基本任务行为，避免 RL 从完全随机的工具使用开始探索。

实测：1 epoch 配置执行 5 个优化步，26.5 秒，最终 train loss 0.8036。

### 4.3 合并为 vLLM 模型

```bash
python scripts/train/sft/merge_lora.py \
  --base ../models/Qwen2.5-7B-Instruct \
  --adapter experiments/sft_lora_4090_smoke \
  --out experiments/sft_lora_4090_smoke_merged
```

为什么：独立 vLLM 服务可以直接加载合并后的 Hugging Face 模型，不依赖训练时的 PEFT 包装。

## 5. GRPO：跑通领域强化学习闭环

### 5.1 构建 parquet

```bash
python scripts/train/grpo/build_grpo_parquet.py \
  --seen-task-ids-from experiments/sft_collect_airline/summary.json \
  --output-train experiments/vanilla/train.parquet \
  --output-val experiments/vanilla/val.parquet
```

为什么：veRL 从 parquet 读取 system prompt、task id 和 interaction 参数。训练集是 40 个 seen tasks，验证集包含全部 50 个任务并标记 10 个 unseen tasks。

### 5.2 启动用户模拟器

低成本 smoke 使用 7B 代替原实验的 72B-AWQ：

```bash
CUDA_DEVICES=3 \
PORT=8001 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=4 \
GPU_MEM_UTIL=0.80 \
MODEL_PATH=../models/Qwen2.5-7B-Instruct \
bash scripts/vllm_server/7b.sh
```

为什么：τ-bench 的“用户”也是一个 LLM。它根据隐藏任务持续回应策略模型，因此训练 rollout 不是静态问答，而是真实的多轮交互。

最终独立评测已进一步使用原实验的 72B-AWQ 用户模拟器。4×4090 需要用 GPU 0、1 做 TP=2；4K context、单并发和 eager 模式用于控制显存：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
conda run -n agentrl --no-capture-output \
python -m vllm.entrypoints.openai.api_server \
  --model ../models/Qwen2.5-72B-Instruct-AWQ \
  --served-model-name Qwen/Qwen2.5-72B-Instruct-AWQ \
  --port 8001 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --quantization awq \
  --enforce-eager \
  --disable-custom-all-reduce \
  --enable-prefix-caching \
  --no-enable-chunked-prefill \
  --trust-remote-code
```

实测每卡加载 19.39 GiB 权重，服务稳定后每卡占用约 22.77/24.56 GB，可用 KV cache 为 13,712 tokens。这个配置适合单任务验证，不适合增加并发。

### 5.3 执行一步 PRM-Lite + LATA GRPO

另开终端：

```bash
cd /home/liuchenyang/agentic-grpo-longhorizon/agentic-grpo-longhorizon
bash scripts/train/grpo/run_4090_smoke.sh
```

配置使用 GPU 0、1，关键缩放如下：

- FSDP + vLLM TP=2
- `group_size=8`
- 单个 prompt batch
- 4096 总上下文
- 3 个 assistant turns / 3 个 user turns
- `load_format=safetensors` + `layered_summon=true`
- vLLM `gpu_memory_utilization=0.45`
- SDPA、关闭 fused kernels
- 只执行 1 个 training step

为什么启用 `layered_summon`：普通 FSDP `state_dict()` 在把 LoRA 权重同步给 vLLM 时出现 24GB 峰值 OOM；预加载 base weights 后逐层汇集 LoRA，可以把峰值控制在 4090 可运行范围内。

最终实测：8 条轨迹、约 19,890 tokens、总耗时 250.75 秒，checkpoint 成功保存。`pg_loss=4.01e-6`、`grad_norm=2.50e-5`，LoRA B 张量已确认非零。

同时，8 条轨迹分数几乎一致，`score/std≈2.5e-10`，说明该极小任务组仍发生奖励饱和。它证明系统闭环和项目问题可以复现，但不能证明模型性能已经提升。

### 5.4 合并 GRPO checkpoint

```bash
python scripts/test/merge_fsdp_to_hf.py \
  --actor-dir experiments/4090_smoke/checkpoints/global_step_1/actor \
  --output-dir experiments/4090_smoke/hf_step_1
```

为什么：veRL 保存的是两个 FSDP rank shard，不能直接交给常规 Hugging Face/vLLM 推理。该步骤先恢复完整参数，再把 196 个 LoRA 矩阵合并进 base weights，最终生成可独立部署的模型目录。

实测产物是一个约 15.23 GB 的 `model.safetensors`；`AutoConfig`、`AutoTokenizer` 和 vLLM 离线加载均已通过。

## 6. 独立评测

### 6.1 启动策略服务

GPU 2 上启动合并后的 GRPO step 1 策略。独立 τ-bench 对话会超过 4096 tokens，因此使用 8192：

```bash
CUDA_DEVICES=2 \
PORT=8000 \
MAX_MODEL_LEN=8192 \
GPU_MEM_UTIL=0.80 \
MODEL_PATH=experiments/4090_smoke/hf_step_1 \
bash scripts/vllm_server/7b_sft.sh
```

`configs/eval/eval_4090_smoke.yaml` 的 API `model_name` 保持 served model name，`tokenizer_name_or_path` 则指向本地 `experiments/4090_smoke/hf_step_1`，用于离线 token 统计。

### 6.2 运行一个任务

```bash
conda activate agentrl
python scripts/eval/run_baseline_eval.py \
  --config configs/eval/eval_4090_smoke.yaml
```

实测 task 0：6 turns、3 次工具调用、无系统错误，reward=0，故 pass^1=0。失败是策略能力结果，不是服务、checkpoint 转换或上下文错误。

### 6.3 使用 72B-AWQ 用户模拟器评测

保持 GRPO 策略服务运行，并按 5.2 节启动 72B 用户模拟器：

```bash
python scripts/eval/run_baseline_eval.py \
  --config configs/eval/eval_4090_72b_user.yaml
```

实测 task 0：28.6 秒、6 turns、3 次工具调用、无系统错误，reward=0、pass^1=0。报告保存在 `experiments/4090_smoke/eval_72b_user/eval_report.json`。

## 7. 当前验证边界

- 已验证：环境、7B 推理、数据 mask、LoRA SFT、LoRA 合并、parquet、τ-bench 多轮 rollout、PRM-Lite、LATA、FSDP/vLLM 权重切换、反向更新、checkpoint、FSDP checkpoint 合并、GRPO 模型部署和独立评测。
- 4090 smoke 未复现论文级指标：只训练一步，且只保留 23 条 4K SFT 轨迹。
- GRPO 训练 smoke 的 rollout 使用 7B 用户模拟器以控制成本；最终独立评测已验证 Qwen2.5-72B-Instruct-AWQ 可在 2×4090 上以 TP=2、4K、单并发运行。
- 训练结束时 Ray 的 DataLoader worker 偶尔在清理阶段打印 `Killed`，但主训练退出码为 0，rollout、指标和 checkpoint 均已落盘。
- 评测配置通过 `tokenizer_name_or_path` 从本地合并模型加载 tokenizer，API 模型名与离线 tokenizer 路径相互独立。

## 8. 简历中如何准确描述

建议项目名：

> 面向长链路工具调用智能体的 SFT-GRPO 后训练与 4×RTX 4090 工程复现

简历可以拆成三条：

> - 在 τ-bench airline 多轮工具调用环境中复现 SFT→GRPO→部署→评测闭环，完成成功轨迹 loss mask、LoRA 冷启动、8-sample on-policy rollout、PRM-Lite 过程奖励与 LATA 长度归一化。
> - 基于 veRL、FSDP 与 vLLM 适配 4×RTX 4090 24GB 环境，通过 TP=2、参数/优化器 offload、safetensors 预加载与 layered summon 解决 LoRA 权重同步峰值 OOM，完成约 19,890 tokens 的单步反向更新并保存可部署 checkpoint。
> - 将 FSDP checkpoint 合并为 Hugging Face 模型并部署独立 OpenAI-compatible API；在 2×4090 上运行 Qwen2.5-72B-Instruct-AWQ 用户模拟器，完成真实多轮工具调用评测与错误率、turn、tool-call、pass^k 报告落盘。

面试时按“问题—方案—证据—边界”讲：

1. 问题：长链路 GRPO 的组内终局奖励容易饱和，且 24GB 卡在 FSDP→vLLM 权重同步时 OOM。
2. 方案：SFT 冷启动提供基本工具行为；PRM-Lite 产生逐轮局部信号；LATA 用 `√L` 归一化减轻长回复稀释；工程上使用分层 LoRA 汇集降低峰值。
3. 证据：SFT 5 步 loss=0.8036；GRPO 8 条 rollout、约 19,890 tokens、checkpoint 与非零 LoRA 更新均已验证；7B/72B 两种用户模拟器的独立评测均无系统错误。
4. 边界：一步 smoke 的 reward/pass^1 为 0，且组内 `score/std≈2.5e-10`，只能证明训练与部署链路有效，不能证明模型能力提升。

如果要把项目从“工程复现”升级为“有性能结论的实验项目”，下一阶段应固定 checkpoint，至少评测 10 个任务、每任务 4 次采样，并加入 SFT baseline 与 GRPO step 1 的同配置对照；随后再增加训练步数和随机种子。优先补这些统计证据，不要先堆更多算法名。

一句话版本可以写：

> 在 τ-bench airline 多轮工具调用环境中复现 SFT→GRPO 后训练链路；基于 veRL、FSDP 与 vLLM 实现 7B 策略的 on-policy rollout 和 LoRA 在线更新，针对 24GB RTX 4090 通过 TP=2、参数/优化器 offload、safetensors 预加载与 layered summon 消除权重同步 OOM，并复现长链路 GRPO 的组内奖励饱和现象。

在没有完成多 checkpoint、N=4 全量评测之前，不应把 README 中 `+37%`、`pass^1=0.240` 写成自己重新实验得到的结果。可以将其标为“原项目报告结果”，把自己的结果单独列为“4×4090 工程复现”。
