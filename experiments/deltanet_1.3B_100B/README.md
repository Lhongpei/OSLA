# OSLA-OSGM dd-decay · 1.3B · 100B tokens

远端 8×A100 80G 启动手册。训练 OSLA-DeltaNet 的 **data-dependent decay** OSGM 变体（340M sweep 里表现最好的那组），在 fineweb-edu sample-100BT 上跑 100B tokens，schedule 对齐 `deltanet_1.3B_100B` baseline。

---

## 1. 前置假设

| 项目 | 要求 |
|---|---|
| GPU | 8× A100 80G（NVLink 互联） |
| 系统 | Linux + CUDA driver ≥ 12.4 |
| 环境 | Miniconda/Anaconda 装在 `$HOME/anaconda3`（可用 `CONDA_HOME` 覆盖） |
| 磁盘 | `/data0` 至少预留 ~1 TB（dataset cache + checkpoints） |
| 网络 | 可达 GitHub + `hf-mirror.com` + wandb.ai |
| 工作路径 | 训练脚本用绝对路径 `/data0/OSLA`。若必须改别的盘，需同时改 `train_osla_osgm_dd_decay_1.3B.sh` 里的四个绝对路径 |

> **40G A100 不建议直接跑**。`seq_len=65536 + full activation ckpt + FSDP(8)` 在 40G 下会 OOM。若只有 40G，见下文 §5 的降档方案。

---

## 2. 一键拉起环境

```bash
# 在目标服务器上：
git clone git@github.com:Lhongpei/OSLA.git /data0/OSLA
cd /data0/OSLA
bash experiments/deltanet_1.3B_100B/scripts/setup_new_server.sh
```

`setup_new_server.sh` 做的事：

1. 拉 OSLA 仓到 `/data0/OSLA`（默认 `main` 分支，已含 dd-decay 所有代码）
2. 拉固定 commit 的 `flame`（`e11e7be7…`）到 `/data0/OSLA/flame`
3. 建 conda env `osla`（python 3.11 + torch 2.5.1/cu124 + triton ≥ 3.0 + datasets / transformers / wandb / hf_transfer 等）
4. `pip install -e .` 本仓 + flame
5. 走 `HF_ENDPOINT=https://hf-mirror.com` 预拉 tokenizer（`fla-hub/delta_net-1.3B-100B`）和整个 `HuggingFaceFW/fineweb-edu sample-100BT`
6. 提示你 `wandb login`

**可选参数**：

```bash
# 跳过数据集预取（会在 step 0 第一次访问时 stream-download，起步较慢但能先启动）
bash experiments/deltanet_1.3B_100B/scripts/setup_new_server.sh --skip-data

# 覆盖默认值
OSLA_ROOT=/mnt/ssd/OSLA CONDA_HOME=/opt/conda \
  bash experiments/deltanet_1.3B_100B/scripts/setup_new_server.sh
```

环境装完后：

```bash
conda activate osla
wandb login   # 贴 API key
```

---

## 3. 启动前自检（强烈建议做）

### 3.1 import 冒烟测试

```bash
conda activate osla
python -c "
from fla.layers.delta_net import DeltaNet
from fla.models.os_delta_net.modeling_os_delta_net import OSDNForCausalLM
from fla.models.os_delta_net.configuration_os_delta_net import OSDNConfig
import json
cfg = OSDNConfig(**json.load(open('experiments/deltanet_1.3B_100B/configs/osla_osgm_dd_decay_1.3B.json')))
cfg.num_hidden_layers = 2; cfg.hidden_size = 128; cfg.num_heads = 4; cfg.hidden_ratio = 2
m = OSDNForCausalLM(cfg)
print('OK, tiny params =', sum(p.numel() for p in m.parameters()))
print('osgm_a_proj.bias.mean ≈', [p.mean().item() for n, p in m.named_parameters() if n.endswith('osgm_a_proj.bias')][:2])
"
```

期望：`osgm_a_proj.bias.mean ≈ [6.9, 6.9, ...]`（初始 γ≈0.999，logsigmoid(6.9)≈-0.001）。**如果看到 0.0 说明 `_init_weights` 重置逻辑坏了**，不要开训。

### 3.2 GPU 2-step dry-run

正式开训前先用极少的 step 把整条路径跑通：

```bash
cd /data0/OSLA
# 临时降到 2 step，把 dry-run 日志扔一边
bash experiments/deltanet_1.3B_100B/scripts/train_osla_osgm_dd_decay_1.3B.sh \
  2>&1 | tee /tmp/ddd-dryrun.log &
TRAIN_PID=$!
# 等第一步 loss 出现后就杀掉
tail -f /tmp/ddd-dryrun.log | awk '/step.*loss/{print; system("kill '"$TRAIN_PID"'")}'
```

或者更干净的做法：在 `train_osla_osgm_dd_decay_1.3B.sh` 里临时把 `--training.steps 190735` 改成 `--training.steps 3`，跑完再改回来。

dry-run 需要验证：
- [ ] 8 卡都起来了（`nvidia-smi` 看显存）
- [ ] step 0 loss 合理（≈ 10～11 之间，log(vocab) 上下）
- [ ] 第一次 optimizer step 之后 loss 不 NaN
- [ ] 无 `ImportError` / `AttributeError`
- [ ] wandb run 出现

---

## 4. 正式训练

```bash
cd /data0/OSLA
nohup bash experiments/deltanet_1.3B_100B/scripts/train_osla_osgm_dd_decay_1.3B.sh \
  > /data0/OSLA/experiments/deltanet_1.3B_100B/exp/osla-osgm-dd-decay-1.3B/train.log 2>&1 &
echo $! > /tmp/ddd-train.pid
```

**关键超参**（在 `train_osla_osgm_dd_decay_1.3B.sh` 里）：

| 参数 | 值 | 说明 |
|---|---|---|
| GPUs | 8 | `data_parallel_shard_degree=8`（纯 FSDP） |
| `seq_len` | 65536 | 单卡 micro-batch seq 长度 |
| `context_len` | 4096 | `varlen` 拼包后的有效 attention 窗口 |
| tokens/step | 524,288 | 8 × 65536 |
| total steps | 190,735 | ≈ 100B tokens |
| lr | 1e-3 | cosine，warmup 2048 step，min 0.1 |
| optimizer | AdamW (eps 1e-15) | |
| grad clip | 1.0 | `skip_nan_inf` on |
| activation ckpt | `full` | 必须开，否则 80G 也放不下 |
| ckpt 间隔 | 10000 step | 保留最近 5 份 |

输出路径：`experiments/deltanet_1.3B_100B/exp/osla-osgm-dd-decay-1.3B/`
- `logs/` — torchrun 的 rank0 日志
- `checkpoint/` — flame 的 DCP checkpoint
- wandb project = `deltanet_1.3B_100B`，run name = `osla-osgm-dd-decay-1.3B`

---

## 5. 40G 显卡降档方案（仅参考，不建议）

保持每步 tokens 不变（524,288），把 seq_len 降到 16384、用梯度累积补回：

```
--training.seq_len 16384
--training.gradient_accumulation_steps 4
--training.steps 190735   # 不变；tokens/step 仍是 524,288
```

注意 `varlen` + 较短 `seq_len` 会增加 packing overhead，**训练曲线会和 80G 基线不完全一致**（loss 不会差太多，但 step/s 会更慢）。

---

## 6. 监控 & 故障排查

### 日常监控
- wandb: `deltanet_1.3B_100B / osla-osgm-dd-decay-1.3B`
- `nvidia-smi dmon -i 0,1,2,3,4,5,6,7` 看功耗/显存
- `tail -f $DUMP/logs/attempt_0/rank0.log`

### 常见坑

| 现象 | 可能原因 | 处理 |
|---|---|---|
| step 0 直接 `ImportError: chunk_osgm_phase_dd_decay` | 拉到旧分支 / 没 `pip install -e .` | 确认 `git log -1` 是带本 commit 的 main；重装 |
| step 0 loss == NaN | `osgm_a_proj.bias` 被 `_init_weights` 清零（γ 坍缩到 0.5） | 跑 §3.1 自检；若 bias≈0 说明代码被改过 |
| OOM on step 0 | 40G 卡或 activation ckpt 没开 | 见 §5 |
| `NCCL timeout` | 8 卡未全部就绪 / p2p disabled | 脚本已设 `NCCL_P2P_LEVEL=NVL`；检查 `nvidia-smi topo -m` |
| hf-mirror 拉不动数据 | 网络 | 换 `HF_ENDPOINT=https://huggingface.co` 或本地 copy `~/.cache/huggingface/datasets` |
| tokenizer 下不下来 | hf-mirror 偶发挂 | `HF_HUB_ENABLE_HF_TRANSFER=0 python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('fla-hub/delta_net-1.3B-100B', trust_remote_code=True)"` |

### 断点续训

flame 的 DCP checkpoint 自动续训：训练脚本已经设了 `--checkpoint.load_step -1`，重跑同一条命令即可从最新 checkpoint 续上。

```bash
# 杀当前进程
kill $(cat /tmp/ddd-train.pid)
# 等所有 rank 退出
sleep 30 && nvidia-smi
# 重启
nohup bash experiments/deltanet_1.3B_100B/scripts/train_osla_osgm_dd_decay_1.3B.sh \
  >> $DUMP/train.log 2>&1 &
```

---

## 7. 相关代码/文件索引

| 文件 | 作用 |
|---|---|
| `experiments/deltanet_1.3B_100B/configs/osla_osgm_dd_decay_1.3B.json` | 1.3B OSLA-OSGM dd-decay 模型配置 |
| `experiments/deltanet_1.3B_100B/scripts/train_osla_osgm_dd_decay_1.3B.sh` | 8×GPU 启动脚本 |
| `experiments/deltanet_1.3B_100B/scripts/setup_new_server.sh` | 一键新机器环境搭建 |
| `fla/layers/delta_net.py` | DeltaNet layer，`osgm_decay_mode="data_dependent"` 分支 |
| `fla/models/os_delta_net/configuration_os_delta_net.py` | 模型 config 定义 |
| `fla/models/os_delta_net/modeling_os_delta_net.py` | 模型整体 + `_init_weights` 里 `osgm_a_proj.bias=6.9` |
| `fla/ops/os_delta_rule/chunk_osgm_dd_decay.py` | 训练用 chunk kernel（autograd Function） |
| `fla/ops/os_delta_rule/chunk_osgm_phase_dd_decay.py` | Phase1：D 递推 fwd/bwd（per-token γ） |
| `fla/ops/os_delta_rule/fused_recurrent_osgm_dd_decay.py` | 推理用 fused recurrent kernel（forward-only） |

---

## 8. 当次训练的背景

- dd-decay 是 340M sweep 里 ppl 最低的 OSGM 变体（优于 `learnable`、`constant`、`ema`、`ema_norm`）
- 这次 1.3B 训练的目的是验证 data-dependent decay 的 scaling 是否保持优势
- 对标 baseline：`experiments/deltanet_1.3B_100B/exp/deltanet-1.3B-100B`（vanilla DeltaNet，相同 100B schedule）
- 如果跑通，后续会扩到 OSGatedDeltaNet + dd-decay 组合（见 `os-gated-deltanet` 分支，未合入 main）
