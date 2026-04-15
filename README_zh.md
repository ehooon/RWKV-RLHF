<h1 align="center">
RWKV-RLHF
</h1>

<p align="center">
  <a href="README.md">English</a> | <strong>中文</strong>
</p>

> 本项目基于 [RWKV-PEFT](https://github.com/Joluck/RWKV-PEFT) 进行二次开发，专注于 RWKV 模型的 DPO（Direct Preference Optimization）偏好对齐训练。
> 本项目同时支持原项目的所有微调功能（如 LoRA / MiSS / State Tuning 等），具体详情请参考上游官方仓库。

> 2026.04.15 完成了 DPO 的训练框架搭建。

## 安装训练环境

建议使用 Python 3.10 及以上版本。

```bash
git clone [https://github.com/ehooon/RWKV-RLHF.git](https://github.com/ehooon/RWKV-RLHF.git)
cd RWKV-RLHF
uv sync  # 或者使用 pip install .
```

## DPO 使用指南

### 数据格式要求
使用标准的 **JSONL** 格式数据，每一行需为一个有效的 JSON 对象，包含以下 3 个字段：
```json
{"prompt": "用户输入/提问内容", "chosen": "期望模型输出的优先回答", "rejected": "需要模型避免的低质量回答"}
```

### 训练启动示例
参考 `scripts/dpo.sh` 脚本，可根据实际机器配置及上方表格调整参数（以下以 1.5B 模型为例）：
```bash
load_model="/path/to/your/rwkv-base-model.pth"
proj_dir="/path/to/save/output"
data_file="/path/to/your/dpo-data.jsonl"

# 模型参数 (以 1.5B 为例)
n_layer=24 
n_embd=2048

# 训练参数
micro_bsz=8
epoch_save=1
epoch_steps=200
ctx_len=1024
dpo_beta=0.1

python train.py --load_model $load_model \
    --proj_dir $proj_dir --data_file $data_file \
    --vocab_size 65536 \
    --data_type dpo \
    --n_layer $n_layer --n_embd $n_embd \
    --ctx_len $ctx_len --micro_bsz $micro_bsz \
    --epoch_steps $epoch_steps --epoch_count 3 --epoch_save $epoch_save \
    --lr_init 5e-6 --lr_final 5e-6 \
    --accelerator gpu --precision bf16 \
    --devices 1 --strategy deepspeed_stage_1 --grad_cp 1 \
    --my_testing "x070" \
    --peft lora --peft_config '{"r":8,"lora_alpha":32,"lora_dropout":0.05}' \
    --dpo_beta $dpo_beta
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_type dpo` | **必填**。指定为 `dpo` 以启用 DPO 训练模式。 | - |
| `--dpo_beta` | DPO 温度系数，控制对齐强度，通常在 `0.05 - 0.2` 之间。 | `0.1` |
| `--peft` | **必填**。支持 `lora`、`miss`、`state` 等微调方式。全量微调请设为 `none`，使用 State Tuning 需要搭配 `--op fla` 参数。 | - |
| `--merge` | 训练结束是否自动合并 PEFT adapter 为全量模型（`1` 为自动合并，`0` 为不合并）。 | `1` |

### 常见模型参数参考
在配置训练脚本时，请根据您使用的基础模型规模，准确填写对应的 `n_layer` 和 `n_embd` 参数：

| 模型参数 | n_layer | n_embd |
| :--- | :--- | :--- |
| **0.1B** | 12 | 768 |
| **0.4B** | 24 | 1024 |
| **1.5B** | 24 | 2048 |
| **3B** | 32 | 2560 |
| **7B** | 32 | 4096 |
| **14B** | 61 | 4096 |

### 模型输出
- 训练过程中，每个 epoch 结束后会自动保存模型到 `--proj_dir` 指定的目录。
- 默认情况下（`--merge 1`），脚本会自动将 PEFT adapter 与基础模型合并，输出全量 RWKV 格式模型，可直接用于推理。
- 如果设置 `--merge 0`，则仅保存 adapter 参数到 `{proj_dir}-adapter` 目录中。

## 致谢与引用

如使用了 RWKV-PEFT 贡献的内容，需进行引用或致谢，详情请参考原 [RWKV-PEFT](https://github.com/Joluck/RWKV-PEFT) 项目。

如果您在学术研究中使用了本项目的相关贡献，请使用以下 BibTeX 格式进行引用：

```bibtex
@misc{rwkvrlhf,
  author       = {Yinhong Fan},
  title        = {RWKV-RLHF: RLHF for RWKV Models base on RWKV-PEFT},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/ehooon/RWKV-RLHF](https://github.com/ehooon/RWKV-RLHF)}}
}
```