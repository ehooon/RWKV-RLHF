<h1 align="center">
RWKV-RLHF
</h1>

<p align="center">
  <strong>English</strong> | <a href="README_zh.md">中文</a>
</p>

> This project is a secondary development based on [RWKV-PEFT](https://github.com/Joluck/RWKV-PEFT), focusing on DPO (Direct Preference Optimization) preference alignment training for RWKV models.
> This project also supports all the fine-tuning features of the original project (such as LoRA / MiSS / State Tuning, etc.). For more details, please refer to the upstream official repository.

> 2026.04.15: Completed the construction of the DPO training framework.

## Installation Environment

Python 3.10 or higher is recommended.

```bash
git clone https://github.com/ehooon/RWKV-RLHF.git
cd RWKV-RLHF
uv sync  # or use `pip install .`
```

## DPO Usage Guide

### Data Format Requirements
Use standard **JSONL** format data. Each line must be a valid JSON object containing the following 3 fields:
```json
{"prompt": "User input/question content", "chosen": "Preferred response expected from the model", "rejected": "Low-quality response the model should avoid"}
```

### Training Startup Example
Refer to the `scripts/dpo.sh` script. Parameters can be adjusted based on your actual machine configuration and the table below (using the 1.5B model as an example here):
```bash
load_model="/path/to/your/rwkv-base-model.pth"
proj_dir="/path/to/save/output"
data_file="/path/to/your/dpo-data.jsonl"

# Model parameters (using 1.5B as an example)
n_layer=24 
n_embd=2048

# Training parameters
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

### Parameter Description

| Parameter | Description | Default |
|------|------|--------|
| `--data_type dpo` | **Required**. Set to `dpo` to enable DPO training mode. | - |
| `--dpo_beta` | DPO beta coefficient, controls the alignment strength, usually between `0.05 - 0.2`. | `0.1` |
| `--peft` | **Required**. Supports fine-tuning methods like `lora`, `miss`, `state`, etc. For full fine-tuning, set to `none`. Using State Tuning requires the `--op fla` parameter. | - |
| `--merge` | Whether to automatically merge the PEFT adapter into a full model after training (`1` for auto-merge, `0` for no merge). | `1` |

### Common Model Parameters Reference
When configuring the training script, please accurately fill in the corresponding `n_layer` and `n_embd` parameters based on the size of the base model you are using:

| Model Size | n_layer | n_embd |
| :--- | :--- | :--- |
| **0.1B** | 12 | 768 |
| **0.4B** | 24 | 1024 |
| **1.5B** | 24 | 2048 |
| **3B** | 32 | 2560 |
| **7B** | 32 | 4096 |
| **14B** | 61 | 4096 |

### Model Output
- During training, the model will be automatically saved to the directory specified by `--proj_dir` after each epoch.
- By default (`--merge 1`), the script will automatically merge the PEFT adapter with the base model, outputting a full RWKV format model that can be used directly for inference.
- If `--merge 0` is set, only the adapter parameters will be saved in the `{proj_dir}-adapter` directory.

## Acknowledgements and Citation

If you use content contributed by RWKV-PEFT, please cite or acknowledge it. For details, please refer to the original [RWKV-PEFT](https://github.com/Joluck/RWKV-PEFT) project.

If you use the relevant contributions of this project in your academic research, please use the following BibTeX format for citation:

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
