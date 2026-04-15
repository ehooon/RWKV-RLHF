# 本脚本使用的微调方法为 Miss，也可也在 lora/stat tuning 中添加和修改 DPO 专用参数来进行 DPO
load_model="/home/manjuan/Project/RWKV-PEFT/RWKV-PEFT/rwkv7-g1e-1.5b-20260309-ctx8192.pth"
proj_dir='./test_dpo' 
data_file='/home/manjuan/Project/RWKV-RL/test.jsonl' 

n_layer=24
n_embd=2048

micro_bsz=4 

epoch_save=1
epoch_steps=200

ctx_len=256 

peft_config='{"r":8}'
dpo_beta=0.1 # DPO 的偏离惩罚系数 beta

python train.py --load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--vocab_size 65536 \
--data_type dpo \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 10 --epoch_save $epoch_save \
--lr_init 1e-6 --lr_final 1e-8 \
--accelerator gpu --precision bf16 \
--devices 1 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x070" \
--peft miss --peft_config "$peft_config" \
--dpo_beta $dpo_beta 