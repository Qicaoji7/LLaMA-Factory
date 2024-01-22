#!/bin/bash

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=4,5,6,7

mkdir -p /home/zqq/.cache/huggingface/accelerate/

# 配置分布式环境
cat <<EOT > /home/zqq/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: no
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOT

# 运行训练脚本
accelerate launch src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path modelscope/Llama-2-7b-ms \
    --dataset self_cognition \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir output/sft \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16


#accelerate launch src/train_bash.py \
#    --stage rm \
#    --do_train \
#    --model_name_or_path path_to_llama_model \
#    --adapter_name_or_path path_to_sft_checkpoint \
#    --create_new_adapter \
#    --dataset comparison_gpt4_zh \
#    --template default \
#    --finetuning_type lora \
#    --lora_target q_proj,v_proj \
#    --output_dir path_to_rm_checkpoint \
#    --per_device_train_batch_size 2 \
#    --gradient_accumulation_steps 4 \
#    --lr_scheduler_type cosine \
#    --logging_steps 10 \
#    --save_steps 1000 \
#    --learning_rate 1e-6 \
#    --num_train_epochs 1.0 \
#    --plot_loss \
#    --fp16

#accelerate launch src/train_bash.py \
#    --stage ppo \
#    --do_train \
#    --model_name_or_path path_to_llama_model \
#    --adapter_name_or_path path_to_sft_checkpoint \
#    --create_new_adapter \
#    --dataset alpaca_gpt4_zh \
#    --template default \
#    --finetuning_type lora \
#    --lora_target q_proj,v_proj \
#    --reward_model path_to_rm_checkpoint \
#    --output_dir path_to_ppo_checkpoint \
#    --per_device_train_batch_size 2 \
#    --gradient_accumulation_steps 4 \
#    --lr_scheduler_type cosine \
#    --top_k 0 \
#    --top_p 0.9 \
#    --logging_steps 10 \
#    --save_steps 1000 \
#    --learning_rate 1e-5 \
#    --num_train_epochs 1.0 \
#    --plot_loss \
#    --fp16