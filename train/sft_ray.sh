
export RAY_ADDRESS="http://127.0.0.1:8265"

uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-32B-Instruct"
lr=1e-5
min_lr=0
epochs=1
micro_batch_size=1 # -> batch_size will be 16 if 8 gpus
push_to_hub=false
gradient_accumulation_steps=1
max_steps=-1

ray job submit --working-dir . -- python \
train/sft_ray.py \
--per_device_train_batch_size=${micro_batch_size} \
--per_device_eval_batch_size=${micro_batch_size} \
--gradient_accumulation_steps=${gradient_accumulation_steps} \
--num_train_epochs=${epochs} \
--max_steps=${max_steps} \
--train_file_path="simplescaling/s1K_tokenized" \
--model_name=${base_model} \
--warmup_ratio=0.05 \
--fsdp="full_shard auto_wrap" \
--fsdp_config="train/fsdp_config_qwen.json" \
--bf16=True \
--eval_strategy="steps" \
--eval_steps=50 \
--logging_steps=1 \
--save_strategy="no" \
--save_steps=1000 \
--load_best_model_at_end=False \
--lr_scheduler_type="cosine" \
--learning_rate=${lr} \
--weight_decay=1e-4 \
--adam_beta1=0.9 \
--adam_beta2=0.95 \
--output_dir="/mnt/workspace/ryan/checkpoints/s1_${uid}" \
--hub_model_id="simplescaling/s1-${uid}" \
--push_to_hub=${push_to_hub} \
--save_only_model=True