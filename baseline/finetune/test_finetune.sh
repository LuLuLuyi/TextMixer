GPU=1
MODEL=roberta-base

for TASK_NAME in sst2
do
MODEL_DIR=mlc_${TASK_NAME}_finetune
OUTPUT_DIR=/root/contrastive_privacy/finetune/ckpts/sst2/roberta-base-baseline

CUDA_VISIBLE_DEVICES=$GPU python test_finetune.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --use_wandb 1 \
  --per_device_train_batch_size 32 \
  --wandb_name mlc_test_finetune_${TASK_NAME}
done
