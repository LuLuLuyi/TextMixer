GPU=1
MODEL=bert-base-uncased

for TASK_NAME in imdb
do
for LR in 2e-5 # {1e-5, 5e-5*, 1e-4}
do
WANDB_NAME=finetune_${TASK_NAME}_maxlen512
MODEL_DIR=${WANDB_NAME}
OUTPUT_DIR=/root/mixup/baseline/finetune/ckpts/${TASK_NAME}/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_finetune.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --use_wandb 1 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 32 \
  --learning_rate $LR \
  --train_task_model \
  --train_inversion_model \
  --max_length 512 \
  --wandb_name $WANDB_NAME
done
done
