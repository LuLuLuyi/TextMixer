GPU=0
MODEL=bert-base-uncased

WANDB_NAME=finetune_ontonotes
MODEL_DIR=${WANDB_NAME}
OUTPUT_DIR=/root/mixup/baseline/noise/ckpts/ontonotes/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_finetune_token.py \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --dataset_name tner/ontonotes5 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 32 \
  --train_task_model \
  --train_inversion_model \
  --num_train_epochs 20 \
  --use_wandb 1 \
  --wandb_name $WANDB_NAME

