GPU=6
MODEL=bert-base-uncased

for EPSILON in 1
do
for TASK_NAME in ag_news
do
for LR in 1e-5 # 2e-5 5e-5 # {1e-5, 5e-5, 1e-4}
do
DATASET_DIR=./output_SanText_glove/${TASK_NAME}/eps_${EPSILON}.00/sword_0.90_p_0.30/
OUTPUT_DIR=./ckpts/${task_name}/${MODEL_DIR}
WANDB_NAME=santext_glove_${TASK_NAME}_eps${EPSILON}_bsz64_lr${LR}
MODEL_DIR=${WANDB_NAME}

CUDA_VISIBLE_DEVICES=$GPU python train_santext.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --santext_dataset_path $DATASET_DIR \
  --use_wandb 1 \
  --wandb_name $WANDB_NAME \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 30 \
  --per_device_train_batch_size 64 \
  --train_task_model \
  --train_inversion_model \
  --learning_rate $LR 
done
done
done
