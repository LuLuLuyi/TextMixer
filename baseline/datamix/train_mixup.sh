GPU=5
MODEL=bert-base-uncased

for TASK_NAME in ag_news
do
for MIX_SIZE in 2
do
for LR in 2e-5 5e-5 # {1e-5, 5e-5*, 1e-4}
do

WANDB_NAME=datamix_${TASK_NAME}_mix${MIX_SIZE}_bsz64_lr${LR}
MODEL_DIR=${WANDB_NAME}
OUTPUT_DIR=./ckpts/${TASK_NAME}/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_mixup.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --use_wandb 1 \
  --num_train_epochs 30 \
  --per_device_train_batch_size 64 \
  --train_task_model \
  --train_inversion_model \
  --learning_rate $LR \
  --mix_size $MIX_SIZE \
  --wandb_name ${WANDB_NAME}
done
done
done

