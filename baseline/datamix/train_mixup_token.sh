GPU=3
MODEL=bert-base-uncased

for MIX_SIZE in 2
do
for LR in 2e-5 5e-5 # {1e-5, 5e-5*, 1e-4}
do
WANDB_NAME=datamix_conll2003_mix${MIX_SIZE}_bsz64_lr${LR}
MODEL_DIR=${WANDB_NAME}
OUTPUT_DIR=./ckpts/conll2003/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_mixup_token.py \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --dataset_name conll2003 \
  --learning_rate $LR \
  --per_device_train_batch_size 64 \
  --train_task_model \
  --train_inversion_model \
  --num_train_epochs 30 \
  --use_wandb 1 \
  --wandb_name $WANDB_NAME

done
done
