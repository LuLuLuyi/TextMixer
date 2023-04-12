GPU=7
# MODEL=bert-base-uncased
MODEL=/root/mixup/baseline/noise/ckpts/sst2/epsilon1_nu-1

for EPSILON in 1   # {0.05, 0.1, 0.5, 1, 5}
do
for NU in -1 # {-1, 0.1, 0.3, 0.5, 0.8}
do
for TASK_NAME in sst2
do
for LR in 2e-5 # {1e-5, 5e-5*, 1e-4}
do
WANDB_NAME=noise_${TASK_NAME}_epsilon${EPSILON}_nu${NU}
MODEL_DIR=${WANDB_NAME}
OUTPUT_DIR=/root/mixup/baseline/noise/ckpts/${TASK_NAME}/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_noise.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --epsilon $EPSILON \
  --use_wandb 1 \
  --num_train_epochs 30 \
  --per_device_train_batch_size 32 \
  --learning_rate $LR \
  --nullification_rate $NU \
  --train_inversion_model \
  --wandb_name $WANDB_NAME
done
done
done
done
