GPU=5
MODEL=bert-base-uncased

for EPSILON in 0.1 0.5 1 2 5 # {0.05, 0.1, 0.5, 1, 5}
do
for NU in 0.3 # {-1, 0.1, 0.3, 0.5, 0.8}
do
WANDB_NAME=noise_ontonotes_epsilon${EPSILON}_nu${NU}
MODEL_DIR=${WANDB_NAME}
OUTPUT_DIR=/root/mixup/baseline/noise/ckpts/ontonotes/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_noise_token.py \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --dataset_name tner/ontonotes5 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 32 \
  --train_task_model \
  --train_inversion_model \
  --num_train_epochs 30 \
  --epsilon $EPSILON \
  --use_wandb 1 \
  --nullification_rate $NU \
  --wandb_name $WANDB_NAME
  done
  done
