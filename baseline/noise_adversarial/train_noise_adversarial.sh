GPU=7
MODEL=bert-base-uncased
# MODEL=/root/mixup/baseline/noise_adversarial/ckpts/sst2/noise_adversarial_sst2_eps1_wadv0.1_lr2e-5_5e-5

for EPSILON in 0.1 0.5 1 2 5 # {0.05, 0.1, 0.5, 1, 5}
do
for W_ADVERSARIAL in 1 # {0.01, 0.05, 0.1, 0.5, 1, 5}
do
for TASK_NAME in sst2
do
WANDB_NAME=noise_adversarial_${TASK_NAME}_eps${EPSILON}_wadv${W_ADVERSARIAL}
MODEL_DIR=${WANDB_NAME}
OUTPUT_DIR=/root/mixup/baseline/noise_adversarial/ckpts/${TASK_NAME}/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_noise_adversarial.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --use_wandb 1 \
  --add_noise 1 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30 \
  --train_task_model \
  --train_inversion_model \
  --epsilon $EPSILON \
  --w_adversarial $W_ADVERSARIAL \
  --wandb_name $WANDB_NAME
done
done
done
