GPU=5
MODEL=roberta-base

for EPSILON in 0.5 # {0.05, 0.1, 0.5, 1, 5}
do
for W_ADVERSARIAL in 1 5 # {0.01, 0.05, 0.1, 0.5, 1, 5}
do
for TASK_NAME in ag_news
do
MODEL_DIR=eps${EPSILON}_w_adversarial${W_ADVERSARIAL}
OUTPUT_DIR=/root/contrastive_privacy/version_adversarial/ckpts/${TASK_NAME}/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_noise_adversarial.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --use_wandb 1 \
  --add_noise 1 \
  --per_device_train_batch_size 32 \
  --num_train_epochs 30 \
  --epsilon $EPSILON \
  --w_adversarial $W_ADVERSARIAL \
  --wandb_name noise_adversarial_${TASK_NAME}_eps${EPSILON}_w_adv${W_ADVERSARIAL}
done
done
done
