GPU=7
MODEL=roberta-base

for W_ADVERSARIAL in 0.1 # {0.01, 0.05, 0.1, 0.5, 1, 5}
do
for TASK_NAME in ag_news
do
MODEL_DIR=w_adversarial${W_ADVERSARIAL}
OUTPUT_DIR=/root/contrastive_privacy/version_adversarial/ckpts/${TASK_NAME}/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_adversarial.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --use_wandb 1 \
  --add_noise 0 \
  --per_device_train_batch_size 32 \
  --num_train_epochs 30 \
  --w_adversarial $W_ADVERSARIAL \
  --seed 42 \
  --wandb_name adversarial_${TASK_NAME}_w_adv${W_ADVERSARIAL}
done
done
