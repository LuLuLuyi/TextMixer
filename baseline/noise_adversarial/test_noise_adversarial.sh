GPU=2
MODEL=/root/contrastive_privacy/version_adversarial/ckpts/sst2/eps0.5_w_adversarial0.1

for EPSILON in 0.5 # {0.05, 0.1, 0.5, 1, 5}
do
for W_ADVERSARIAL in 0.1 # {0.01, 0.05, 0.1, 0.5, 1, 5}
do
for TASK_NAME in sst2
do
MODEL_DIR=mlc_${TASK_NAME}_eps${EPSILON}_w_adversarial${W_ADVERSARIAL}
OUTPUT_DIR=/root/contrastive_privacy/version_adversarial/ckpts/test/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python test_noise_adversarial.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --use_wandb 1 \
  --add_noise 1 \
  --per_device_train_batch_size 32 \
  --num_train_epochs 30 \
  --epsilon $EPSILON \
  --w_adversarial $W_ADVERSARIAL \
  --wandb_name mlc_test_noise_adversarial_${TASK_NAME}_eps${EPSILON}_w_adv${W_ADVERSARIAL}
done
done
done
