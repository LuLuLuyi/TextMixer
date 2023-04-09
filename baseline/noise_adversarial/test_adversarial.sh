GPU=6
MODEL=/root/contrastive_privacy/version_adversarial/ckpts/sst2/w_adversarial0.5_lr1e-5_1e-4

for W_ADVERSARIAL in 0.5 # {0.01, 0.05, 0.1, 0.5, 1, 5}
do
for TASK_NAME in sst2
do
MODEL_DIR=mlc_${TASK_NAME}_w_adversarial${W_ADVERSARIAL}
OUTPUT_DIR=/root/contrastive_privacy/version_adversarial/ckpts/test/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python test_adversarial.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --use_wandb 1 \
  --add_noise 0 \
  --per_device_train_batch_size 32 \
  --num_train_epochs 30 \
  --w_adversarial $W_ADVERSARIAL \
  --wandb_name mlc_test_adversarial_${TASK_NAME}_w_adv${W_ADVERSARIAL}
done
done
