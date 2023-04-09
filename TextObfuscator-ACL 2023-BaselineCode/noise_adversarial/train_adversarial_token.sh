GPU=1
MODEL=roberta-base

for W_ADVERSARIAL in 0.05 0.1 0.5 1 5 # {0.01, 0.05, 0.1, 0.5, 1, 5}
do
MODEL_DIR=ontonotes_w_adversarial${W_ADVERSARIAL}
OUTPUT_DIR=/root/contrastive_privacy/version_adversarial/ckpts/ontonotes/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_adversarial_token.py \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --use_wandb 1 \
  --add_noise 0 \
  --w_adversarial $W_ADVERSARIAL \
  --wandb_name adversarial_ontonotes_w_adversarial${W_ADVERSARIAL}
done

