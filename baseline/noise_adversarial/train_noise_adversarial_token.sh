GPU=7
MODEL=roberta-base

for EPSILON in 0.05 0.1 0.5  # {0.05, 0.1, 0.5, 1, 5}
do
for W_ADVERSARIAL in 0.01 0.05 0.1 0.5 1 5 # {0.01, 0.05, 0.1, 0.5, 1, 5}
do
MODEL_DIR=ontonotes_eps${EPSILON}_w_adversarial${W_ADVERSARIAL}
OUTPUT_DIR=/root/contrastive_privacy/version_adversarial/ckpts/ontonotes/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_noise_adversarial_token.py \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --epsilon $EPSILON \
  --use_wandb 1 \
  --add_noise 1 \
  --w_adversarial $W_ADVERSARIAL \
  --wandb_name noise_adversarial_ontonotes_eps${EPSILON}_w_adv${W_ADVERSARIAL}
done
done

