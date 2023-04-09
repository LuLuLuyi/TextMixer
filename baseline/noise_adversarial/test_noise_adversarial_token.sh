GPU=7
MODEL=/root/contrastive_privacy/version_adversarial/ckpts/ner/ner_eps5_w_adversarial0.1

for DATASET_NAME in conll2003
do
for EPSILON in 5  # {0.05, 0.1, 0.5, 1, 5}
do
for W_ADVERSARIAL in 0.1  # {0.01, 0.05, 0.1, 0.5, 1, 5}
do
MODEL_DIR=mlc_${DATASET_NAME}_eps${EPSILON}_w_adversarial${W_ADVERSARIAL}
OUTPUT_DIR=/root/contrastive_privacy/version_adversarial/ckpts/test/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python test_noise_adversarial_token.py \
  --model_name_or_path $MODEL \
  --dataset_name $DATASET_NAME \
  --output_dir $OUTPUT_DIR \
  --epsilon $EPSILON \
  --use_wandb 1 \
  --add_noise 1 \
  --w_adversarial $W_ADVERSARIAL \
  --wandb_name mlc_noise_adversarial_${DATASET_NAME}_eps${EPSILON}_w_adv${W_ADVERSARIAL}
done
done
done

