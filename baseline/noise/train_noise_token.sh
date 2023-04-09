GPU=0
MODEL=roberta-base

for EPSILON in 5 # {0.05, 0.1, 0.5, 1, 5}
do
for NU in 0.1 # {-1, 0.1, 0.3, 0.5, 0.8}
do
MODEL_DIR=epsilon${EPSILON}_nu${NU}_check_eval_word_dropout
OUTPUT_DIR=/root/contrastive_privacy/version_noise/ckpts/ontonotes/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_noise_token.py \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size 64 \
  --num_train_epochs 30 \
  --epsilon $EPSILON \
  --use_wandb 1 \
  --nullification_rate $NU \
  --wandb_name noise_ontonotes_epsilon${EPSILON}_nu${NU}_check_eval_word_dropout
  done
  done
