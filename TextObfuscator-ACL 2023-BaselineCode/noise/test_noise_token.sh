GPU=0
MODEL=/root/contrastive_privacy/version_noise/ckpts/test/ontonotes_epsilon5_nu0.1_bsz64_check
for DATASET_NAME in ontonotes
do
for EPSILON in 5 # {0.05, 0.1, 0.5, 1, 5}
do
for NU in 0.1 # {-1, 0.1, 0.3, 0.5, 0.8}
do
MODEL_DIR=mlc_${DATASET_NAME}_epsilon${EPSILON}_nu${NU}_bsz64_check
OUTPUT_DIR=/root/contrastive_privacy/version_noise/ckpts/test/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python test_noise_token.py \
  --model_name_or_path $MODEL \
  --dataset_name $DATASET_NAME \
  --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size 64 \
  --num_train_epochs 30 \
  --epsilon $EPSILON \
  --use_wandb 1 \
  --nullification_rate $NU \
  --wandb_name mlc_test_noise_${DATASET_NAME}_epsilon${EPSILON}_nu${NU}_bsz64_check
done
done
done