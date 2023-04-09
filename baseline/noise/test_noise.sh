GPU=2
MODEL=/root/contrastive_privacy/version_noise/ckpts/ag_news/epsilon1_nu0.1

for EPSILON in 1  # {0.05, 0.1, 0.5, 1, 5}
do
for NU in 0.1 # {-1, 0.1, 0.3, 0.5, 0.8}
do
for TASK_NAME in ag_news
do
for LR in 1e-5 # {1e-5, 5e-5*, 1e-4}
do
MODEL_DIR=mlc_${TASK_NAME}_epsilon${EPSILON}_nu${NU}
OUTPUT_DIR=/root/contrastive_privacy/version_noise/ckpts/test/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python test_noise.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --epsilon $EPSILON \
  --use_wandb 1 \
  --num_train_epochs 30 \
  --per_device_train_batch_size 64 \
  --learning_rate $LR \
  --nullification_rate $NU \
  --wandb_name mlc_test_noise_${TASK_NAME}_epsilon${EPSILON}_nu${NU}
done
done
done
done
