GPU=0
MODEL=roberta-base

for EPSILON in 1 2 3
do
for TASK_NAME in ag_news
do
for LR in 1e-5 # {1e-5, 5e-5, 1e-4}
do
MODEL_DIR=rouge_${TASK_NAME}_epsilon${EPSILON}
DATASET_DIR=/root/contrastive_privacy/version_text/output_SanText_glove/${TASK_NAME}/eps_${EPSILON}.00/sword_0.90_p_0.30/
OUTPUT_DIR=/root/contrastive_privacy/version_text/ckpts/test/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_santext.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --santext_dataset_path $DATASET_DIR \
  --output_dir $OUTPUT_DIR \
  --epsilon $EPSILON \
  --num_train_epochs 30 \
  --per_device_train_batch_size 64 \
  --learning_rate $LR \
done
done
done
