GPU=5
MODEL=roberta-base
for DATASET_NAME in conll2003
do
for EPSILON in 3
do
MODEL_DIR=rouge_${DATASET_NAME}_epsilon${EPSILON}
DATASET_DIR=/root/contrastive_privacy/version_text/output_SanText_glove/${DATASET_NAME}/eps_${EPSILON}.00/sword_0.90_p_0.30/
OUTPUT_DIR=/root/contrastive_privacy/version_text/ckpts/test/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_santext_token.py \
  --model_name_or_path $MODEL \
  --dataset_name $DATASET_NAME \
  --santext_dataset_path $DATASET_DIR \
  --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size 64 \
  --num_train_epochs 30 \
  --use_wandb 1 \
  --wandb_name rouge_santext_${DATASET_NAME}_epsilon${EPSILON}
done
done

