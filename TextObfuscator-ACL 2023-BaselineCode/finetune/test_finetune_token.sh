GPU=6
MODEL=/root/contrastive_privacy/finetune/ckpts/conll2003/roberta-base-baseline
for DATASET_NAME in conll2003
do
for INV_LR in 1e-5 2e-5 5e-5 1e-4
do 
MODEL_DIR=${DATASET_NAME}_finetune
OUTPUT_DIR=/root/contrastive_privacy/finetune/ckpts/test/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python test_finetune_token.py \
  --model_name_or_path $MODEL \
  --dataset_name $DATASET_NAME \
  --output_dir $OUTPUT_DIR \
  --use_wandb 1 \
  --wandb_name test_finetune_${DATASET_NAME}_epoch3
done
done

