GPU=4,5,6
MODEL=roberta-base

for TASK_NAME in sst2
do
for MIX_SIZE in 1
do
for LR in 1e-5 # {1e-5, 5e-5*, 1e-4}
do
MODEL_DIR=roberta_baseline_${TASK_NAME}
OUTPUT_DIR=/root/mixup/mixup_roberta/ckpts/${TASK_NAME}/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_mixup.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --use_wandb 1 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 64 \
  --learning_rate $LR \
  --mix_size $MIX_SIZE \
  --wandb_name roberta_baseline_sst2_actfn_relu
done
done
done

