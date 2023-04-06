GPU=3
MODEL=princeton-nlp/datamux-sst2-10

for TASK_NAME in sst2
do
for MIX_SIZE in 2
do
for LR in 1e-5 # {1e-5, 5e-5*, 1e-4}
do
MODEL_DIR=datamux-sst2-10
OUTPUT_DIR=/root/mixup/mixup_roberta/ckpts/${TASK_NAME}/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_inversion_model.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --use_wandb 1 \
  --num_train_epochs 30 \
  --per_device_train_batch_size 60 \
  --learning_rate $LR \
  --mix_size $MIX_SIZE \
  --wandb_name datamux-sst2-10-inversion_attack
done
done
done

