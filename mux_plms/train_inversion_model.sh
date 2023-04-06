GPU=4
MODEL=checkpoints/finetune/datamux_pretraining/configs/bert_base.json_sst2_princeton-nlp/muxbert_base_gaussian_hadamard_index_pos_10_gaussian_hadamard_index_pos_10_norm_1_rc_0_lr5e-5_tc_1_embedding_noise_eps1

for TASK_NAME in sst2
do
for MIX_SIZE in 10
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
  --per_device_train_batch_size $((32 * ${MIX_SIZE})) \
  --per_device_eval_batch_size $((32 * ${MIX_SIZE})) \
  --learning_rate $LR \
  --mix_size $MIX_SIZE \
  --wandb_name muxplm-sst2-10-inversion_attack-embedding_noise_eps1
done
done
done

