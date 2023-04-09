GPU=2
MODEL=roberta-base

CUDA_VISIBLE_DEVICES=$GPU python train_santext_sst2.py \
  --model_name_or_path $MODEL \
  --task_name sst-2 \
  --do_train \
  --do_eval \
  --data_dir ./output_SanText_glove/SST-2/eps_3.00 \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir ./ckpts/glove-sst2-sanitize/ \
  --overwrite_output_dir \
  --overwrite_cache \
  --save_steps 5000 \
  --evaluation_strategy epoch \
  --use_wandb 1 \
  --wandb_name santext_glove_sst2_eps3