GPU=4
MODEL=bert-base-uncased

for EPSILON in 0.1 0.5 1 2 5  # {0.05, 0.1, 0.5, 1, 5}
do
for W_ADVERSARIAL in 0.01 # {0.01, 0.05, 0.1, 0.5, 1, 5}
do
WANDB_NAME=noise_adversarial_conll2003_eps${EPSILON}_wadv${W_ADVERSARIAL}
MODEL_DIR=${WANDB_NAME}
OUTPUT_DIR=/root/mixup/baseline/noise_adversarial/ckpts/conll2003/${MODEL_DIR}

CUDA_VISIBLE_DEVICES=$GPU python train_noise_adversarial_token.py \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --epsilon $EPSILON \
  --use_wandb 1 \
  --add_noise 1 \
  --dataset_name conll2003 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 30 \
  --train_task_model \
  --train_inversion_model \
  --w_adversarial $W_ADVERSARIAL \
  --wandb_name $WANDB_NAME
done
done

