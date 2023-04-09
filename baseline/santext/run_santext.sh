GPU=5

for EPSILON in 1 2 3
do
for TASK_NAME in conll2003 ontonotes
do
CUDA_VISIBLE_DEVICES=$GPU python run_SanText.py \
    --task $TASK_NAME \
    --method SanText_plus \
    --epsilon $EPSILON \
    --embedding_type glove \
    --data_dir ./data/${TASK_NAME}/ \
    --output_dir ./output_SanText_glove/${TASK_NAME}/ \
    --threads 8 \
    --word_embedding_path ./data/glove.840B.300d.txt \
    --p 0.3 \
    --sensitive_word_percentage 0.9 
done
done            
    