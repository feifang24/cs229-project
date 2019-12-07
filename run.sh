export MODEL_ID=992534b7   # this is ignored if mode=train

export BERT_BASE_DIR=gs://cs229-checkpoints/uncased_L-12_H-768_A-12
export TASK_NAME=imdb
export DATA_DIR=gs://cs229-data/imdb-data
export OUTPUT_DIR=gs://cs229-checkpoints/$TASK_NAME

for SUBSET_DIR in sd100  sd200 sd400 sd800 sd1600 sd3200 sd6400 sd12800 og
do
  python3 bert/run_classifier.py \
    --task_name=$TASK_NAME \
    --mode=train \
    --data_dir=$DATA_DIR \
    --subset_dir=$SUBSET_DIR \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=10 \
    --patience=2 \
    --output_dir=$OUTPUT_DIR \
    --model_id=$MODEL_ID 
done
