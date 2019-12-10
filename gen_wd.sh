export SUBSET_DIR=sd800
export MODEL_ID=12090849
# if we want to evaluate multiple we should give a list of tuples and iterate through the list

export BERT_BASE_DIR=gs://cs229-checkpoints/uncased_L-12_H-768_A-12
export TASK_NAME=imdb
export DATA_DIR=/home/src/imdb-data  #gs://cs229-data/imdb-data
export OUTPUT_DIR=gs://cs229-checkpoints/$TASK_NAME

for PRED_DS in dev train
do
python3 bert/run_classifier.py \
    --task_name=$TASK_NAME \
    --mode=predict \
    --pred_ds=$PRED_DS \
    --data_dir=$DATA_DIR \
    --subset_dir=$SUBSET_DIR \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=16 \
    --learning_rate=2e-5 \
    --num_train_epochs=10 \
    --patience=2 \
    --output_dir=$OUTPUT_DIR \
    --model_id=$MODEL_ID 
done