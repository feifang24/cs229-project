export SUBSET_DIR=sd100
export MODEL_ID=992534b7   # this is ignored if do_train=True

export BERT_BASE_DIR=gs://cs229-checkpoints/uncased_L-12_H-768_A-12
export TASK_NAME=imdb
export DATA_DIR=gs://cs229-data/imdb-data
export OUTPUT_DIR=gs://cs229-checkpoints/$TASK_NAME

python3 bert/run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --subset_dir=$SUBSET_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --output_dir=$OUTPUT_DIR \
  --model_id=$MODEL_ID \
  --save_checkpoints_steps=30
