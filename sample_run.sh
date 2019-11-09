export BERT_BASE_DIR=gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12
export IMDB_DIR=gs://cs229-data/imdb-data
export SUBSET_DIR=sd100
export OUTPUT_DIR=gs://cs229-checkpoints/sample-run

python bert/run_classifier.py \
  --task_name=imdb \
  --do_train=true \
  --do_eval=true \
  --data_dir=$IMDB_DIR \
  --subset_dir=$SUBSET_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=20.0 \
  --output_dir=$OUTPUT_DIR
