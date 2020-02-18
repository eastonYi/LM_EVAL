export BERT_BASE_DIR=/data3/easton/data/pretrain/chinese_L-12_H-768_A-12
python main.py \
  --input_file=test.zh.tsv \
  --vocab_file=vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --output_dir=/tmp/lm_output/
  # --vocab_file=$BERT_BASE_DIR/vocab.txt \
