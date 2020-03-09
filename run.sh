export BERT_BASE_DIR=/data3/easton/data/pretrain/chinese_L-12_H-768_A-12
CUDA_VISIBLE_DEVICES=0 python main.py \
  --input_file=preds_of_beam1_from_average8ckpts-0.probs \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=30 \
  --output='output.txt'
