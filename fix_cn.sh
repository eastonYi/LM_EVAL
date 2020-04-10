gpu=$1
export BERT_BASE_DIR=/data3/easton/data/pretrain/chinese_L-12_H-768_A-12
CUDA_VISIBLE_DEVICES=$gpu python main.py \
  --input_file='inputs/model-100h_beam-cand-decode_300h.cand' \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=40 \
  --predict_batch_size=10 \
  --output='outputs/model-100h_beam-cand-decode_300h.cand' \
  --is_cn
