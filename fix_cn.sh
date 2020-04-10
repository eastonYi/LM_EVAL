gpu=$1
BERT_BASE_DIR=/data3/easton/data/pretrain/chinese_L-12_H-768_A-12
CUDA_VISIBLE_DEVICES=$gpu python main.py \
  -m 'iter_fix' \
  --input='inputs/model-100h_beam-cand-decode_300h.cand' \
  --output='outputs/model-100h_beam-cand-decode_300h.cand' \
  --is_cn
