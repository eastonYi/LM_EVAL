gpu=$1
CUDA_VISIBLE_DEVICES=$gpu python main.py \
  -m 'iter_fix' \
  --bert_dir='/data3/easton/data/pretrain/chinese_L-12_H-768_A-12' \
  --input='inputs/model-100h_beam-cand-decode_300h.cand' \
  --output='outputs/model-100h_beam-cand-decode_300h.iterfixed' \
  --max_seq_length=40 \
  --threshold=0.02
