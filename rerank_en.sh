gpu=$1
CUDA_VISIBLE_DEVICES=$gpu python main.py \
  -m 'rerank' \
  --bert_dir='/data3/easton/data/pretrain/uncased_L-12_H-768_A-12' \
  --input='inputs/libri_model-100h_beam-cand-decode_300h.filted-cand' \
  --output='outputs/libri_model-100h_beam-cand-decode_300h.iterfixed' \
  --max_seq_length=110
