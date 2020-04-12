threshold=$1
python analyse_cand.py \
    -m cand \
    --threshold $threshold \
    --input inputs/model-100h_beam-cand-decode_300h.cand \
    --output outputs/model-100h_beam-cand-decode_300h.ref-fixed
