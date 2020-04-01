export BERT_BASE_DIR=/data3/easton/data/pretrain/chinese_L-12_H-768_A-12
# python create_pretraining_data.py \
#   --input_file=finetune/trans.text \
#   --output_file=finetune/trans.tfrecord \
#   --vocab_file=$BERT_BASE_DIR/vocab.txt \
#   --do_lower_case=True \
#   --max_seq_length=128 \
#   --max_predictions_per_seq=20 \
#   --masked_lm_prob=0.15 \
#   --random_seed=12345 \
#   --dupe_factor=5


python run_pretraining.py \
  --input_file=finetune/trans.tfrecord \
  --output_dir=finetune/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=100000 \
  --num_warmup_steps=50 \
  --learning_rate=2e-5
