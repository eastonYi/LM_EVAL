fin=''
fout=''
vocab='/data3/easton/data/pretrain/uncased_L-12_H-768_A-12/vocab.txt'
python tokenization.py --vocab $vocab --fin $fin --fout $fout
