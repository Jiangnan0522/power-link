### RGCN-TransE
python pretrain.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --gcn_dim 100 --init_dim 100 --epoch 500 --batch 256 --num_base 5 --n_layer 1 --encoder rgcn --name repro --data wn18rr

### RGCN-Distmult
python pretrain.py --score_func distmult --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --gcn_dim 100 --init_dim 100 --epoch 500 --batch 256 --num_base 5 --n_layer 1 --encoder rgcn --name repro --data wn18rr

### RGCN-ConvE
python pretrain.py --score_func conve --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --gcn_dim 100 --init_dim 100 --epoch 500 --batch 256 --num_base 5 --n_layer 1 --encoder rgcn --name repro --data wn18rr