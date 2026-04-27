### RGCN-TransE
python pretrain.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --gcn_dim 100 --init_dim 100 --epoch 500 --batch 128 --num_base 5 --n_layer 1 --encoder rgcn --name repro --data yago
