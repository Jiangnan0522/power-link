# TransE
python pretrain.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --gcn_dim 128 --init_dim 128 --embed_dim=128 --epoch 3000 --batch 512 --num_base 5 --n_layer 2 --encoder wgcn --data wn18rr

# DistMult
python pretrain.py --score_func distmult --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --gcn_dim 128 --init_dim 128 --embed_dim=128 --epoch 3000 --batch 512 --num_base 5 --n_layer 2 --encoder wgcn --data wn18rr

# ConvE
python pretrain.py --score_func conve --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --gcn_dim 128 --init_dim 128 --k_h 16 --k_w 8 --embed_dim 128 --epoch 3000 --batch 512 --n_layer 2 --encoder wgc --data wn18rr