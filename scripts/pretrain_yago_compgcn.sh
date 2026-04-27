### CompGCN-TransE
python pretrain.py --score_func transe --opn mult --n_layer 1 --hid_drop 0.2 --init_dim 100 --gcn_dim 100 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 128 --gpu 0 --name wn_repro --data yago
