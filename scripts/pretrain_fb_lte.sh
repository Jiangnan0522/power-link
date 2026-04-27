## TransE
python pretrain.py --score_func transe --opn mult --hid_drop 0.2 --gpu 0 --x_ops "p" --n_layer 0 --init_dim 200 --name lte

## DistMult
python pretrain.py --score_func distmult --opn mult --gpu 0 --x_ops "p.b.d" --n_layer 0 --init_dim 200 --name lte

## ConvE
python pretrain.py --score_func conve --opn mult --gpu 0 --x_ops "p.b.d" --n_layer 0 --init_dim 200 --name lte