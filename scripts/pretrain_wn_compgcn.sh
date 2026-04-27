# REPRO
## TransE
python pretrain.py --score_func transe --opn mult --n_layer 1 --hid_drop 0.2 --init_dim 100 --gcn_dim 100 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --gpu 0 --name wn_repro --data wn18rr

## DistMult
python pretrain.py --score_func distmult --opn mult --n_layer 1 --init_dim 200 --batch 256 --gpu 0 --name wn_repro --data wn18rr

## ConvE
python pretrain.py --score_func conve --opn mult --n_layer 1 --init_dim 200 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --name wn_repro --data wn18rr



# RAT
## TransE
python pretrain.py --score_func transe --opn mult --n_layer 1 --hid_drop 0.2 --init_dim 100 --gcn_dim 100 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --gpu 0 --name wn_rat --data wn18rr --rat

## DistMult
python pretrain.py --score_func distmult --opn mult --n_layer 1 --init_dim 200 --batch 256 --gpu 0 --name wn_rat --data wn18rr --rat

## ConvE
python pretrain.py --score_func conve --opn mult --n_layer 1 --init_dim 200 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --name wn_rat --data wn18rr --rat



# WSI
## TransE
python pretrain.py --score_func transe --opn mult --n_layer 1 --hid_drop 0.2 --init_dim 100 --gcn_dim 100 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --gpu 0 --name wn_wsi --data wn18rr --wsi

## DistMult
python pretrain.py --score_func distmult --opn mult --n_layer 1 --init_dim 200 --batch 256 --gpu 0 --name wn_wsi --data wn18rr --wsi

## ConvE
python pretrain.py --score_func conve --opn mult --n_layer 1 --init_dim 200 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --name wn_wsi --data wn18rr --wsi



## WNI
## TransE
python pretrain.py --score_func transe --opn mult --n_layer 1 --hid_drop 0.2 --init_dim 100 --gcn_dim 100 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --gpu 0 --name wn_wni --data wn18rr --wni

## DistMult
python pretrain.py --score_func distmult --opn mult --n_layer 1 --init_dim 200 --batch 256 --gpu 0 --name wn_wni --data wn18rr --wni

## ConvE
python pretrain.py --score_func conve --opn mult --n_layer 1 --init_dim 200 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --name wn_wni --data wn18rr --wni



## WSI_RAT
## TransE
python pretrain.py --score_func transe --opn mult --n_layer 1 --hid_drop 0.2 --init_dim 100 --gcn_dim 100 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --gpu 0 --name wn_rat_wsi --data wn18rr --rat --wsi

## DistMult
python pretrain.py --score_func distmult --opn mult --n_layer 1 --init_dim 200 --batch 256 --gpu 0 --name wn_rat_wsi --data wn18rr --rat --wsi

## ConvE
python pretrain.py --score_func conve --opn mult --n_layer 1 --init_dim 200 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --name wn_rat_wsi --data wn18rr --rat --wsi



# nogcn
## TransE
python pretrain.py --score_func transe --opn mult --n_layer 0 --init_dim 100 --gcn_dim 100 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --name wn_nogcn --data wn18rr

## DistMult
python pretrain.py --score_func distmult --opn mult --n_layer 0 --init_dim 200 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --name wn_nogcn --data wn18rr

## ConvE
python pretrain.py --score_func conve --opn mult --n_layer 0 --init_dim 200 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --name wn_nogcn --data wn18rr



# no_LTR
## TransE
python pretrain.py --score_func transe --opncd mult --n_layer 1 --init_dim 100 --gcn_dim 100 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --name wn_noLTR --data wn18rr --noltr

## DistMult
python pretrain.py --score_func distmult --opn mult --n_layer 1 --init_dim 200 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --name wn_noLTR --data wn18rr --noltr

## ConvE
python pretrain.py --score_func conve --opn mult --n_layer 1 --init_dim 200 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --name wn_noLTR --data wn18rr --noltr
