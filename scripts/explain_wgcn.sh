# FB15K237
## WGCN-TransE
python run_powerlink.py --num_hops=2 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_wgcn_transe_fb15k237.json --device_id=0 --max_num_samples=500 --save_explanation --regularisation_weight=0.03  --comp_g_size_limit=5000

## WGCN-Distmult
python run_powerlink.py --num_hops=2 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_wgcn_distmult_fb15k237.json --device_id=0 --max_num_samples=500 --save_explanation --regularisation_weight=0.04  --comp_g_size_limit=5000

## WGCN-ConvE
python run_powerlink.py --num_hops=2 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_wgcn_conve_fb15k237.json --device_id=0 --max_num_samples=500 --save_explanation --regularisation_weight=0.03  --comp_g_size_limit=5000


# WN18RR
## WGCN-TransE
python run_powerlink.py --num_hops=3 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_wgcn_transe_wn18rr.json --device_id=0 --max_num_samples=200 --save_explanation --regularisation_weight=0.15  --comp_g_size_limit=5000 --hit3

## WGCN-Distmult
python run_powerlink.py --num_hops=3 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_wgcn_distmult_wn18rr.json --device_id=0 --max_num_samples=200 --save_explanation --regularisation_weight=0.15  --comp_g_size_limit=5000 --hit3

## WGCN-ConvE
python run_powerlink.py --num_hops=3 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_wgcn_conve_wn18rr.json --device_id=0 --max_num_samples=200 --save_explanation --regularisation_weight=0.15  --comp_g_size_limit=5000 --hit3
