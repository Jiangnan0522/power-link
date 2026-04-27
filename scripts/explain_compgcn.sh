# FB15K237
## CompGCN-TransE
python run_powerlink.py --num_hops=1 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --device_id=0 --max_num_samples=500 --save_explanation --kge_model_config_path=./saved_models/config_compgcn_transe_fb15k237.json --regularisation_weight=0.02 

## CompGCN-DistMult
python run_powerlink.py --num_hops=1 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.001  --device_id=0 --max_num_samples=500 --save_explanation --kge_model_config_path=./saved_models/config_compgcn_distmult_fb15k237.json  --regularisation_weight=0.03

## CompGCN-ConvE
python run_powerlink.py --num_hops=1 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --device_id=0 --max_num_samples=500 --save_explanation --kge_model_config_path=./saved_models/config_compgcn_conve_fb15k237.json  --regularisation_weight=0.001  --power_order=3 --comp_g_size_limit=1000


# WN18-RR
## CompGCN-TransE
python run_powerlink.py --num_hops=3 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --device_id=0 --max_num_samples=200 --save_explanation --kge_model_config_path="./saved_models/config_compgcn_transe_wn18rr.json" --regularisation_weight=0.03 --hit1

## CompGCN-DistMult
python run_powerlink.py --num_hops=3 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_compgcn_distmult_wn18rr.json --device_id=0 --max_num_samples=200 --save_explanation --regularisation_weight=0.1  --comp_g_size_limit=2000 --hit1

## CompGCN-Conve
python run_powerlink.py --num_hops=3 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_compgcn_conve_wn18rr.json --device_id=0 --max_num_samples=200 --save_explanation --regularisation_weight=0.1  --comp_g_size_limit=2000 --hit1