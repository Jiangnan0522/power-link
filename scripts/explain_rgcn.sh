# FB15K237
## RGCN-TransE
python run_powerlink.py --num_hops=1 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_rgcn_transe_fb15k237.json --device_id=0 --max_num_samples=500 --save_explanation --regularisation_weight=0.03  --comp_g_size_limit=1000

## RGCN-DistMult
python run_powerlink.py --num_hops=1 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_rgcn_distmult_fb15k237.json --device_id=0 --max_num_samples=500 --save_explanation --regularisation_weight=0.03  --comp_g_size_limit=1000

## RGCN-ConvE
python run_powerlink.py --num_hops=1 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_rgcn_conve_fb15k237.json --device_id=0 --max_num_samples=500 --save_explanation --regularisation_weight=0.03  --comp_g_size_limit=1000


# WN18RR
# RGCN-TransE
python run_powerlink.py --num_hops=3 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_rgcn_transe_wn18rr.json --device_id=0 --max_num_samples=200 --save_explanation --regularisation_weight=0.03  --comp_g_size_limit=2000 --hit1

# RGCN-DistMult
python run_powerlink.py --num_hops=3 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_rgcn_distmult_wn18rr.json --device_id=0 --max_num_samples=200 --save_explanation --regularisation_weight=0.04  --comp_g_size_limit=2000 --hit1

# RGCN-ConvE
python run_powerlink.py --num_hops=3 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_rgcn_conve_wn18rr.json --device_id=0 --max_num_samples=200 --save_explanation --regularisation_weight=0.03  --comp_g_size_limit=2000 --hit1




