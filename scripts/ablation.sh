# TES combine method: concat VS euclidean
### TransE euc [114]
python run_powerlink.py --num_hops=2 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_wgcn_transe_fb15k237.json --device_id=0 --max_num_samples=500 --save_explanation --regularisation_weight=0.03  --comp_g_size_limit=5000 --combination_method=euclidean
python run_powerlink.py --num_hops=2 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_rgcn_transe_fb15k237.json --device_id=0 --max_num_samples=500 --save_explanation --regularisation_weight=0.03  --comp_g_size_limit=5000 --combination_method=euclidean

# Ablation study of Power-Link
## MI:√, Path Loss:×  [87]
python run_powerlink.py --num_hops=2 --num_paths=40 --k_core=2 --num_epochs=50 --lr=0.005  --kge_model_config_path=./saved_models/config_rgcn_transe_fb15k237.json --device_id=0 --max_num_samples=500 --save_explanation --regularisation_weight=0.03  --comp_g_size_limit=5000 --combination_method=concat --without_path_loss
