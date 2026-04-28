# Power-Link
This is an official re-implementation of the paper:

> **Path-based Explanation for Knowledge Graph Completion** (KDD '24)
> *Heng Chang, Jiangnan Ye, Alejo Lopez-Avila, Jinhua Du, Jia Li.*
> [arXiv:2401.02290](https://arxiv.org/abs/2401.02290).

The algorithmic core lives in
[`power_link/explainer.py:powerlink_path_loss`](power_link/explainer.py) —
about 100 lines of sparse-tensor manipulation. See
[`docs/method.md`](docs/method.md) for the math.

## Layout

```
power-link-master/
├── pretrain.py             # Train a CompGCN/RGCN/WGCN + TransE/DistMult/ConvE KGC model
├── run_powerlink.py        # Run PowerLinkExplainer on a saved checkpoint
├── run_gnnexp.py           # GNNExplainer-Link baseline
├── run_pgexp.py            # PGExplainer-Link baseline
├── evaluate.py             # Compute Fidelity+/Fidelity-, HΔR, sparsity from explanations
├── power_link/             # The library
│   ├── explainer.py        # PowerLinkExplainer
│   ├── baselines.py        # GNNExplainer-Link / PGExplainer-Link
│   ├── kge/                # GCN encoders (CompGCN/RGCN/WGCN) + decoders (TransE/DistMult/ConvE)
│   └── utils/              # Graph, paths (Dijkstra/Yen), eval, seeding
├── scripts/                # Reproducibility shell scripts
├── datasets/               # FB15k-237, WN18RR, YAGO
├── tests/                  # pytest unit + smoke tests
└── docs/                   # method walkthrough + reproduction guide
```

## Install

We use [uv](https://docs.astral.sh/uv/) for dependency management. PyTorch and
DGL ship CPU + CUDA build variants and need to be installed against the GPU
toolkit available on your system.

First, install the hardware-invariant dependencies via `uv`.
```bash
# 1. Clone & set up the env (Python 3.9 or 3.10).
uv sync
```

### For Pre-Ampere GPUs (V100, P100, RTX 2000-series, etc. — sm ≤ 70)

Use the paper's exact stack:

```bash
uv pip install --extra-index-url https://download.pytorch.org/whl/cu102 \
    torch==1.12.1+cu102
uv pip install dgl-cu102==0.9.1.post1 -f https://data.dgl.ai/wheels/repo.html
```

### For Ampere GPUs (A100, A6000, RTX 30-series, etc. — sm 80+)

DGL 0.9.1 doesn't ship a wheel for CUDA ≥ 11.x, so we use DGL 1.0.0:

```bash
uv pip install --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch==1.12.1+cu113
uv pip install dgl==1.0.0+cu113 -f https://data.dgl.ai/wheels/cu113/repo.html
```

The explainer handles the API difference in `g.adj()` (DGL 0.9.x returns a
torch sparse tensor, 1.x returns `dgl.sparse.SparseMatrix`) — both stacks
produce the same numbers.

### For CPU-only

Drop the `+cuXXX` suffix and the extra index URL: `pip install torch==1.12.1`
and `pip install dgl==0.9.1`. Useful for running tests or working with toy
graphs; pretraining FB15k-237 / WN18RR on CPU is impractical.

## Quick and simple reproduce

```bash
# 1. Pretrain CompGCN+TransE on FB15k-237 (~4 hours on a single A100).
python pretrain.py --score_func transe --opn mult --gpu 0 --gamma 9 \
    --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 2 \
    --encoder compgcn --data fb15k237

# 2. Run the explainer over saved test triplets.
python run_powerlink.py \
    --kge_model_config_path saved_models/config_compgcn_transe_fb15k237.json \
    --num_hops 1 --k_core 2 --num_paths 40 --num_epochs 50 --lr 0.005 \
    --max_num_samples 500 --regularisation_weight 0.02 \
    --save_explanation

# 3. Compute Fidelity+/Fidelity-, HΔR, sparsity over the saved explanations.
python evaluate.py \
    --kge_model_config_path saved_models/config_compgcn_transe_fb15k237.json \
    --explain_results_dir saved_explanations_experiment \
    --num_paths 5
```

For the full reproduction (all 18 (encoder, decoder, dataset) combos and the
ablation tables), see [`docs/reproduce.md`](docs/reproduce.md). The shell
scripts under [`scripts/`](scripts) bundle each experiment as a one-liner.

## Tests

```bash
uv pip install -e '.[dev]'
pytest tests/
```

The unit test for `powerlink_path_loss` checks the maths of the powering
trick on a hand-built 5-node graph; the smoke test exercises the full
`PowerLinkExplainer.explain(...)` pipeline on a 10-node synthetic KG.

## Citing

```bibtex
@inproceedings{chang2024powerlink,
  title     = {Path-based Explanation for Knowledge Graph Completion},
  author    = {Chang, Heng and Ye, Jiangnan and Lopez-Avila, Alejo and Du, Jinhua and Li, Jia},
  booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2024},
}
```

## Acknowledgements

Power-Link builds on the path-based-explanation idea introduced in
**PaGE-Link**. The KGE pretraining stack is adapted
from **"Rethinking Graph Convolutional Networks in Knowledge Graph
Completion"**. The path-finding code
(bidirectional Dijkstra, Yen's k-shortest paths) is adapted from NetworkX.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
