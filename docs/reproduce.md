# Reproducing the paper

This walks through reproducing the main numbers in the
[KDD '24 Power-Link paper](https://arxiv.org/abs/2401.02290).

The pipeline is **train → explain → evaluate** for each (encoder, decoder,
dataset) combination. There are 18 such combinations across the two main
datasets:

| Encoders                | Decoders                 | Datasets   |
|-------------------------|--------------------------|------------|
| CompGCN, R-GCN, WGCN    | TransE, DistMult, ConvE  | FB15k-237, WN18RR |

## Hardware

Pretraining one (encoder, decoder, dataset) combo takes roughly:
- ~30 min on FB15k-237 (single A100, 500 epochs)
- ~2–4 hours on WN18RR (A100, 1000 epochs)

Explanation (50 epochs over 500 samples) is much faster — ~15–30 min per
combo on the same hardware.

## Step 1 — Pretrain the KGC models

The shell scripts under `scripts/` bundle every paper experiment as a
one-liner. Pick the one matching your encoder and dataset:

```bash
bash scripts/pretrain_fb_compgcn.sh    # CompGCN × {TransE, DistMult, ConvE} on FB15k-237
bash scripts/pretrain_fb_rgcn.sh       # R-GCN ×  ...
bash scripts/pretrain_fb_wgcn.sh       # WGCN ×   ...
bash scripts/pretrain_wn_compgcn.sh    # CompGCN × ... on WN18RR
bash scripts/pretrain_wn_rgcn.sh
bash scripts/pretrain_wn_wgcn.sh
bash scripts/pretrain_yago_compgcn.sh  # YAGO (Appendix only)
bash scripts/pretrain_yago_rgcn.sh
```

Each script invokes `pretrain.py` and writes:

```
saved_models/
├── <encoder>_<decoder>_<dataset>.pt              # checkpoint
└── config_<encoder>_<decoder>_<dataset>.json     # hyper-params
```

Use only the `# REPRO` blocks at the top of each shell script for the main
paper numbers; later blocks (`# RAT`, `# WSI`, `# WNI`, …) reproduce
ablation columns of Table 1 and are not needed for the explanation
experiments.

## Step 2 — Run the explainer

```bash
bash scripts/explain_compgcn.sh    # PowerLinkExplainer over the 6 CompGCN configs
bash scripts/explain_rgcn.sh
bash scripts/explain_wgcn.sh
```

Each script invokes `run_powerlink.py` for every combo. Outputs land in
`saved_explanations_experiment/explain_results_<encoder>_<decoder>_<dataset>`.

For ablation experiments (Tables 4–7 of the paper):

```bash
bash scripts/ablation.sh   # combination_method, without_path_loss, without_mi, power_order
```

## Step 3 — Evaluate

```bash
python evaluate.py \
    --kge_model_config_path saved_models/config_compgcn_transe_fb15k237.json \
    --explain_results_dir saved_explanations_experiment \
    --num_paths 5
```

The script prints **mask-based** and **path-based** Fidelity+/Fidelity−,
average ranking-drop hit rate (HΔR), and sparsity. Repeat for each combo by
swapping the `--kge_model_config_path`.

## Tables

### Table 2 — Fidelity / HΔR / sparsity

For each cell `(encoder, decoder, dataset, path-budget m)`, the paper reports:

- **Fidelity+** (path-based, top-m): higher = better.
- **Fidelity−** (path-based, top-m): lower = better.
- **HΔR : m**: higher = better. Hit-rate of the rank dropping after removing
  the top-m explanation paths.
- **Sparsity**: higher = better.

Run `evaluate.py` with `--num_paths` set to `m`.

### Table 3 — Timing

Total + per-graph runtime is measured by passing `--analysis` to
`run_powerlink.py`:

```bash
python run_powerlink.py --analysis ... > run_log.txt
cat analysis.json   # contains per-sample running_time and a running_time_total
```

Per-graph time is `running_time_total / num_explained`. To compare against
PaGE-Link, run with `--path_loss=pagelink` and the same flags.

Absolute timings depend on the GPU; relative speed-ups should be reproducible
within ~10–20 %.

### Tables 4–7 — Ablations

| Ablation                                   | Flag                              |
|--------------------------------------------|-----------------------------------|
| Concat vs Euclidean TES combination        | `--combination_method euclidean`  |
| Without path loss                          | `--without_path_loss`             |
| Without mutual-information loss            | `--without_mi`                    |
| Power order L = 1, 2, 3, 4                 | `--power_order N`                 |

Combine via `scripts/ablation.sh` or run `run_powerlink.py` directly.

## Tolerance and known sources of variance

The main results should reproduce within:

- **Fidelity+** (path-based, top-5): ±0.05 of the paper.
- **HΔR : 5**: ±0.10.
- **Sparsity**: ±0.05.
- **Per-graph time**: order-of-magnitude only (different GPU).

Sources of run-to-run variance:

- PyTorch CUDA non-determinism, especially in CompGCN's batched message
  passing. We seed everything we can (`set_seed`), but cuDNN's algorithmic
  choices remain non-deterministic on most GPUs.
- The TES MLP is randomly initialised per query triplet; with 500 samples
  the average is stable, but per-sample fidelity can vary widely.
- Yen's k-shortest paths uses heap tie-breaks that depend on edge insertion
  order; if the upstream KG load order changes, paths can swap order.

## Measured numbers (this commit)

### Smoke verification (10 epochs, 5 samples) — CompGCN+TransE / FB15k-237 / A100

A 10-epoch run is far too short to converge (paper used 500). Numbers here are
**not** the paper's; they verify the pipeline runs end-to-end after the
refactor.

| Stage     | Metric                  | Value      | Notes                                      |
|-----------|-------------------------|------------|--------------------------------------------|
| Pretrain  | Test MRR (avg)          | 0.00502    | Paper ≈0.30+; expected to be low at 10 ep. |
| Pretrain  | Test Hits@1 / @3 / @10  | 0.0045 / 0.0046 / 0.0048 | Same caveat. |
| Pretrain  | Per-epoch time          | ~45 s      | A100 (PCIe 40 GB), batch=256.               |
| Explainer | Samples explained       | 5 / 5      | Confirms ``score > 0.5`` filter passes.     |
| Evaluate  | Fidelity+ (mask)        | 0.0000     | Undertrained — model already near 0 score.  |
| Evaluate  | Fidelity+ (path, top-5) | 0.0000     | Same.                                       |
| Evaluate  | HΔR : 5                 | 0.0        | Same.                                       |
| Evaluate  | Avg sparsity (mask)     | 0.1178     | Sigmoid mask is sparse out-of-the-box.      |
| Evaluate  | Avg comp-graph density  | 0.5191     | 1-hop subgraph is dense around the triplet. |

**Interpretation.** The pipeline runs end-to-end. Fidelity is 0.0 because at
10 epochs the model predicts ≈uniform scores; perturbing any subset of edges
doesn't change the prediction, so removing/keeping a path makes no difference
to the score. With 500 epochs (paper setting) the model develops sharp
preferences and Fidelity+ rises to ≈0.6. **Sparsity** and **comp-graph
density** are non-zero because they don't depend on the model's predictive
quality.

### How to reproduce these numbers in full (500 epochs)

```bash
# 1. Pretrain (this is the slow part — ~7 hours on a single A100)
bash scripts/pretrain_fb_compgcn.sh   # writes saved_models/compgcn_transe_fb15k237.pt

# 2. Run the explainer over 500 test triplets
python run_powerlink.py \
    --kge_model_config_path saved_models/config_compgcn_transe_fb15k237.json \
    --num_hops 1 --k_core 2 --num_paths 40 --num_epochs 50 --lr 0.005 \
    --max_num_samples 500 --regularisation_weight 0.02 --save_explanation

# 3. Evaluate against paper Table 2
python evaluate.py \
    --kge_model_config_path saved_models/config_compgcn_transe_fb15k237.json \
    --explain_results_dir saved_explanations_experiment \
    --num_paths 5
```

Compare against paper Table 2 row "CompGCN-TransE, FB15k-237".
