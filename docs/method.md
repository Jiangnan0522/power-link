# Method walkthrough

This document distils the algorithmic core of Power-Link — the **powering
trick** — and the surrounding pipeline. For the formal treatment see §5 of
the [KDD '24 paper](https://arxiv.org/abs/2401.02290).

## Setting

We are given a trained GNN-based KGC model `Φ(G, ·, ·)` and a query triplet
`⟨ĥ, r̂, t̂⟩` that the model predicts as factual. The goal is to find a small
set of multi-hop paths `ĥ → ... → t̂` in the KG that explain *why* `Φ`
predicts the triplet.

This is a **post-hoc, instance-level, path-based** explanation:
- *Post-hoc* — `Φ` is fixed and frozen; we don't retrain it.
- *Instance-level* — one explanation per query triplet.
- *Path-based* — the explanation is a list of multi-hop paths, not a sub-graph
  or a set of triplets.

## Pipeline

`PowerLinkExplainer.explain(...)` goes through three stages per query.

### 1. Computational subgraph extraction

`utils.graph.hetero_src_tgt_khop_in_subgraph` extracts the L-hop ego graph
around the union `{ĥ, t̂}`; this is the paper's `Gᵏ_c`. A k-core peeling pass
(`prune_max_degree`, `k_core` flags) drops peripheral low-degree nodes — both
to shrink the search space and to remove obvious noise.

### 2. Mask training (the powering trick)

We train a per-edge importance mask `M ∈ ℝ^{N×N}` parameterised by the
**Triplet Edge Scorer (TES)** — a small 3-layer MLP that takes
`(h_i, r_j, t_k, ĥ, r̂, t̂)` embeddings and produces a logit per edge. After
sigmoid, this becomes the entry `M[i, j]` of the weighted adjacency matrix.

The training objective combines three terms:

```
L_total = w · L_prediction  +  L_path  +  γ · ‖M‖₂
```

- `L_prediction` (mutual-information loss): `−log P_Φ(Y=1 | M ⊙ Gᵏ_c, ⟨ĥ, r̂, t̂⟩)`.
  The frozen KGC model is run with edges weighted by M; this ties M to what
  the model actually believes.
- `L_path` (the powering-trick path loss; see below).
- `γ · ‖M‖₂` regulariser.

Only TES parameters are updated; gradients flow through the sparse matrix
multiplications.

### 3. Path extraction

After TES is trained, run **Yen's k-shortest paths** (built on bidirectional
Dijkstra in `power_link/utils/paths.py`) on the homogeneous version of `Gᵏ_c`,
with edge cost `1 / sigmoid(TES(edge))`. Keep the top-`num_paths` paths of
length ≤ `max_path_length` from `ĥ` to `t̂`.

If no path exists (rare), the implementation falls back to "top-k highest-
scored edges as a single pseudo-path".

## The powering trick

`power_link.explainer.PowerLinkExplainer.powerlink_path_loss` is the central
novelty.

### What we want

A differentiable proxy for *the average probability of all on-path edges of
length 1..L between ĥ and t̂*. PaGE-Link computed this by running Dijkstra
inside the training loop — slow, CPU-only, single-threaded. Power-Link uses
sparse matrix powers instead, end-to-end on GPU.

### Math

Let `M ∈ ℝ^{N×N}` be the weighted adjacency: `M[i, j] = sigmoid(TES(edge i→j))`.

- `u = M[ĥ, :]` — the **power vector**, a single row of `M`.
- `u^(l) = u · M^(l−1)` is a row-vector whose entry at `t̂` is the **sum of
  products of edge probabilities on every length-`l` path** from `ĥ` to `t̂`.
- Let `A` be the binary adjacency. `a^(l)_t̂` is the number of length-`l`
  paths from `ĥ` to `t̂`.
- Therefore `(u^(l)_t̂ / a^(l)_t̂)^{1/l}` is the **geometric-mean per-edge
  probability** averaged over all length-`l` paths.
- Average across `l = 1..L`:

  ```
  P_on  =  (1 / (L − 1)) · Σ_{l = 1..L}  (u^(l)_t̂ / a^(l)_t̂)^{1/l}
  L_path  =  − log P_on
  ```

Maximising `P_on` pushes `M` to put high probability on edges that lie on
**some** short path between `ĥ` and `t̂` — exactly the desired behaviour.

### Why this is fast

- We never materialise `M^L`. Each iteration does **(1×N row) × (N×N sparse
  matrix)** → still 1×N. Cost per multiplication is `O(nnz(M))`.
- Total cost over `L` iterations: `O(L · |ℰᵏ_c|)`.
- Compare to PaGE-Link's per-epoch Dijkstra-based loss — roughly
  `O(|ℰᵏ_c| · log |ℰᵏ_c| · k)` per epoch, on CPU only.
- The whole computation is differentiable w.r.t. `M`'s values via
  `torch.sparse.mm`, so gradients flow back to TES.

### Implementation note

`dgl.sparse.coalesce` does not propagate gradients, so the implementation
uses `torch.sparse_coo_tensor(...).coalesce()` directly. See the comments
near `power_link/explainer.py` line ~440.

## Hyper-parameters

The flags exposed by `run_powerlink.py` (defaults shown; paper settings in
parentheses):

| Flag                       | Default | Paper (FB15k-237 / WN18RR) | Role |
|----------------------------|---------|----------------------------|------|
| `--num_hops`               | 2       | 1–2 / 3                    | L-hop subgraph extraction |
| `--k_core`                 | 2       | 2 / 2                      | Pruning depth |
| `--prune_max_degree`       | 200     | 200                        | Degree cap during pruning |
| `--num_epochs`             | 20      | 50                         | TES training epochs |
| `--lr`                     | 0.01    | 0.005                      | Adam LR for TES |
| `--num_paths`              | 40      | 40                         | Paths kept after Dijkstra |
| `--max_path_length`        | 5       | 5                          | Length cap during path extraction |
| `--power_order`            | 3       | 3                          | L in the powering trick |
| `--regularisation_weight`  | 0.001   | 0.02–0.15 (per encoder)    | γ in `‖M‖₂` |
| `--combination_method`     | concat  | concat (default), euclidean (TransE) | TES feature combiner |
| `--without_path_loss`      | False   | ablation only              | Disable `L_path` |
| `--without_mi`             | False   | ablation only              | Disable `L_prediction` |
| `--path_loss=power\|pagelink` | power | power (default), pagelink (baseline) | Path loss variant |
| `--comp_g_size_limit`      | 1000    | 1000 / 2000 / 5000         | Drop oversize subgraphs |
| `--hit1` / `--hit3`        | False   | hit1 on WN18RR             | Filter to top-1 / top-3 samples |
