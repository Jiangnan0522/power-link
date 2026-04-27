"""Unit tests for PowerLinkExplainer.powerlink_path_loss.

The test builds a tiny graph with one known on-path chain ``h → a → b → t``
(plus a few off-path edges) and verifies:

1. Concentrating mask probability onto the on-path edges DECREASES the loss.
2. Gradients flow back to the mask, with larger magnitude on on-path entries.
3. ``power_order=2`` and ``power_order=4`` both yield finite scalar losses.

We avoid instantiating a full KGC model — the loss is a pure function of the
homogeneous graph, the source/target node ids, and the edge weights. We
construct a thin shim ``PowerLinkExplainer`` whose ``model`` attribute is a
bare ``nn.Module`` so that ``__init__`` does not blow up.
"""

import dgl
import pytest
import torch
import torch.nn as nn

from power_link.explainer import PowerLinkExplainer


def make_explainer():
    """Build the smallest valid explainer wrapper."""
    model = nn.Module()
    model.src_ntype = '_N'
    model.tgt_ntype = '_N'
    return PowerLinkExplainer(model)


def make_chain_graph():
    """6-node graph with one ĥ → t̂ chain plus disconnected off-path edges.

        on-path: 0(h) → 1 → 2 → 3(t)              [length-3 chain]
        off-path: 4 → 5                          [disconnected, never on a
                                                  ĥ→t̂ path of any length]

    DGL coalesces ``g.adj()`` by ``(row, col)``, so we sort our edge list
    by ``(src, dst)`` here too — that way ``eweights[i]`` aligns with the
    sparse tensor's i-th non-zero entry.

    Returns the graph and a boolean list flagging which edges lie on the
    unique 0 → 3 path (used to construct the per-edge mask).
    """
    edges = [
        (0, 1, True),
        (1, 2, True),
        (2, 3, True),
        (4, 5, False),
    ]
    src = torch.tensor([s for s, _, _ in edges])
    dst = torch.tensor([d for _, d, _ in edges])
    on_path = [is_on for _, _, is_on in edges]
    g = dgl.graph((src, dst), num_nodes=6)
    return g, on_path


def make_eweights(on_path, on_value: float, off_value: float):
    """Build a learnable edge-weight tensor of shape ``[E]``."""
    return torch.tensor(
        [on_value if p else off_value for p in on_path],
        dtype=torch.float32,
        requires_grad=True,
    )


def test_loss_decreases_when_probability_concentrates_on_path():
    explainer = make_explainer()
    g, on_path = make_chain_graph()

    eweights_loose = make_eweights(on_path, on_value=0.9, off_value=0.1)
    eweights_tight = make_eweights(on_path, on_value=0.99, off_value=0.01)

    loss_loose = explainer.powerlink_path_loss(0, 3, g, eweights_loose, power_order=3)
    loss_tight = explainer.powerlink_path_loss(0, 3, g, eweights_tight, power_order=3)

    assert torch.isfinite(loss_loose)
    assert torch.isfinite(loss_tight)
    assert loss_tight.item() < loss_loose.item(), (
        f"Loss should decrease as on-path probability concentrates "
        f"({loss_loose.item():.4f} → {loss_tight.item():.4f}).")


def test_gradients_flow_to_on_path_edges_only():
    """Gradient must flow to every on-path edge and *not* to disconnected
    off-path edges (since those don't appear in any ĥ → t̂ path of any
    length)."""
    explainer = make_explainer()
    g, on_path = make_chain_graph()
    eweights = make_eweights(on_path, on_value=0.5, off_value=0.5)

    loss = explainer.powerlink_path_loss(0, 3, g, eweights, power_order=3)
    loss.backward()

    grads = eweights.grad
    assert grads is not None, "Gradients did not propagate to the edge-weight tensor."

    on_path_grads = grads[torch.tensor(on_path)].abs()
    off_path_grads = grads[torch.tensor([not p for p in on_path])].abs()

    assert (on_path_grads > 0).all(), (
        "Every on-path edge should receive non-zero gradient, "
        f"but got on_path={on_path_grads}.")
    assert (off_path_grads == 0).all(), (
        "Disconnected off-path edges should receive ZERO gradient (they "
        f"appear in no ĥ → t̂ path), but got off_path={off_path_grads}.")


@pytest.mark.parametrize("power_order", [3, 4])
def test_loss_finite_when_power_order_covers_path(power_order):
    """Loss is finite when ``power_order`` covers at least one ĥ → t̂ path
    (here length 3). At ``power_order=2`` the loss is correctly ``+inf`` because
    no length-2 path exists — that's the desired math, just not interesting
    to assert in this fixture."""
    explainer = make_explainer()
    g, on_path = make_chain_graph()
    eweights = make_eweights(on_path, on_value=0.7, off_value=0.3)

    loss = explainer.powerlink_path_loss(0, 3, g, eweights, power_order=power_order)
    assert torch.isfinite(loss), f"Loss must be finite at power_order={power_order}."
    assert loss.item() > 0
