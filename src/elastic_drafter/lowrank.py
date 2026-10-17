"""Make a model cheaper by replacing its linear layers with low-rank factors.

A weight matrix W (m-by-n) is replaced by two thin matrices B (m-by-r) and A (r-by-n) with
W ~ B A. Multiplying an input by W costs m*n multiply-adds; by B and A it costs r*(m+n). So the
factors are cheaper only when r < m*n / (m+n), which is n/2 for a square matrix. We therefore
choose r from the fraction of cost to keep, not from a fraction of the rank.

Which r components to keep is set by a scaling applied before the SVD and removed after it:
  - none: plain truncated SVD (README E2);
  - `right`, an n-by-n matrix: the Cholesky factor of the second-moment matrix of the layer's
    inputs, i.e. whitened SVD (SVD-LLM, https://arxiv.org/abs/2403.07378; E3);
  - `row_scale`, a length-m vector: the square root of each output row's importance, e.g. summed
    squared gradients of a loss (FWSVD, https://arxiv.org/abs/2207.00112; E4 and E5).
"""
import torch
from torch import nn


def rank_for_cost(m, n, fraction):
    """The rank at which the factors of an m-by-n matrix cost `fraction` of the full matrix."""
    return max(1, int(fraction * m * n / (m + n)))


def factor(W, r, row_scale=None, right=None):
    """Rank-r factors (A, B) of W, best in the norm || diag(row_scale) (W - B A) right ||."""
    M = W.float()
    if row_scale is not None:
        M = row_scale.float()[:, None] * M
    if right is not None:
        M = M @ right.float()
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    root = S[:r].sqrt()
    B = U[:, :r] * root
    A = root[:, None] * Vh[:r]
    if row_scale is not None:
        B = B / row_scale.float()[:, None]
    if right is not None:  # A <- A right^{-1}
        A = torch.linalg.solve(right.float().T, A.T).T
    return A.to(W.dtype), B.to(W.dtype)


class LowRankLinear(nn.Module):
    """y = B (A x) + bias."""

    def __init__(self, A, B, bias=None):
        super().__init__()
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.bias = None if bias is None else nn.Parameter(bias)

    def forward(self, x):
        y = (x @ self.A.T) @ self.B.T
        return y if self.bias is None else y + self.bias


def block_linears(model):
    """(name, layer) for every linear layer inside the transformer blocks. The embedding table and
    the output layer are left as they are."""
    return [(name, mod) for name, mod in model.named_modules()
            if isinstance(mod, (nn.Linear, LowRankLinear)) and ".layers." in name]


def cost(model):
    """Multiply-adds per token of the linear layers inside the transformer blocks."""
    total = 0
    for _, mod in block_linears(model):
        if isinstance(mod, LowRankLinear):
            total += mod.A.shape[0] * (mod.A.shape[1] + mod.B.shape[0])
        else:
            total += mod.weight.numel()
    return total


def shrink(model, fraction, scales=None):
    """Replace every linear layer inside the transformer blocks with factors that keep `fraction` of
    its cost. `scales` maps a layer name to (row_scale, right); a missing name means plain SVD."""
    scales = scales or {}
    for name, lin in block_linears(model):
        m, n = lin.weight.shape
        A, B = factor(lin.weight.data, rank_for_cost(m, n, fraction), *scales.get(name, (None, None)))
        bias = None if lin.bias is None else lin.bias.data
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child, LowRankLinear(A, B, bias))
    return model


if __name__ == "__main__":
    # Checks that pass or fail: full-rank factors reproduce W under every scaling, and shrinking a
    # small random Llama keeps the requested fraction of cost and still runs.
    torch.manual_seed(0)
    W = torch.randn(64, 48)
    X = torch.randn(500, 48)
    right = torch.linalg.cholesky(X.T @ X / len(X))
    row_scale = torch.rand(64) + 0.1
    for rs, R in [(None, None), (None, right), (row_scale, None)]:
        A, B = factor(W, 48, rs, R)
        assert torch.allclose(B @ A, W, atol=1e-4), "full-rank factors must reproduce W"
    assert rank_for_cost(100, 100, 1.0) == 50

    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(hidden_size=128, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=1000)
    model = LlamaForCausalLM(cfg).eval()
    full = cost(model)
    shrink(model, 0.5)
    ratio = cost(model) / full
    assert 0.45 < ratio <= 0.5, ratio
    model(input_ids=torch.tensor([[1, 2, 3]]))
    print(f"lowrank checks pass; shrunk model keeps {ratio:.3f} of block cost")
