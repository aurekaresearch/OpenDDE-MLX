# SPDX-License-Identifier: Apache-2.0
"""Generate PyTorch reference outputs for numerical parity tests.

Run with the *original* OpenDDE repository on ``PYTHONPATH`` (PyTorch CPU):

    OPENDDE_TORCH_REPO=/path/to/OpenDDE python tests/parity/reference.py <case> out.npz

Each case builds a randomly initialised reference module, runs it on random
inputs and stores ``state_dict`` + inputs + outputs in one ``.npz`` file.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
os.environ.setdefault("LAYERNORM_TYPE", "torch")


def _state(module: torch.nn.Module) -> dict[str, np.ndarray]:
    return {f"param:{k}": v.detach().float().numpy() for k, v in module.state_dict().items()}


def _randomize(module: torch.nn.Module, std: float = 0.3) -> None:
    """Replace zero-initialised weights so every path contributes."""
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * std)


def case_primitives() -> dict[str, np.ndarray]:
    from opendde.model.modules.primitives import AdaptiveLayerNorm, Attention, Transition
    from opendde.model.modules.transformer import AttentionPairBias, ConditionedTransitionBlock

    out: dict[str, np.ndarray] = {}
    a = torch.randn(2, 70, 32)
    attn = Attention(c_q=32, c_k=32, c_v=32, c_hidden=8, num_heads=4)
    _randomize(attn)
    bias = torch.randn(2, 4, 70, 70)
    out.update({f"attn.{k}": v for k, v in _state(attn).items()})
    out["attn.in:a"] = a.numpy()
    out["attn.in:bias"] = bias.numpy()
    out["attn.out:dense"] = attn(a, a, attn_bias=bias).detach().numpy()
    tb = torch.randn(2, 4, 3, 32, 128)
    out["attn.in:trunk_bias"] = tb.numpy()
    out["attn.out:local"] = (
        attn(a, a, trunked_attn_bias=tb, n_queries=32, n_keys=128).detach().numpy()
    )

    trans = Transition(c_in=32, n=2)
    _randomize(trans)
    out.update({f"trans.{k}": v for k, v in _state(trans).items()})
    out["trans.out"] = trans(a).detach().numpy()

    s = torch.randn(2, 70, 16)
    adaln = AdaptiveLayerNorm(c_a=32, c_s=16)
    _randomize(adaln)
    out.update({f"adaln.{k}": v for k, v in _state(adaln).items()})
    out["adaln.in:s"] = s.numpy()
    out["adaln.out"] = adaln(a, s).detach().numpy()

    ctb = ConditionedTransitionBlock(c_a=32, c_s=16, n=2)
    _randomize(ctb)
    out.update({f"ctb.{k}": v for k, v in _state(ctb).items()})
    out["ctb.out"] = ctb(a, s).detach().numpy()

    z = torch.randn(70, 70, 24)
    apb = AttentionPairBias(has_s=True, n_heads=4, c_a=32, c_s=16, c_z=24)
    _randomize(apb)
    out.update({f"apb.{k}": v for k, v in _state(apb).items()})
    out["apb.in:z"] = z.numpy()
    out["apb.out"] = apb(a=a, s=s, z=z).detach().numpy()
    out["apb.out:fused"] = (
        apb(
            a=a,
            s=s,
            z=torch.nn.functional.layer_norm(z, (24,)).permute(2, 0, 1).contiguous(),
            enable_efficient_fusion=True,
        )
        .detach()
        .numpy()
    )
    apb_nos = AttentionPairBias(has_s=False, create_offset_ln_z=True, n_heads=4, c_a=32, c_z=24)
    _randomize(apb_nos)
    out.update({f"apb_nos.{k}": v for k, v in _state(apb_nos).items()})
    out["apb_nos.out"] = apb_nos(a=a, s=None, z=z).detach().numpy()
    return out


def case_triangular() -> dict[str, np.ndarray]:
    from opendde.model.triangular.layers import OuterProductMean
    from opendde.model.triangular.triangular import (
        TriangleAttention,
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )

    out: dict[str, np.ndarray] = {}
    z = torch.randn(40, 40, 32)
    out["in:z"] = z.numpy()
    for name, mod in (
        ("tmo", TriangleMultiplicationOutgoing(c_z=32, c_hidden=32)),
        ("tmi", TriangleMultiplicationIncoming(c_z=32, c_hidden=32)),
    ):
        _randomize(mod)
        out.update({f"{name}.{k}": v for k, v in _state(mod).items()})
        out[f"{name}.out"] = (
            mod(z.clone(), _add_with_inplace=False, inplace_safe=False).detach().numpy()
        )
    for name, mod in (
        ("tas", TriangleAttention(c_in=32, c_hidden=8, no_heads=4, starting=True)),
        ("tae", TriangleAttention(c_in=32, c_hidden=8, no_heads=4, starting=False)),
    ):
        _randomize(mod)
        out.update({f"{name}.{k}": v for k, v in _state(mod).items()})
        out[f"{name}.out"] = mod(z).detach().numpy()
        out[f"{name}.out:chunk"] = mod(z, chunk_size=16).detach().numpy()
    m = torch.randn(6, 40, 16)
    opm = OuterProductMean(c_m=16, c_z=32, c_hidden=8)
    _randomize(opm)
    out.update({f"opm.{k}": v for k, v in _state(opm).items()})
    out["opm.in:m"] = m.numpy()
    out["opm.out"] = opm(m).detach().numpy()
    return out


CASES = {"primitives": case_primitives, "triangular": case_triangular}


def main() -> None:
    repo = os.environ.get("OPENDDE_TORCH_REPO")
    if repo:
        sys.path.insert(0, repo)
    case, output = sys.argv[1], sys.argv[2]
    arrays = CASES[case]()
    np.savez(output, **arrays)
    print(f"wrote {len(arrays)} arrays to {output}")


if __name__ == "__main__":
    main()
