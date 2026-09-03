# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from opendde.metrics.clash import Clash
from opendde.model.sample_confidence import compute_full_data_and_summary
from opendde.model.shape_complementarity import (
    build_shape_comp_pred_outputs,
    compute_shape_complementarity_fields,
    shape_comp_pred_uses_structural_tokens,
    structural_shape_comp_feature_dict,
    summarize_shape_comp_pair,
)
from opendde.model.structural_tokens import StructuralTokenExpander
from tests.parity.common import assert_close, load_case, load_module

pytestmark = pytest.mark.parity


class _Model(nn.Module):
    """Parent module so parameter paths match the released checkpoint keys."""

    def __init__(self, expander: StructuralTokenExpander) -> None:
        super().__init__()
        self.structural_token_expander = expander


def _inputs(ref: dict[str, np.ndarray], prefix: str = "in:") -> dict[str, mx.array]:
    return {k[len(prefix) :]: mx.array(v) for k, v in ref.items() if k.startswith(prefix)}


@pytest.mark.parametrize("mode", ["full", "none"])
@pytest.mark.parametrize("pair_chunk_size", [None, 16])
def test_structural_token_expander(mode, pair_chunk_size):
    ref = load_case("structural")
    feat = _inputs(ref)
    s_inputs, s, z = feat.pop("s_inputs_res"), feat.pop("s_res"), feat.pop("z_res")
    mod = StructuralTokenExpander(
        c_s=32, c_z=16, c_s_inputs=24, pair_projection_mode=mode, pair_chunk_size=pair_chunk_size
    )
    load_module(_Model(mod), ref, mode)
    s_inputs_struct, s_struct, z_struct, pair_features = mod(feat, s_inputs, s, z)
    assert_close(s_inputs_struct, ref[f"{mode}.out:s_inputs_struct"])
    assert_close(s_struct, ref[f"{mode}.out:s_struct"])
    assert_close(z_struct, ref[f"{mode}.out:z_struct"])
    for key in (
        "structural_pair_attn_bias",
        "same_parent_residue",
        "same_residue_twin",
        "prev_bb_chain",
        "next_bb_chain",
        "role_pair_type",
        "same_chain",
        "same_polymer_type",
        "residue_index",
    ):
        assert_close(pair_features[key], ref[f"{mode}.out:{key}"])


@pytest.mark.parametrize("token_space", ["residue", "structural"])
@pytest.mark.parametrize("pair_chunk_size", [128, 7])
def test_shape_complementarity_fields(token_space, pair_chunk_size):
    ref = load_case("shape_comp")
    feat = _inputs(ref)
    coord, atom_mask = feat.pop("coordinate"), feat.pop("atom_mask")
    if token_space == "structural":
        feat = structural_shape_comp_feature_dict(feat)
    fields = compute_shape_complementarity_fields(
        coord, feat, atom_mask=atom_mask, pair_chunk_size=pair_chunk_size
    )
    for key, value in fields.items():
        assert_close(value, ref[f"{token_space}.out:{key}"])
    pair_mean, pair_topk_mean, valid_frac = summarize_shape_comp_pair(
        fields["shape_comp_pair"], fields["shape_comp_pair_mask"]
    )
    assert_close(pair_mean, ref[f"{token_space}.out:summary_pair_mean"])
    assert_close(pair_topk_mean, ref[f"{token_space}.out:summary_pair_topk_mean"])
    assert_close(valid_frac, ref[f"{token_space}.out:summary_valid_pair_frac"])
    pred = build_shape_comp_pred_outputs(fields, keep_pair_map=True)
    assert_close(pred["shape_comp_pair_pred"], ref[f"{token_space}.out:shape_comp_pair"])
    uses_structural = shape_comp_pred_uses_structural_tokens(_inputs(ref), pred)
    assert uses_structural == (token_space == "structural")


def _confidence_configs() -> SimpleNamespace:
    def bins(lo, hi, n):
        return SimpleNamespace(min_bin=lo, max_bin=hi, no_bins=n)

    return SimpleNamespace(
        confidence=SimpleNamespace(
            plddt=bins(0.0, 1.0, 50), pde=bins(0.0, 32.0, 64), pae=bins(0.0, 32.0, 64)
        ),
        metrics=SimpleNamespace(clash=SimpleNamespace(af3_clash_threshold=1.1)),
    )


def test_compute_full_data_and_summary():
    ref = load_case("confidence")
    summary, full_data = compute_full_data_and_summary(
        _confidence_configs(), **_inputs(ref), N_recycle=3, return_full_data=True
    )
    assert len(summary) == len(full_data) == 2
    for i, (summary_i, full_i) in enumerate(zip(summary, full_data)):
        for key, value in summary_i.items():
            expected = ref[f"summary{i}:{key}"]
            assert np.asarray(value).shape == expected.shape, key
            if expected.dtype == bool:
                assert np.array_equal(value, expected), key
            else:
                assert_close(mx.array(np.asarray(value, dtype=np.float32)), expected)
        assert isinstance(summary_i["has_clash"], (bool, np.bool_))
        for key, value in full_i.items():
            expected = ref[f"full{i}:{key}"]
            assert_close(mx.array(np.asarray(value, dtype=np.float32)), expected.astype(np.float32))


def test_clash():
    ref = load_case("clash")
    result = Clash()(**_inputs(ref))
    assert np.array_equal(result["summary"]["af3_clash"], ref["summary:af3_clash"])
    assert np.array_equal(result["summary"]["vdw_clash"], ref["summary:vdw_clash"])
    assert result["summary"]["chain_types"] == list(ref["summary:chain_types"])
    assert np.array_equal(
        np.array(result["summary"]["skipped_pairs"]), ref["summary:skipped_pairs"]
    )
    assert_close(mx.array(result["details"]["af3_clash"]), ref["details:af3_clash"])
    vdw_details = result["details"]["vdw_clash"]
    expected_keys = {
        tuple(int(x) for x in k.split(":")[-1].split("_"))
        for k in ref
        if k.startswith("details:vdw_clash:")
    }
    assert set(vdw_details) == expected_keys
    for (s, i, j), pairs in vdw_details.items():
        assert_close(mx.array(pairs.astype(np.float32)), ref[f"details:vdw_clash:{s}_{i}_{j}"])
