# SPDX-License-Identifier: Apache-2.0
"""Compare the NumPy featurization pipeline against the original PyTorch pipeline."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from opendde.config.inference import build_inference_config

REPO_DIR = Path(__file__).resolve().parents[1]
TORCH_REPO_DIR = Path(os.environ.get("OPENDDE_TORCH_REPO", "/Users/mrzz/Documents/github/OpenDDE"))
INPUT_JSON = REPO_DIR / "examples" / "example_without_msa.json"
DIMENSION_KEYS = ("N_asym", "N_token", "N_atom", "N_msa", "N_prot_atom", "N_lig_atom", "N_dna_atom")

REFERENCE_SCRIPT = """
import sys
sys.path.insert(0, {torch_repo!r})
import numpy as np
from opendde.config.inference import build_inference_config
from opendde.data.inference.infer_dataloader import InferenceDataset

configs = build_inference_config()
configs.input_json_path = {input_json!r}
configs.use_msa = False
configs.use_template = False
configs.use_rna_msa = False
np.random.seed(0)
data, _, error = InferenceDataset(configs)[0]
assert error == "", error
arrays = {{k: v.numpy() for k, v in data["input_feature_dict"].items()}}
arrays.update({{k: np.asarray(data[k]).reshape(()) for k in {dimension_keys!r}}})
np.savez({output!r}, **arrays)
"""


def _featurize_new(configs):
    from opendde.data.inference.infer_dataloader import InferenceDataset

    np.random.seed(0)
    data, atom_array, error = InferenceDataset(configs)[0]
    assert error == "", error
    return data


@pytest.fixture(scope="module")
def configs():
    configs = build_inference_config()
    configs.input_json_path = str(INPUT_JSON)
    configs.use_msa = False
    configs.use_template = False
    configs.use_rna_msa = False
    if not os.path.exists(configs.data.ccd_components_file):
        pytest.skip(f"CCD assets missing: {configs.data.ccd_components_file}")
    return configs


@pytest.fixture(scope="module")
def reference(configs, tmp_path_factory):
    if not (TORCH_REPO_DIR / "opendde").is_dir():
        pytest.skip(f"original PyTorch repo not found at {TORCH_REPO_DIR}")
    output = tmp_path_factory.mktemp("reference") / "features.npz"
    script = textwrap.dedent(REFERENCE_SCRIPT).format(
        torch_repo=str(TORCH_REPO_DIR),
        input_json=str(INPUT_JSON),
        dimension_keys=DIMENSION_KEYS,
        output=str(output),
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=str(TORCH_REPO_DIR))
    return dict(np.load(output))


def test_features_match_torch_pipeline(configs, reference):
    data = _featurize_new(configs)
    features = data["input_feature_dict"]

    for key in DIMENSION_KEYS:
        assert isinstance(data[key], int), key
        assert data[key] == int(reference.pop(key)), key

    assert set(features) == set(reference), set(features) ^ set(reference)
    for key, expected in reference.items():
        actual = features[key]
        assert isinstance(actual, np.ndarray), key
        assert actual.dtype == expected.dtype, f"{key}: {actual.dtype} vs {expected.dtype}"
        assert actual.shape == expected.shape, f"{key}: {actual.shape} vs {expected.shape}"
        np.testing.assert_array_equal(actual, expected, err_msg=key)
