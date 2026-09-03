# SPDX-License-Identifier: Apache-2.0
import json
import os

import pytest
from click.testing import CliRunner

from opendde.config.inference import build_inference_config
from runner.cli import opendde_cli

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_doctor_and_help():
    runner = CliRunner()
    assert "Apple" in runner.invoke(opendde_cli, ["doctor"]).output
    result = runner.invoke(opendde_cli, ["pred", "--help"])
    assert result.exit_code == 0 and "--dtype" in result.output


def test_tojson(tmp_path):
    ccd = build_inference_config().data.ccd_components_file
    if not os.path.exists(ccd):
        pytest.skip(f"CCD assets missing: {ccd}")
    pdb = os.path.join(REPO, "examples", "7pzb.pdb")
    result = CliRunner().invoke(opendde_cli, ["json", "-i", pdb, "-o", str(tmp_path)])
    assert result.exit_code == 0, result.output
    jobs = json.load(open(tmp_path / "7pzb.json"))
    kinds = [next(iter(entity)) for entity in jobs[0]["sequences"]]
    assert kinds.count("proteinChain") == 2 and "ligand" in kinds
