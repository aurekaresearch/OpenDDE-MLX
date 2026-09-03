# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Click commands: prediction, JSON conversion and MSA/template preprocessing."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import click
import tqdm
from Bio import SeqIO

from opendde.config.inference import build_inference_config, validate_inference_schedule
from opendde.config.model_registry import DEFAULT_MODEL_NAME, model_configs
from opendde.config.schema import INFERENCE_DTYPE_CHOICES, InferenceDtype, OpenDDEConfig
from opendde.data.inference.input_validation import validate_inference_jobs, validate_inference_seed
from opendde.data.inference.json_maker import cif_to_input_json
from opendde.data.tools import kalign
from opendde.data.utils import pdb_to_cif
from opendde.utils.logger import get_logger
from opendde.utils.logging_config import init_logging
from runner.cli import CONTEXT_SETTINGS
from runner.inference import InferenceRunner, infer_predict
from runner.msa_search import msa_search, update_infer_json
from runner.rna_msa_search import update_rna_msa_info
from runner.template_search import update_template_info

logger = get_logger(__name__)
SUPPORTED_MODELS = tuple(model_configs.keys())
_GENERATED_INPUT_SUFFIXES = ("-update-msa.json", "-final-updated.json")


def _discover_inference_jsons(json_file: str, out_dir: str) -> list[str]:
    """Source JSON paths under ``json_file`` excluding generated preprocessing outputs."""
    input_path = Path(json_file)
    if input_path.is_file():
        if input_path.suffix != ".json":
            raise RuntimeError(f"Inference input file must end with .json: {json_file}")
        return [str(input_path)]
    if not input_path.is_dir():
        raise RuntimeError(f"Can not read input file or directory: {json_file}")
    output_root = Path(out_dir).resolve()
    infer_jsons = sorted(
        str(path)
        for path in input_path.rglob("*.json")
        if path.is_file()
        and not path.name.endswith(_GENERATED_INPUT_SUFFIXES)
        and output_root not in path.resolve().parents
        and path.resolve() != output_root
    )
    if not infer_jsons:
        raise RuntimeError(f"Can not read a valid source JSON file in {json_file}")
    return infer_jsons


def _validate_input_collection(paths: list[str]) -> None:
    owners: dict[str, str] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for job in validate_inference_jobs(json.load(handle)):
                if job["name"] in owners:
                    raise ValueError(
                        f"Inference job name {job['name']!r} occurs in both {owners[job['name']]} "
                        f"and {path}; the outputs would collide."
                    )
                owners[job["name"]] = path


def preprocess_input(
    input_json: str,
    out_dir: str,
    use_msa: bool = True,
    use_template: bool = False,
    use_rna_msa: bool = False,
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
    nhmmer_binary_path: Optional[str] = None,
    hmmalign_binary_path: Optional[str] = None,
    hmmbuild_rna_binary_path: Optional[str] = None,
    ntrna_database_path: Optional[str] = None,
    rfam_database_path: Optional[str] = None,
    rna_central_database_path: Optional[str] = None,
    nhmmer_n_cpu: Optional[int] = None,
) -> str:
    """Run MSA / template / RNA-MSA searches and return the path of the updated JSON."""
    generated_json_dir = os.path.join(
        os.path.abspath(out_dir),
        ".opendde_preprocessed",
        hashlib.blake2b(os.path.abspath(input_json).encode("utf-8"), digest_size=8).hexdigest(),
    )
    if use_msa:
        msa_updated_json, _ = update_infer_json(
            input_json, out_dir, use_msa=True, json_output_dir=generated_json_dir
        )
    else:
        msa_updated_json = input_json
    with open(msa_updated_json, "r") as f:
        json_data = json.load(f)

    updated = False
    if use_template:
        updated |= update_template_info(
            json_data,
            hmmsearch_binary_path=hmmsearch_binary_path,
            hmmbuild_binary_path=hmmbuild_binary_path,
            seqres_database_path=seqres_database_path,
        )
    if use_rna_msa:
        updated |= update_rna_msa_info(
            json_data,
            out_dir=out_dir,
            nhmmer_binary_path=nhmmer_binary_path,
            hmmalign_binary_path=hmmalign_binary_path,
            hmmbuild_binary_path=hmmbuild_rna_binary_path or hmmbuild_binary_path,
            ntrna_database_path=ntrna_database_path,
            rfam_database_path=rfam_database_path,
            rna_central_database_path=rna_central_database_path,
            nhmmer_n_cpu=nhmmer_n_cpu,
        )
    if not updated:
        return msa_updated_json
    base, ext = os.path.splitext(os.path.basename(msa_updated_json))
    name = (
        base.replace("-update-msa", "-final-updated")
        if "-update-msa" in base
        else f"{base}-final-updated"
    )
    os.makedirs(generated_json_dir, exist_ok=True)
    output_json = os.path.join(generated_json_dir, name + ext)
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=4)
    logger.info("Input preprocessing completed, results saved to %s", output_json)
    return output_json


def get_default_runner(
    seeds: Optional[list[int]] = None,
    dump_dir: str = "./output",
    n_cycle: int = 10,
    n_step: int = 200,
    n_sample: int = 5,
    dtype: InferenceDtype = "bf16",
    model_name: str = DEFAULT_MODEL_NAME,
    load_checkpoint_path: str = "",
    use_msa: bool = True,
    enable_cache: bool = True,
    enable_fusion: bool = True,
    use_template: bool = False,
    use_rna_msa: bool = False,
    need_atom_confidence: bool = True,
    kalign_binary_path: Optional[str] = None,
    fp32_diffusion: bool = True,
    fp32_confidence: bool = True,
) -> InferenceRunner:
    """Build an ``InferenceRunner`` from the released defaults plus the given overrides."""
    if dtype not in INFERENCE_DTYPE_CHOICES:
        raise ValueError(f"dtype must be one of {INFERENCE_DTYPE_CHOICES}; got {dtype!r}.")
    if use_rna_msa and not use_msa:
        raise ValueError("--use_rna_msa true requires --use_msa true.")
    configs = build_inference_config(model_name=model_name, fill_required_with_null=True)
    if seeds is not None:
        configs.seeds = [validate_inference_seed(seed, location="seeds") for seed in seeds]
    configs.dump_dir = dump_dir
    configs.load_checkpoint_path = load_checkpoint_path
    configs.model.N_cycle = n_cycle
    configs.sample_diffusion.N_sample = n_sample
    configs.sample_diffusion.N_step = n_step
    configs.dtype = dtype
    configs.use_msa = use_msa
    configs.enable_diffusion_shared_vars_cache = enable_cache
    configs.enable_efficient_fusion = enable_fusion
    configs.use_template = use_template
    configs.use_rna_msa = use_rna_msa
    configs.need_atom_confidence = need_atom_confidence
    configs.skip_amp.sample_diffusion = fp32_diffusion
    configs.skip_amp.confidence_head = fp32_confidence
    configs = OpenDDEConfig.model_validate(configs.model_dump())
    validate_inference_schedule(configs)
    if kalign_binary_path is not None or use_template:
        configs.data.template.kalign_binary_path = kalign.resolve_kalign_binary(kalign_binary_path)
    runner = InferenceRunner(configs)
    logger.info(
        "Inference by OpenDDE-MLX: model %s, dtype %s, cache=%s, fusion=%s",
        model_name,
        configs.dtype,
        enable_cache,
        enable_fusion,
    )
    return runner


def inference_jsons(
    json_file: str, out_dir: str = "./output", *, runner_kwargs: dict, preprocess_kwargs: dict
) -> None:
    """Run inference on a JSON file or a directory of JSON files."""
    infer_jsons = _discover_inference_jsons(json_file, out_dir)
    _validate_input_collection(infer_jsons)
    runner = get_default_runner(dump_dir=out_dir, **runner_kwargs)
    errors: dict[str, str] = {}
    try:
        logger.info("Will infer with %d jsons", len(infer_jsons))
        for infer_json in tqdm.tqdm(infer_jsons):
            try:
                runner.configs.input_json_path = preprocess_input(
                    infer_json, out_dir=out_dir, **preprocess_kwargs
                )
                infer_predict(runner, runner.configs)
            except Exception as exc:
                errors[infer_json] = str(exc)
        if errors:
            raise RuntimeError(f"One or more inference inputs failed: {errors}")
    finally:
        runner.close()


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-i", "--input", type=str, required=True, help="Input JSON file or directory.")
@click.option("-o", "--out_dir", default="./output", type=str, help="Output directory.")
@click.option(
    "-s",
    "--seeds",
    type=str,
    default=None,
    help="Seeds (comma-separated); overrides JSON modelSeeds.",
)
@click.option("-c", "--cycle", type=int, default=10, help="Pairformer cycle number.")
@click.option("-p", "--step", type=int, default=200, help="Diffusion steps.")
@click.option("-e", "--sample", type=int, default=5, help="Number of samples.")
@click.option(
    "-d",
    "--dtype",
    type=click.Choice(INFERENCE_DTYPE_CHOICES, case_sensitive=False),
    default="bf16",
    help="Trunk dtype; fp16 is slightly faster than bf16 on M1/M2.",
)
@click.option(
    "-n",
    "--model_name",
    type=click.Choice(SUPPORTED_MODELS),
    default=DEFAULT_MODEL_NAME,
    help="Model checkpoint name.",
)
@click.option(
    "--load_checkpoint_path",
    type=str,
    default="",
    help="Explicit checkpoint path (.safetensors or .pt).",
)
@click.option("--use_msa", type=bool, default=True, help="Use protein MSA.")
@click.option(
    "--enable_cache", type=bool, default=True, help="Cache shared variables across diffusion steps."
)
@click.option(
    "--enable_fusion",
    type=bool,
    default=True,
    help="Share the pair LayerNorm across diffusion blocks.",
)
@click.option("--use_template", type=bool, default=False, help="Use templates.")
@click.option("--use_rna_msa", type=bool, default=False, help="Use RNA MSA.")
@click.option(
    "--need_atom_confidence", type=bool, default=True, help="Write atom-level confidence JSON."
)
@click.option("--kalign_binary_path", type=str, default=None, help="Path to kalign.")
@click.option(
    "--fp32_diffusion",
    type=bool,
    default=True,
    help="Keep the diffusion module in fp32 under a reduced --dtype.",
)
@click.option(
    "--fp32_confidence",
    type=bool,
    default=True,
    help="Keep the confidence head in fp32 under a reduced --dtype.",
)
@click.option("--hmmsearch_binary_path", type=str, default=None, help="Path to hmmsearch.")
@click.option("--hmmbuild_binary_path", type=str, default=None, help="Path to hmmbuild.")
@click.option(
    "--seqres_database_path", type=str, default=None, help="Sequence database for template search."
)
@click.option("--nhmmer_binary_path", type=str, default=None, help="Path to nhmmer.")
@click.option("--hmmalign_binary_path", type=str, default=None, help="Path to hmmalign.")
@click.option("--hmmbuild_rna_binary_path", type=str, default=None, help="Path to RNA hmmbuild.")
@click.option("--ntrna_database_path", type=str, default=None, help="NT-RNA database.")
@click.option("--rfam_database_path", type=str, default=None, help="Rfam database.")
@click.option("--rna_central_database_path", type=str, default=None, help="RNAcentral database.")
@click.option("--nhmmer_n_cpu", type=int, default=None, help="CPUs for nhmmer.")
def predict(
    input: str,
    out_dir: str,
    seeds: Optional[str],
    cycle: int,
    step: int,
    sample: int,
    dtype: InferenceDtype,
    model_name: str,
    load_checkpoint_path: str,
    use_msa: bool,
    enable_cache: bool,
    enable_fusion: bool,
    use_template: bool,
    use_rna_msa: bool,
    need_atom_confidence: bool,
    kalign_binary_path: Optional[str],
    fp32_diffusion: bool,
    fp32_confidence: bool,
    **search_kwargs,
) -> None:
    """Run OpenDDE structure prediction."""
    init_logging()
    logger.info(
        "Run infer with input=%s, out_dir=%s, sample=%d, cycle=%d, step=%d",
        input,
        out_dir,
        sample,
        cycle,
        step,
    )
    seed_list = (
        [validate_inference_seed(s.strip(), location="--seeds") for s in seeds.split(",")]
        if seeds
        else None
    )
    inference_jsons(
        input,
        out_dir,
        runner_kwargs=dict(
            seeds=seed_list,
            n_cycle=cycle,
            n_step=step,
            n_sample=sample,
            dtype=dtype,
            model_name=model_name,
            load_checkpoint_path=load_checkpoint_path,
            use_msa=use_msa,
            enable_cache=enable_cache,
            enable_fusion=enable_fusion,
            use_template=use_template,
            use_rna_msa=use_rna_msa,
            need_atom_confidence=need_atom_confidence,
            kalign_binary_path=kalign_binary_path,
            fp32_diffusion=fp32_diffusion,
            fp32_confidence=fp32_confidence,
        ),
        preprocess_kwargs=dict(
            use_msa=use_msa, use_template=use_template, use_rna_msa=use_rna_msa, **search_kwargs
        ),
    )


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-i", "--input", type=str, required=True, help="PDB/CIF files or directory.")
@click.option("-o", "--out_dir", type=str, default="./output", help="Output directory.")
@click.option("--altloc", default="first", type=str, help="Altloc selection ('first' or a letter).")
@click.option("--assembly_id", default=None, type=str, help="Assembly ID for structure extension.")
@click.option(
    "--include_discont_poly_poly_bonds",
    default=False,
    is_flag=True,
    help="Include discontinuous polymer-polymer bonds.",
)
def tojson(
    input: str,
    out_dir: str,
    altloc: str,
    assembly_id: Optional[str],
    include_discont_poly_poly_bonds: bool,
) -> list[str]:
    """Convert PDB or CIF files to OpenDDE inference JSON."""
    init_logging()
    if not os.path.exists(input):
        raise RuntimeError(f"input file {input} not exists.")
    files = (
        [str(f) for f in Path(input).rglob("*") if f.is_file()] if os.path.isdir(input) else [input]
    )
    files = [f for f in files if f.endswith((".pdb", ".cif"))]
    if not files:
        raise RuntimeError(f"can not read a valid `pdb` or `cif` file from {input}")
    os.makedirs(out_dir, exist_ok=True)
    output_jsons = []
    for input_file in files:
        pdb_name = os.path.splitext(os.path.basename(input_file))[0][:20]
        output_json = os.path.join(out_dir, f"{pdb_name}.json")
        kwargs = dict(
            assembly_id=assembly_id,
            altloc=altloc,
            output_json=output_json,
            include_discont_poly_poly_bonds=include_discont_poly_poly_bonds,
        )
        if input_file.endswith(".pdb"):
            with tempfile.NamedTemporaryFile(suffix=".cif") as tmp:
                pdb_to_cif(input_file, tmp.name)
                cif_to_input_json(tmp.name, sample_name=pdb_name, **kwargs)
        else:
            cif_to_input_json(input_file, **kwargs)
        output_jsons.append(output_json)
    logger.info("%d generated jsons have been saved to %s.", len(output_jsons), out_dir)
    return output_jsons


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-i", "--input", type=str, required=True, help="JSON or FASTA file for MSA search.")
@click.option("-o", "--out_dir", type=str, default="./output", help="Output directory.")
def msa(input: str, out_dir: str):
    """Run protein MSA search (MMseqs2 server)."""
    init_logging()
    if input.endswith(".json"):
        msa_input_json, _ = update_infer_json(input, out_dir, use_msa=True)
        logger.info("msa results have been update to %s", msa_input_json)
        return msa_input_json
    if input.endswith(".fasta"):
        protein_seqs = sorted(str(rec.seq) for rec in SeqIO.parse(input, "fasta"))
        msa_res_subdirs = msa_search(protein_seqs, out_dir)
        assert len(msa_res_subdirs) == len(protein_seqs), "msa search failed"
        return dict(zip(protein_seqs, msa_res_subdirs))
    raise RuntimeError(f"only support `json` or `fasta` format, but got : {input}")


_SEARCH_OPTIONS = [
    click.option("--hmmsearch_binary_path", type=str, default=None, help="Path to hmmsearch."),
    click.option("--hmmbuild_binary_path", type=str, default=None, help="Path to hmmbuild."),
    click.option(
        "--seqres_database_path",
        type=str,
        default=None,
        help="Sequence database for template search.",
    ),
]
_RNA_OPTIONS = [
    click.option("--nhmmer_binary_path", type=str, default=None, help="Path to nhmmer."),
    click.option("--hmmalign_binary_path", type=str, default=None, help="Path to hmmalign."),
    click.option(
        "--hmmbuild_rna_binary_path", type=str, default=None, help="Path to RNA hmmbuild."
    ),
    click.option("--ntrna_database_path", type=str, default=None, help="NT-RNA database."),
    click.option("--rfam_database_path", type=str, default=None, help="Rfam database."),
    click.option(
        "--rna_central_database_path", type=str, default=None, help="RNAcentral database."
    ),
    click.option("--nhmmer_n_cpu", type=int, default=None, help="CPUs for nhmmer."),
]


def _with_options(options):
    def decorator(func):
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i", "--input", type=str, required=True, help="JSON file for MSA and template search."
)
@click.option("-o", "--out_dir", type=str, default="./output", help="Output directory.")
@_with_options(_SEARCH_OPTIONS)
def msatemplate(input: str, out_dir: str, **kwargs) -> str:
    """Run protein MSA and template search."""
    init_logging()
    return preprocess_input(
        input, out_dir, use_msa=True, use_template=True, use_rna_msa=False, **kwargs
    )


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-i", "--input", type=str, required=True, help="JSON file to prepare.")
@click.option("-o", "--out_dir", type=str, default="./output", help="Output directory.")
@_with_options(_SEARCH_OPTIONS + _RNA_OPTIONS)
def inputprep(input: str, out_dir: str, **kwargs) -> str:
    """Run MSA, template and RNA MSA search sequentially."""
    init_logging()
    return preprocess_input(
        input, out_dir, use_msa=True, use_template=True, use_rna_msa=True, **kwargs
    )
