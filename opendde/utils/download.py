# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Download and verify release-managed runtime assets."""

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from os.path import exists as opexists
from typing import Any, Callable
from urllib.parse import urlsplit

import requests

from opendde.config.dependency_url import CHECKPOINT_FILES, MANAGED_ASSETS, URL, ManagedAsset

logger = logging.getLogger(__name__)
_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_TIMEOUT = (30, 300)


def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size <= 0:
        print(f"\rDownloaded {downloaded} bytes", end="", flush=True)
        return
    percent = min(100, downloaded * 100 / total_size)
    filled = int(30 * percent // 100)
    print(f"\r[{'=' * filled}{'-' * (30 - filled)}] {percent:.1f}%", end="", flush=True)
    if downloaded >= total_size:
        print()


def _decompress_zst(zst_path: str, output_path: str, source_url: str) -> None:
    try:
        import zstandard as zstd
    except ImportError:
        zstd = None
    try:
        if zstd is not None:
            with open(zst_path, "rb") as compressed, open(output_path, "wb") as output:
                zstd.ZstdDecompressor().copy_stream(compressed, output)
            return
        zstd_binary = shutil.which("zstd")
        if zstd_binary is None:
            raise RuntimeError(
                f"Downloaded {source_url} is a .zst archive. Install `zstd` or the Python "
                f"`zstandard` package, or decompress it manually to {output_path}."
            )
        subprocess.run([zstd_binary, "-d", "-f", "-o", output_path, zst_path], check=True)
    except Exception as e:
        if opexists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Failed to decompress {source_url} to {output_path}: {e}") from e


def _retrieve_url(source_url: str, destination: str) -> None:
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            with requests.get(source_url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", -1))
                downloaded = 0
                with open(destination, "wb") as destination_file:
                    for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            destination_file.write(chunk)
                            downloaded += len(chunk)
                            progress_callback(downloaded, 1, total_size)
                if 0 <= total_size != downloaded:
                    raise requests.exceptions.ChunkedEncodingError(
                        f"retrieval incomplete: got {downloaded} out of {total_size} bytes"
                    )
            return
        except requests.RequestException as error:
            if opexists(destination):
                os.remove(destination)
            if attempt == _DOWNLOAD_ATTEMPTS:
                raise
            logger.warning("Download from %s failed (%s); retrying.", source_url, error)


@contextmanager
def _temporary_download_path(path: str, *, suffix: str) -> Iterator[str]:
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=suffix,
        dir=os.path.dirname(os.path.abspath(path)),
    )
    os.close(fd)
    try:
        yield temporary_path
    finally:
        if opexists(temporary_path):
            os.remove(temporary_path)


def download_from_url(
    tos_url: str,
    checkpoint_path: str,
    check_weight: bool = False,
    *,
    validator: Callable[[str], None] | None = None,
) -> None:
    """Download ``tos_url`` to ``checkpoint_path`` atomically, optionally validating."""
    with _temporary_download_path(checkpoint_path, suffix=".part") as staged_path:
        if urlsplit(tos_url).path.endswith(".zst") and not checkpoint_path.endswith(".zst"):
            with _temporary_download_path(checkpoint_path, suffix=".download.zst") as compressed:
                _retrieve_url(tos_url, compressed)
                _decompress_zst(compressed, staged_path, tos_url)
        else:
            _retrieve_url(tos_url, staged_path)
        if validator is not None:
            try:
                validator(staged_path)
            except Exception as e:
                raise RuntimeError(f"Downloaded asset from {tos_url} failed validation: {e}") from e
        os.replace(staged_path, checkpoint_path)


def resolve_checkpoint_path(configs: Any) -> str:
    """Resolve the checkpoint path; ``.safetensors`` is preferred over ``.pt``."""
    checkpoint_path = configs.get("load_checkpoint_path", "")
    if checkpoint_path:
        return checkpoint_path
    checkpoint_file = CHECKPOINT_FILES.get(configs.model_name, f"{configs.model_name}.pt")
    path = os.path.join(configs.load_checkpoint_dir, checkpoint_file)
    safetensors_path = os.path.splitext(path)[0] + ".safetensors"
    return safetensors_path if opexists(safetensors_path) else path


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_managed_asset(path: str, asset: ManagedAsset) -> None:
    actual_size = os.path.getsize(path)
    if actual_size != asset.size:
        raise ValueError(f"expected {asset.size} bytes, got {actual_size} bytes")
    actual_sha256 = _sha256(path)
    if actual_sha256 != asset.sha256:
        raise ValueError(f"expected SHA256 {asset.sha256}, got {actual_sha256}")


def _ensure_managed_asset(asset_name: str, destination: str) -> None:
    asset = MANAGED_ASSETS[asset_name]
    if opexists(destination) and os.path.getsize(destination) == asset.size:
        return
    if opexists(destination):
        logger.warning(
            "Managed asset at %s has an unexpected size; downloading a replacement.", destination
        )
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    source_url = URL[asset_name]
    logger.info("Downloading managed asset from\n %s...\n to %s", source_url, destination)
    download_from_url(
        source_url, destination, validator=lambda p: _validate_managed_asset(p, asset)
    )


def download_inference_cache(configs: Any) -> None:
    """Download the CCD assets and the released checkpoint when missing."""
    for cache_name in ("ccd_components_file", "ccd_components_rdkit_mol_file"):
        _ensure_managed_asset(cache_name, configs["data"][cache_name])
    if configs.use_template:
        for cache_name in ("obsolete_pdbs_path", "release_dates_path"):
            _ensure_managed_asset(cache_name, configs["data"]["template"][cache_name])

    checkpoint_path = resolve_checkpoint_path(configs)
    if configs.get("load_checkpoint_path", ""):
        if not opexists(checkpoint_path):
            raise FileNotFoundError(f"Given checkpoint path not exist [{checkpoint_path}]")
        return
    if checkpoint_path.endswith(".safetensors") and opexists(checkpoint_path):
        return
    _ensure_managed_asset(configs.model_name, checkpoint_path)
