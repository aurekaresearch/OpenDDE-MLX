# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import gzip
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import mlx.core as mx
import numpy as np

from opendde.utils.logger import get_logger

if TYPE_CHECKING:
    import lmdb

logger = get_logger(__name__)

_JSON_FILE_CACHE: dict[str, Any] = {}


def compat_pickle_load(file_obj: Any) -> Any:
    return pickle.load(file_obj)


class LMDBDict:
    """Read-only dict-like wrapper around an LMDB database with SHA1-hashed keys."""

    def __init__(self, lmdb_path, lock=False):
        self.path = str(lmdb_path)
        self.lock = lock
        self._env_by_pid: dict[int, Any] = {}

    def _get_env(self) -> "lmdb.Environment":
        pid = os.getpid()
        env = self._env_by_pid.get(pid)
        if env is None:
            try:
                import lmdb
            except ImportError as e:
                raise ImportError("LMDB is required to use LMDBDict: pip install lmdb") from e
            env = lmdb.open(
                self.path,
                subdir=False,
                readonly=True,
                lock=self.lock,
                readahead=False,
                meminit=False,
            )
            self._env_by_pid[pid] = env
        return env

    def _hash_key(self, key):
        return hashlib.sha1(str(key).encode("utf-8")).hexdigest().encode("utf-8")

    def __getitem__(self, key):
        with self._get_env().begin() as txn:
            value_bytes = txn.get(self._hash_key(key))
            if value_bytes is None:
                raise KeyError(f"Key {key} not found in LMDB")
            try:
                return value_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return value_bytes

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __contains__(self, key):
        with self._get_env().begin() as txn:
            return txn.get(self._hash_key(key)) is not None

    def __len__(self):
        with self._get_env().begin() as txn:
            return txn.stat()["entries"]

    def keys(self):
        with self._get_env().begin() as txn:
            for k, _ in txn.cursor():
                yield k

    def close(self):
        for env in self._env_by_pid.values():
            try:
                env.close()
            except Exception:
                pass
        self._env_by_pid.clear()


def load_json_cached(path: Union[str, Path]) -> Any:
    """Load a JSON or LMDB file with a simple in-process cache."""
    path_str = str(Path(path))
    if path_str.endswith(".json") and path_str in _JSON_FILE_CACHE:
        return _JSON_FILE_CACHE[path_str]
    t0 = time.time()
    if path_str.endswith(".lmdb"):
        data = LMDBDict(path_str)
    elif path_str.endswith(".json"):
        with open(path_str, "r") as f:
            data = json.load(f)
        _JSON_FILE_CACHE[path_str] = data
    elif path_str == ".":
        data = {}
    else:
        raise ValueError(f"Unsupported file format: {path_str}")
    logger.info(
        "[OpenDDE IO] load_json_cached finished in %.3fs: path=%s", time.time() - t0, path_str
    )
    return data


def load_gzip_pickle(pkl: Union[str, Path]) -> Any:
    with gzip.open(pkl, "rb") as f:
        return compat_pickle_load(f)


def to_serializable(value: Any) -> Any:
    """Recursively convert arrays to plain Python lists for JSON output."""
    if isinstance(value, mx.array):
        return np.asarray(
            value.astype(mx.float32) if value.dtype == mx.bfloat16 else value
        ).tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def save_json(data: dict, output_fpath: str, indent: int | None = 4) -> None:
    with open(output_fpath, "w") as f:
        json.dump(to_serializable(data), f, indent=indent)
