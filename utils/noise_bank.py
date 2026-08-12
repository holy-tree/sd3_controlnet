"""Persistent, dataset-bound latent noise bank for reproducible evaluation."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


NOISE_BANK_VERSION = 2


def tensor_checksum(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def sample_identifier(subdataset: str, lq_path: str, gt_path: str) -> str:
    return f"{subdataset}|{Path(lq_path).as_posix()}|{Path(gt_path).as_posix()}"


class NoiseBank:
    """Load fixed latent noises from chunked files described by one .pt manifest."""

    def __init__(self, manifest_path: Path, manifest: Dict):
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.bank_size = int(manifest["bank_size"])
        self.sample_ids = list(manifest["sample_ids"])
        self.latent_shape = tuple(int(value) for value in manifest["latent_shape"])
        self.chunk_size = int(manifest["chunk_size"])
        self.storage_dtype = getattr(torch, manifest["storage_dtype"])
        self._cache_key = None
        self._cache_tensor = None

    @property
    def set_stats(self) -> List[Dict]:
        return self.manifest["set_stats"]

    def _chunk_path(self, noise_index: int, chunk_index: int) -> Path:
        relative = self.manifest["shards"][noise_index][chunk_index]
        return self.manifest_path.parent / relative

    def _load_chunk(self, noise_index: int, chunk_index: int) -> torch.Tensor:
        cache_key = (noise_index, chunk_index)
        if cache_key != self._cache_key:
            path = self._chunk_path(noise_index, chunk_index)
            try:
                tensor = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:  # Older PyTorch without weights_only.
                tensor = torch.load(path, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Noise shard is not a tensor: {path}")
            expected_start = chunk_index * self.chunk_size
            expected_stop = min(
                expected_start + self.chunk_size, len(self.sample_ids)
            )
            expected_shape = (expected_stop - expected_start, *self.latent_shape)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Noise shard shape {tuple(tensor.shape)} != {expected_shape}: {path}"
                )
            if tensor.dtype != self.storage_dtype:
                raise ValueError(
                    f"Noise shard dtype {tensor.dtype} != {self.storage_dtype}: {path}"
                )
            actual_checksum = tensor_checksum(tensor)
            expected_checksum = self.manifest["shard_checksums"][noise_index][chunk_index]
            if actual_checksum != expected_checksum:
                raise ValueError(f"Noise shard checksum mismatch: {path}")
            self._cache_key = cache_key
            self._cache_tensor = tensor
        return self._cache_tensor

    def get(self, noise_index: int, sample_indices: Sequence[int]) -> torch.Tensor:
        if not 0 <= noise_index < self.bank_size:
            raise IndexError(f"noise_index={noise_index} outside [0, {self.bank_size})")
        tensors = []
        for sample_index in sample_indices:
            if not 0 <= sample_index < len(self.sample_ids):
                raise IndexError(f"sample_index={sample_index} is outside the noise bank")
            chunk_index, local_index = divmod(sample_index, self.chunk_size)
            tensors.append(self._load_chunk(noise_index, chunk_index)[local_index])
        return torch.stack(tensors, dim=0)


def _save_tensor_atomic(tensor: torch.Tensor, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, temporary)
    temporary.replace(path)


def create_noise_bank(
    manifest_path: Path,
    sample_ids: Sequence[str],
    latent_shape: Tuple[int, int, int],
    bank_size: int = 10,
    base_seed: int = 20240805,
    chunk_size: int = 128,
    storage_dtype: torch.dtype = torch.float16,
) -> NoiseBank:
    """Create a bank once. Existing manifests are never overwritten."""
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError(f"Noise Bank already exists: {manifest_path}")
    if bank_size <= 0 or chunk_size <= 0 or not sample_ids:
        raise ValueError("bank_size, chunk_size, and sample_ids must be non-empty")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids contain duplicates; Noise Bank mapping is ambiguous")
    if storage_dtype not in (torch.float16, torch.float32):
        raise ValueError(f"Unsupported Noise Bank dtype: {storage_dtype}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shard_directory = manifest_path.parent / f"{manifest_path.stem}_shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    number_of_chunks = math.ceil(len(sample_ids) / chunk_size)
    shards: List[List[str]] = []
    shard_checksums: List[List[str]] = []
    set_stats: List[Dict] = []

    for noise_index in range(bank_size):
        generator = torch.Generator(device="cpu").manual_seed(base_seed + noise_index)
        checksum = hashlib.sha256()
        value_sum = 0.0
        squared_sum = 0.0
        value_count = 0
        set_shards = []
        set_shard_checksums = []

        for chunk_index in range(number_of_chunks):
            start = chunk_index * chunk_size
            stop = min(start + chunk_size, len(sample_ids))
            noise = torch.randn(
                (stop - start, *latent_shape),
                generator=generator,
                dtype=torch.float32,
            ).to(storage_dtype)
            checksum.update(noise.contiguous().numpy().tobytes())
            values = noise.float()
            value_sum += float(values.sum())
            squared_sum += float((values * values).sum())
            value_count += values.numel()

            filename = f"noise_{noise_index:02d}_chunk_{chunk_index:05d}.pt"
            shard_path = shard_directory / filename
            _save_tensor_atomic(noise, shard_path)
            set_shards.append(str(shard_path.relative_to(manifest_path.parent)))
            set_shard_checksums.append(tensor_checksum(noise))

        mean = value_sum / value_count
        variance = max(squared_sum / value_count - mean * mean, 0.0)
        set_checksum = checksum.hexdigest()
        shards.append(set_shards)
        shard_checksums.append(set_shard_checksums)
        set_stats.append({
            "noise_index": noise_index,
            "seed_used_for_creation_only": base_seed + noise_index,
            "mean": mean,
            "std": math.sqrt(variance),
            "norm": math.sqrt(squared_sum),
            "checksum_sha256": set_checksum,
        })

    checksums = [row["checksum_sha256"] for row in set_stats]
    if len(set(checksums)) != bank_size:
        raise RuntimeError("Generated Noise Bank contains duplicate noise-set checksums")

    manifest = {
        "version": NOISE_BANK_VERSION,
        "bank_size": bank_size,
        "base_seed_used_for_creation_only": base_seed,
        "sample_ids": list(sample_ids),
        "latent_shape": list(latent_shape),
        "chunk_size": chunk_size,
        "storage_dtype": str(storage_dtype).removeprefix("torch."),
        "shards": shards,
        "shard_checksums": shard_checksums,
        "set_stats": set_stats,
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    torch.save(manifest, temporary)
    temporary.replace(manifest_path)
    return NoiseBank(manifest_path, manifest)


def load_noise_bank(
    manifest_path: Path,
    expected_sample_ids: Sequence[str],
    expected_latent_shape: Tuple[int, int, int],
    expected_bank_size: int = 10,
) -> NoiseBank:
    manifest_path = Path(manifest_path)
    try:
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
    except TypeError:
        manifest = torch.load(manifest_path, map_location="cpu")
    if manifest.get("version") != NOISE_BANK_VERSION:
        raise ValueError(f"Unsupported Noise Bank version in {manifest_path}")
    if int(manifest["bank_size"]) != expected_bank_size:
        raise ValueError(
            f"Noise Bank K={manifest['bank_size']} but this run requires K={expected_bank_size}"
        )
    if list(manifest["sample_ids"]) != list(expected_sample_ids):
        raise ValueError(
            "Noise Bank sample mapping does not match this validation set. "
            "Use the same dataset/order or create a different bank path."
        )
    if tuple(manifest["latent_shape"]) != tuple(expected_latent_shape):
        raise ValueError(
            f"Noise Bank latent shape {tuple(manifest['latent_shape'])} does not match "
            f"model shape {tuple(expected_latent_shape)}"
        )

    bank = NoiseBank(manifest_path, manifest)
    checksums = [row["checksum_sha256"] for row in bank.set_stats]
    if len(set(checksums)) != bank.bank_size:
        raise ValueError("Noise Bank contains duplicate noise-set checksums")
    for noise_index, shard_paths in enumerate(manifest["shards"]):
        for chunk_index, _ in enumerate(shard_paths):
            if not bank._chunk_path(noise_index, chunk_index).is_file():
                raise FileNotFoundError(bank._chunk_path(noise_index, chunk_index))
            bank._load_chunk(noise_index, chunk_index)
    return bank


def load_or_create_noise_bank(
    manifest_path: Path,
    sample_ids: Sequence[str],
    latent_shape: Tuple[int, int, int],
    bank_size: int = 10,
    base_seed: int = 20240805,
    chunk_size: int = 128,
) -> Tuple[NoiseBank, bool]:
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        return load_noise_bank(
            manifest_path,
            sample_ids,
            latent_shape,
            expected_bank_size=bank_size,
        ), False
    return create_noise_bank(
        manifest_path,
        sample_ids,
        latent_shape,
        bank_size=bank_size,
        base_seed=base_seed,
        chunk_size=chunk_size,
    ), True
