"""Lightweight reward/pair contracts and persisted preference provenance checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from .rewards import build_reward, preference_signature, reward_signature


def require_scalar_reward(reward_config: Mapping | None) -> None:
    if (reward_config or {}).get("mode") == "dynamic":
        raise ValueError(
            "Dynamic reward has no single global candidate scalar reward (pairs can use Q or F); "
            "candidate reward export and tail-risk weighting/training are unsupported. "
            "Ordinary DPO with dynamic preferences is supported with tail_risk.enabled=false."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalization_signature(meta: Mapping | None) -> dict:
    if meta is None:
        return {"status": "unattested", "statistics_sha256": None, "parameters_sha256": None}
    payload = {key: meta[key] for key in (
        "statistics", "normalization", "clip_z", "metric_models",
    ) if key in meta}
    statistics = meta.get("statistics")
    attested = isinstance(statistics, Mapping) and bool(statistics)

    def digest(value):
        return hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()

    return {
        "status": "attested" if attested else "unattested",
        "statistics_sha256": digest(statistics) if attested else None,
        "parameters_sha256": digest(payload),
    }


def validate_preference_reward(train_config: Mapping) -> dict:
    """Validate even when candidate-policy provenance and tail weighting are disabled.

    Missing historical server CSV/normalization paths are permitted: embedded
    normalization statistics remain checked, and every available saved input is hashed.
    """
    reward_config = train_config.get("reward") or {}
    selection = train_config.get("preference_filter") or {}
    formula = reward_signature(reward_config)
    contract = preference_signature(reward_config, selection)
    mode = formula["mode"]
    manifest = Path(train_config["preference_manifest"]).expanduser().resolve()
    summary_path = manifest.parent / "preference_summary.json"
    summary = None
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        if not isinstance(summary, dict):
            raise ValueError("Invalid preference summary")
    with manifest.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records or any(not isinstance(row, dict) for row in records):
        raise ValueError("Preference manifest must contain nonempty JSON objects")
    saved = summary or {}
    new_metadata = any(key in saved for key in (
        "reward_contract", "normalization_signature", "branch_counts_final",
    )) or saved.get("reward_mode", "legacy") != "legacy"
    new_rows = any(
        row.get("reward_mode", "legacy") != "legacy" or any(
            key in row for key in ("reward_branch", "chosen_q", "chosen_f", "tau_q")
        ) for row in records
    )
    legacy_metadata = summary is not None and not new_metadata and not new_rows
    if mode == "legacy":
        if new_metadata or new_rows:
            raise ValueError("New-mode preference data cannot fall back to legacy reward mode")
        if "reward_weights" in saved and saved["reward_weights"] != formula["weights"]:
            raise ValueError("Config reward.weights mismatch with preference summary.reward_weights")
    else:
        if saved.get("reward_contract") != contract or saved.get("reward_mode") != mode:
            raise ValueError("Preference reward_contract/mode mismatch or missing new-mode signature")
        if any(row.get("reward_mode") != mode for row in records):
            raise ValueError("Preference manifest reward mode mismatch")
        if "selection" not in saved or preference_signature(reward_config, saved["selection"]) != contract:
            raise ValueError("Preference summary selection contract mismatch")
        if "reward_weights" in saved and saved["reward_weights"] != dict(build_reward(reward_config).weights):
            raise ValueError("Preference summary effective reward weights mismatch")
        for key in ("manifest_sha256", "candidate_metrics_sha256", "normalization_path",
                    "normalization_sha256", "normalization_meta", "normalization_signature"):
            if key not in saved:
                raise ValueError(f"Preference summary missing new-mode provenance: {key}")
        if not saved["manifest_sha256"] or not saved["candidate_metrics_sha256"]:
            raise ValueError("Preference summary requires manifest/candidate metrics hashes")

    manifest_hash = _sha256(manifest)
    if "manifest_sha256" in saved and saved["manifest_sha256"] != manifest_hash:
        raise ValueError("Preference manifest hash mismatch")
    candidate_hash = saved.get("candidate_metrics_sha256")
    if candidate_hash is not None:
        # Check the persisted full server path as well as any configured local copy.
        for value in (saved.get("candidate_metrics_path"), selection.get("candidate_metrics_path")):
            if value:
                path = Path(value).expanduser().resolve()
                if path.is_file() and _sha256(path) != candidate_hash:
                    raise ValueError("Preference candidate metrics hash mismatch")

    normalization_hash = saved.get("normalization_sha256")
    meta = saved.get("normalization_meta")
    if meta is not None and not isinstance(meta, Mapping):
        raise ValueError("Invalid preference normalization_meta")
    if "normalization_signature" in saved:
        if saved["normalization_signature"] != _normalization_signature(meta):
            raise ValueError("Preference normalization signature mismatch")
        normalization_path = saved.get("normalization_path")
        if bool(normalization_path) != bool(normalization_hash) or bool(normalization_hash) != (meta is not None):
            raise ValueError("Incomplete preference normalization hash provenance")
        if normalization_path:
            path = Path(normalization_path).expanduser().resolve()
            if path.is_file():
                if _sha256(path) != normalization_hash:
                    raise ValueError("Preference normalization hash mismatch")
                with path.open(encoding="utf-8") as handle:
                    if json.load(handle) != meta:
                        raise ValueError("Preference normalization metadata mismatch")
    return {
        "version": 1, "reward_mode": mode, "reward_signature": formula,
        "preference_signature": contract, "legacy_metadata": legacy_metadata,
        "preference_summary_sha256": _sha256(summary_path) if summary is not None else None,
        "manifest_sha256": manifest_hash, "candidate_metrics_sha256": candidate_hash,
        "normalization_sha256": normalization_hash,
        "normalization_signature": saved.get("normalization_signature"),
    }


def validate_preference_resume(metadata: Mapping, train_config: Mapping) -> dict:
    current = validate_preference_reward(train_config)
    if "reward_provenance" not in metadata:
        if current["reward_mode"] != "legacy" or not current["legacy_metadata"]:
            raise ValueError("Resume missing reward provenance signature for new-mode preferences")
        if metadata.get("version", 1) >= 4:
            raise ValueError("Resume missing reward provenance signature in a version 4+ checkpoint")
    elif metadata["reward_provenance"] != current:
        raise ValueError("Resume reward configuration/provenance mismatch")
    return current
