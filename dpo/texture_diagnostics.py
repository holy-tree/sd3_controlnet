"""Disk-backed, unscaled gradient measurements; never steps an optimizer.

This module contains measurement/statistics only. Model forwards live in the
diagnostic runtime and reuse the production training helpers.
"""
from __future__ import annotations

import hashlib
import math
import mmap
from collections import defaultdict
from pathlib import Path

import numpy as np

SCOPES = ("all", "controlnet", "ra")
CHUNK = 262144
PAGE_RELEASE_FLOATS = 16*CHUNK  # ~16 MiB per mapped vector between writeback/release


def release_pages(*vectors):
    """Keep Linux cgroup-charged dirty/mapped pages bounded as well as Python RAM."""
    for vector in vectors:
        vector.flush()
        if hasattr(vector._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
            vector._mmap.madvise(mmap.MADV_DONTNEED)


def gradient_metrics(q2, f2, qf, coefficient, metadata=None):
    """Raw GT dot total is qf + lambda*f2. No clipping or ratio cutoffs."""
    result = dict(metadata or {})
    if not all(math.isfinite(v) for v in (q2, f2, qf)):
        result.update(status="nonfinite_gradient", dpo_norm=None, gt_norm=None,
                      weighted_gt_norm=None, gt_to_dpo_ratio=None, cosine=None,
                      gt_dot_total=None, total_norm=None)
        return result
    q2, f2 = max(0.0, q2), max(0.0, f2)
    q, f = math.sqrt(q2), math.sqrt(f2)
    result.update(dpo_norm=q, gt_norm=f, weighted_gt_norm=coefficient*f,
                  gt_to_dpo_ratio=coefficient*f/q if q else None,
                  cosine=max(-1.0, min(1.0, qf/(q*f))) if q and f else None,
                  gt_dot_dpo=qf, gt_dot_total=qf+coefficient*f2,
                  total_norm=math.sqrt(max(0.0, q2+2*coefficient*qf+coefficient**2*f2)))
    result["dpo_gradient_state"] = "all_missing" if result.get("q_present") == 0 else "zero" if not q else "nonzero"
    result["gt_gradient_state"] = "all_missing" if result.get("f_present") == 0 else "zero" if not f else "nonzero"
    missing = result.get("q_missing", 0) + result.get("f_missing", 0)
    result["status"] = ("no_or_zero_both_gradients" if not q and not f else
                        "no_or_zero_dpo_gradient" if not q else "no_or_zero_gt_gradient" if not f else
                        "partial_missing_gradients" if missing else "finite")
    return result


def parameter_registry(controlnet, transformer):
    """Match the actual optimizer sets; shared parameters occur once in 'all'."""
    entries, by_id = [], {}
    ra_ids = {id(p) for p in transformer.ra_fusion_parameters() if p.requires_grad}
    groups = (("controlnet", controlnet.named_parameters()), ("ra", transformer.named_parameters()))
    raw_count = 0
    for group, named in groups:
        for name, parameter in named:
            if not parameter.requires_grad or (group == "ra" and id(parameter) not in ra_ids):
                continue
            raw_count += 1
            identity = id(parameter)
            if identity not in by_id:
                by_id[identity] = len(entries)
                entries.append({"name": group+"."+name, "parameter": parameter,
                                "numel": parameter.numel(), "groups": {"all"}})
            entries[by_id[identity]]["groups"].add(group)
    if not entries:
        raise ValueError("No actual trainable ControlNet/RA parameters")
    return entries, {"unique_parameters": len(entries), "optimizer_entries_before_dedup": raw_count,
                     "shared_parameters": [e["name"] for e in entries if {"controlnet", "ra"} <= e["groups"]],
                     "numel": sum(e["numel"] for e in entries)}


def tensor_chunks(tensor):
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), CHUNK):
        yield start, flat[start:start+CHUNK].float().cpu().numpy()


def tensor_digest(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(str((name, tuple(tensor.shape), str(tensor.dtype))).encode())
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), CHUNK):
            # Preserve original bytes, including BF16 and integer buffers.
            digest.update(flat[start:start+CHUNK].contiguous().view(__import__("torch").uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def individual_gradient_norms(grads, registry, divisor):
    values = {group: {"square": 0.0, "missing": 0, "nonfinite": 0} for group in SCOPES}
    for entry, grad in zip(registry, grads):
        if grad is None:
            for group in entry["groups"]:
                values[group]["missing"] += 1
            continue
        for _, chunk in tensor_chunks(grad):
            a = chunk.astype(np.float64)/divisor
            square = float(np.dot(a, a))
            for group in entry["groups"]:
                values[group]["square"] += square
                values[group]["nonfinite"] += int(not np.isfinite(a).all())
    return {group: {"norm": math.sqrt(v["square"]) if math.isfinite(v["square"]) else None,
                    "missing": v["missing"], "nonfinite_chunks": v["nonfinite"]}
            for group, v in values.items()}


class GradientLedger:
    """Three reusable disk vectors, bounded CPU chunks instead of multi-GB RAM.

    q_micro stores current microbatch Q; q_acc/f_acc contain averages over the
    configured number of accumulation steps. None gradients have zero vector
    contribution but are explicitly counted, never silently called normal.
    """
    def __init__(self, directory, registry, accumulation):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.registry, self.accumulation = registry, accumulation
        self.size = sum(e["numel"] for e in registry)
        self.files = [self.directory/(name+".f32") for name in ("q_micro", "q_acc", "f_acc")]
        self.q_micro, self.q_acc, self.f_acc = [np.memmap(p, dtype="float32", mode="w+", shape=(self.size,)) for p in self.files]
        self.reset()

    def reset(self):
        for start in range(0, self.size, CHUNK):
            self.q_acc[start:start+CHUNK] = 0
            self.f_acc[start:start+CHUNK] = 0
            if (start+CHUNK) % PAGE_RELEASE_FLOATS == 0:
                release_pages(self.q_acc, self.f_acc)
        release_pages(self.q_acc, self.f_acc)
        self.acc_meta = {group: defaultdict(int) for group in SCOPES}
        self.acc_missing = {kind:{group:set() for group in SCOPES} for kind in ("q","f")}

    def collect_q(self, divisor):
        self.micro_meta = {group: defaultdict(int) for group in SCOPES}
        self.micro_missing = {kind:{group:[] for group in SCOPES} for kind in ("q","f")}
        offset = 0
        flushed_at = 0
        for entry in self.registry:
            parameter = entry["parameter"]
            grad = parameter.grad
            for group in entry["groups"]:
                self.micro_meta[group]["q_missing" if grad is None else "q_present"] += 1
                if grad is None and len(self.micro_missing["q"][group]) < 12:
                    self.micro_missing["q"][group].append(entry["name"])
            chunks = tensor_chunks(grad) if grad is not None else (
                (start, np.zeros(min(CHUNK, entry["numel"]-start), np.float32))
                for start in range(0, entry["numel"], CHUNK))
            for start, chunk in chunks:
                a = chunk/divisor
                sl = slice(offset+start, offset+start+len(a))
                self.q_micro[sl] = a
                self.q_acc[sl] += a/self.accumulation
                if offset+start+len(a)-flushed_at >= PAGE_RELEASE_FLOATS:
                    release_pages(self.q_micro, self.q_acc)
                    flushed_at = offset+start+len(a)
                for group in entry["groups"]:
                    self.micro_meta[group]["q_nonfinite_chunks"] += int(not np.isfinite(a).all())
            parameter.grad = None
            offset += entry["numel"]
        release_pages(self.q_micro, self.q_acc)

    def collect_f(self, divisor, coefficient, due=True):
        moments = {group: [0.0, 0.0, 0.0] for group in SCOPES}
        offset = 0
        flushed_at = 0
        for entry in self.registry:
            parameter = entry["parameter"]
            grad = parameter.grad
            for group in entry["groups"]:
                self.micro_meta[group]["f_missing" if grad is None else "f_present"] += 1
                if grad is None and len(self.micro_missing["f"][group]) < 12:
                    self.micro_missing["f"][group].append(entry["name"])
            chunks = tensor_chunks(grad) if grad is not None else (
                (start, np.zeros(min(CHUNK, entry["numel"]-start), np.float32))
                for start in range(0, entry["numel"], CHUNK))
            for start, chunk in chunks:
                raw_f = chunk/divisor
                sl = slice(offset+start, offset+start+len(raw_f))
                self.f_acc[sl] += raw_f/self.accumulation
                q, f = np.asarray(self.q_micro[sl], dtype=np.float64), raw_f.astype(np.float64)
                sums = (float(np.dot(q, q)), float(np.dot(f, f)), float(np.dot(q, f)))
                if offset+start+len(raw_f)-flushed_at >= PAGE_RELEASE_FLOATS:
                    release_pages(self.q_micro, self.f_acc)
                    flushed_at = offset+start+len(raw_f)
                for group in entry["groups"]:
                    moments[group] = [a+b for a, b in zip(moments[group], sums)]
                    self.micro_meta[group]["f_nonfinite_chunks"] += int(not np.isfinite(f).all())
            parameter.grad = None
            offset += entry["numel"]
        release_pages(self.q_micro, self.f_acc)
        for group in SCOPES:
            for key, value in self.micro_meta[group].items():
                self.acc_meta[group][key] += value
            for kind in ("q","f"):
                self.acc_missing[kind][group].update(self.micro_missing[kind][group])
        result = self._results(moments, self.micro_meta, coefficient, due)
        for group in SCOPES:
            for kind in ("q","f"):
                result[group][kind+"_missing_names_preview"] = self.micro_missing[kind][group]
        return result

    def accumulated(self, coefficient, due=True):
        moments = {group: [0.0, 0.0, 0.0] for group in SCOPES}
        offset = 0
        flushed_at = 0
        for entry in self.registry:
            for start in range(0, entry["numel"], CHUNK):
                sl = slice(offset+start, offset+min(start+CHUNK, entry["numel"]))
                q, f = np.asarray(self.q_acc[sl], dtype=np.float64), np.asarray(self.f_acc[sl], dtype=np.float64)
                sums = (float(np.dot(q, q)), float(np.dot(f, f)), float(np.dot(q, f)))
                if offset+min(start+CHUNK,entry["numel"])-flushed_at >= PAGE_RELEASE_FLOATS:
                    release_pages(self.q_acc, self.f_acc)
                    flushed_at = offset+min(start+CHUNK,entry["numel"])
                for group in entry["groups"]:
                    moments[group] = [a+b for a, b in zip(moments[group], sums)]
            offset += entry["numel"]
        release_pages(self.q_acc, self.f_acc)
        result = self._results(moments, self.acc_meta, coefficient, due)
        for group in SCOPES:
            for kind in ("q","f"):
                result[group][kind+"_missing_names_preview"] = sorted(self.acc_missing[kind][group])[:12]
        return result

    @staticmethod
    def _results(moments, metadata, coefficient, due):
        result = {}
        for group in SCOPES:
            value = gradient_metrics(*moments[group], coefficient, dict(metadata[group]))
            scheduled = gradient_metrics(*moments[group], coefficient if due else 0.0)
            value.update(applied_gt_coefficient=coefficient if due else 0.0,
                         scheduled_gt_dot_total=scheduled.get("gt_dot_total"),
                         scheduled_total_norm=scheduled.get("total_norm"), gt_due=due)
            result[group] = value
        return result

    def close(self, keep=False):
        for value in (self.q_micro, self.q_acc, self.f_acc):
            value.flush()
            value._mmap.close()
        if not keep:
            for path in self.files:
                path.unlink()
            self.directory.rmdir()


def distribution(values):
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not valid:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p25": None, "p75": None, "p95": None}
    a = np.asarray(valid, dtype=np.float64)
    return {"n": len(valid), "mean": float(a.mean()), "median": float(np.median(a)),
            **{f"p{q:02d}": float(np.quantile(a, q/100)) for q in (5, 25, 75, 95)}}


def correlation(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return None
    a, b = np.asarray(pairs, dtype=np.float64).T
    if not a.std() or not b.std():
        return None
    return float(np.corrcoef(a, b)[0, 1])
