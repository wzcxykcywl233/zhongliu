"""Observation-only paired diagnostics. Feature summaries are explicitly sampled."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


class ChainTrace:
    def __init__(self, sample_points=64):
        self.data = {}
        self.levels = []
        self.counts = Counter()
        self.sample_points = sample_points

    def tensor(self, name, value):
        self.data[name] = array(value)

    def mask(self, name, value):
        value = array(value).astype(bool)
        self.data[name + "/shape"] = np.array(value.shape, dtype=np.int64)
        self.data[name + "/bits"] = np.packbits(value.ravel())

    def begin_level(self, start, end, queries):
        pair = (start, end)
        occurrence = self.counts[pair]
        self.counts[pair] += 1
        key = f"segment_{start}_{end}_{occurrence}"
        self.levels.append({"key": key, "start": start, "end": end})
        self.tensor(key + "/queries", queries)
        return key

    def observer(self, key):
        def observe(level, kind, value):
            # CoTracker pyramids use B,...,N,C. Keep full channel vectors.
            points, channels = value.shape[-2:]
            ids = np.unique(np.linspace(0, points - 1, min(points, self.sample_points), dtype=int))
            flattened = value.detach().reshape(-1, points, channels)
            patches = np.unique(np.array([0, flattened.shape[0] // 2, flattened.shape[0] - 1]))
            import torch
            sampled = flattened.index_select(0, torch.as_tensor(patches, device=value.device))
            sampled = sampled.index_select(1, torch.as_tensor(ids, device=value.device)).float()
            name = f"{key}/feature/{level}/{kind}"
            self.tensor(name, sampled)
            self.tensor(name + "/point_ids", ids)
            self.tensor(name + "/patch_ids", patches)
        return observe

    def save(self, folder):
        folder = Path(folder)
        with (folder / "trace.npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, **self.data)
        (folder / "trace.npz.tmp").replace(folder / "trace.npz")
        (folder / "trace-meta.json").write_text(json.dumps({
            "schema": 1, "levels": self.levels, "sample_points": self.sample_points,
            "support_patch_sampling": "first, middle, last; full channels",
            "trajectory_scope": "all points and frames; exclude query frame in comparison",
        }, indent=2), encoding="utf-8")


def stats(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not len(values):
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {"count": int(len(values)), "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def vector_difference(a, b):
    if a.shape != b.shape:
        raise ValueError("Paired vector shapes differ")
    a, b = a.astype(np.float64), b.astype(np.float64)
    distances = np.linalg.norm(a - b, axis=-1)
    return {**stats(distances), **{
        "fraction_gt_" + str(threshold): float((distances > threshold).mean()) if distances.size else 0.0
        for threshold in (1e-6, 0.01, 0.1, 1.0)}}


def unpack(data, name):
    shape = tuple(data[name + "/shape"])
    return np.unpackbits(data[name + "/bits"], count=int(np.prod(shape))).reshape(shape).astype(bool)


def mask_difference(a, b):
    # B,T,H,W; omit initial annotated frame.
    a, b = a[:, 1:], b[:, 1:]
    diff = a != b
    union = a | b
    frame_counts = diff.sum(axis=(-2, -1))
    return {"changed_pixels": int(diff.sum()), "pixels": int(diff.size),
            "changed_fraction": float(diff.mean()) if diff.size else 0.0,
            "union_pixels": int(union.sum()),
            "changed_fraction_of_union": float(diff.sum() / union.sum()) if union.any() else 0.0,
            "changed_frames": int((frame_counts > 0).sum()), "frames": int(frame_counts.size),
            "changed_pixels_per_frame": stats(frame_counts)}


def compare_traces(reference_dir, candidate_dir):
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    with np.load(reference_dir / "trace.npz", allow_pickle=False) as ref, np.load(candidate_dir / "trace.npz", allow_pickle=False) as cand:
        if not np.array_equal(ref["initial_queries"], cand["initial_queries"]):
            raise ValueError("Initial query points differ: not a paired comparison")
        rm = json.loads((reference_dir / "trace-meta.json").read_text(encoding="utf-8"))
        cm = json.loads((candidate_dir / "trace-meta.json").read_text(encoding="utf-8"))
        rlevels = {x["key"] for x in rm["levels"]}
        clevels = {x["key"] for x in cm["levels"]}
        common = sorted(rlevels & clevels)
        result = {"matched_segments": len(common), "reference_segments": len(rlevels),
                  "candidate_segments": len(clevels), "unmatched_reference": sorted(rlevels - clevels),
                  "unmatched_candidate": sorted(clevels - rlevels), "missing_feature_pairs": [], "rows": []}
        def emit(stage, name, numbers):
            result["rows"].append({"stage": stage, "segment": name, **numbers})
        for key in common:
            emit("segment_query_displacement", key, vector_difference(ref[key + "/queries"][..., 1:], cand[key + "/queries"][..., 1:]))
            for stage in ("local_trajectory", "fused_trajectory", "global_trajectory"):
                field = key + "/" + stage
                if field in ref and field in cand:
                    emit(stage, key, vector_difference(ref[field][:, 1:], cand[field][:, 1:]))
            field = key + "/local_fusion_weight"
            if field in ref and field in cand:
                emit("local_fusion_weight_abs_delta", key, stats(np.abs(ref[field][:, 1:] - cand[field][:, 1:])))
                emit("candidate_local_fusion_weight", key, stats(cand[field][:, 1:]))
            for level in range(4):
                for kind in ("current_track", "current_support", "memory_track", "memory_support", "input_track", "input_support"):
                    field = f"{key}/feature/{level}/{kind}"
                    if field not in ref or field not in cand:
                        if (field in ref) != (field in cand):
                            result["missing_feature_pairs"].append(field)
                        continue  # first segment has no historical memory
                    for suffix in ("/point_ids", "/patch_ids"):
                        if not np.array_equal(ref[field + suffix], cand[field + suffix]):
                            raise ValueError("Feature sampling changed")
                    a, b = ref[field].astype(np.float64), cand[field].astype(np.float64)
                    if a.shape != b.shape:
                        raise ValueError("Feature shapes differ")
                    emit(f"feature_l2/{level}/{kind}", key, vector_difference(a, b))
                    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
                    valid = norms > 1e-12
                    cosine = np.clip(1 - (a * b).sum(-1)[valid] / norms[valid], 0, 2)
                    emit(f"feature_cosine/{level}/{kind}", key, {**stats(cosine), "zero_norm_vectors": int((~valid).sum())})
        emit("final_trajectory", "all", vector_difference(ref["final_trajectory"][:, 1:], cand["final_trajectory"][:, 1:]))
        a, b = ref["integer_contour"][:, 1:], cand["integer_contour"][:, 1:]
        if a.shape != b.shape:
            raise ValueError("Contour shapes differ")
        changed = (a != b).any(-1)
        emit("integer_contour", "all", {"count": int(changed.size), "changed_points": int(changed.sum()), "changed_fraction": float(changed.mean())})
        for name in ("model_mask", "native_mask"):
            a, b = unpack(ref, name), unpack(cand, name)
            if a.shape != b.shape:
                raise ValueError("Mask dimensions differ")
            emit(name, "all", mask_difference(a, b))
        result["all_saved_arrays_exact"] = set(ref.files) == set(cand.files) and all(np.array_equal(ref[k], cand[k]) for k in ref.files)
    return result


def overview_rows(rows):
    """Pool means by observations; never report an average of p95s as pooled p95."""
    groups = {}
    for row in rows:
        groups.setdefault((row["Profile"], row["stage"]), []).append(row)
    output = []
    for (profile, stage), group in sorted(groups.items()):
        item = {"Profile": profile, "stage": stage, "Cases": len({r["Case"] for r in group}),
                "Records": len(group), "PairChecksPassed": all(r["PairChecksPassed"] for r in group)}
        for field in ("count", "changed_points", "changed_pixels", "pixels", "union_pixels", "changed_frames", "frames", "zero_norm_vectors"):
            if any(field in r for r in group):
                item[field] = sum(r.get(field, 0) for r in group)
        measured = [r for r in group if r.get("count", 0) and r.get("mean") is not None]
        if measured:
            total = sum(r["count"] for r in measured)
            item["pooled_mean"] = sum(r["mean"] * r["count"] for r in measured) / total
            item["max"] = max(r["max"] for r in measured)
            item["worst_record_p95_not_pooled"] = max(r["p95"] for r in measured)
            for key in measured[0]:
                if key.startswith("fraction_gt_"):
                    item[key] = sum(r[key] * r["count"] for r in measured) / total
        for numerator, denominator, name in (("changed_points", "count", "changed_point_fraction"),
                                             ("changed_pixels", "pixels", "changed_pixel_fraction"),
                                             ("changed_pixels", "union_pixels", "changed_fraction_of_union")):
            if numerator in item:
                item[name] = item[numerator] / item[denominator] if item.get(denominator) else 0.0
        output.append(item)
    return output


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
