"""Bounded dynamic query-feature memory for hierarchical CoTracker inference."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch


class QueryMemoryEntry(NamedTuple):
    """One accepted query anchor and its admission statistics."""

    frame_index: int
    memory: Any
    reliability: float
    similarity_to_original: float
    quality: float
    point_quality: torch.Tensor | None = None
    valid: torch.Tensor | None = None


def _add(
    diagnostics: dict[str, int | float] | None,
    key: str,
    value: int | float,
) -> None:
    if diagnostics is not None:
        diagnostics[key] = diagnostics.get(key, 0) + value


def _maximum(
    diagnostics: dict[str, int | float] | None,
    key: str,
    value: int | float,
) -> None:
    if diagnostics is not None:
        diagnostics[key] = max(diagnostics.get(key, value), value)


def _primary_feature(memory: Any) -> torch.Tensor:
    feature = memory.track[0]
    if feature.ndim == 4:
        feature = feature[:, 0]
    return torch.nn.functional.normalize(feature.float(), dim=-1)


def memory_similarity(left: Any, right: Any) -> float:
    """Return mean per-point cosine similarity between two query memories."""
    left_feature = _primary_feature(left)
    right_feature = _primary_feature(right)
    if left_feature.shape != right_feature.shape:
        raise ValueError("query memories must have matching feature shapes")
    return float((left_feature * right_feature).sum(dim=-1).mean().item())


class DynamicQueryMemoryBank:
    """Keep the first anchor plus a small, audited set of reliable later anchors."""

    MODES = {"latest", "topk_confidence", "topk_confidence_diversity"}

    def __init__(
        self,
        mode: str,
        capacity: int,
        min_reliability: float = 0.5,
        min_similarity: float = 0.5,
        original_floor: float = 0.3,
        diversity_weight: float = 0.25,
        diagnostics: dict[str, int | float] | None = None,
        refinement: str = "none",
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unsupported query memory mode: {mode}")
        if capacity < 2:
            raise ValueError("query memory capacity must be at least 2")
        for name, value in (
            ("min_reliability", min_reliability),
            ("original_floor", original_floor),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be in [-1, 1]")
        if diversity_weight < 0.0:
            raise ValueError("diversity_weight must be non-negative")

        self.mode = mode
        self.capacity = capacity
        self.min_reliability = min_reliability
        self.min_similarity = min_similarity
        self.original_floor = original_floor
        self.diversity_weight = diversity_weight
        self.diagnostics = diagnostics
        self.refinement = refinement
        if refinement not in {"none", "pointwise_write", "pointwise_fusion", "current_retrieval", "cycle_write", "contour_guard", "recent_slot"}:
            raise ValueError("unsupported memory refinement")
        self._entries: list[QueryMemoryEntry] = []

    @property
    def entries(self) -> tuple[QueryMemoryEntry, ...]:
        return tuple(self._entries)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(entry.frame_index for entry in self._entries)

    def add(self, frame_index: int, memory: Any, reliability: float,
            point_reliability: torch.Tensor | None = None,
            cycle_valid: torch.Tensor | None = None) -> bool:
        """Admit an anchor, preserving frame zero and enforcing the memory budget."""
        _add(self.diagnostics, "query_memory_write_candidates", 1)
        reliability = float(max(0.0, min(1.0, reliability)))

        if not self._entries:
            entry = QueryMemoryEntry(frame_index, memory, 1.0, 1.0, 1.0)
            self._entries.append(entry)
            _add(self.diagnostics, "query_memory_writes_accepted", 1)
            _maximum(self.diagnostics, "query_memory_bank_size_max", 1)
            return True

        if frame_index in self.frame_indices:
            _add(self.diagnostics, "query_memory_duplicate_frames", 1)
            return False

        similarity = memory_similarity(memory, self._entries[0].memory)
        point_similarity = (_primary_feature(memory) * _primary_feature(self._entries[0].memory)).sum(-1)
        if point_reliability is None:
            point_reliability = torch.full_like(point_similarity, reliability)
        point_reliability = point_reliability.to(point_similarity).clamp(0, 1)
        if point_reliability.shape != point_similarity.shape:
            raise ValueError("point reliability shape must match query points")
        point_quality = 0.6 * point_reliability + 0.4 * ((point_similarity + 1) * 0.5).clamp(0, 1)
        valid = None
        if self.refinement == "pointwise_write":
            valid = (point_reliability >= self.min_reliability) & (point_similarity >= self.min_similarity)
            _add(self.diagnostics, "memory_point_candidates", valid.numel())
            _add(self.diagnostics, "memory_point_rejected", int((~valid).sum().item()))
            _add(self.diagnostics, "memory_point_accepted", int(valid.sum().item()))
            if not bool(valid.any()):
                return False
        if self.refinement != "pointwise_write" and reliability < self.min_reliability:
            _add(self.diagnostics, "query_memory_writes_rejected_reliability", 1)
            return False
        if self.refinement != "pointwise_write" and similarity < self.min_similarity:
            _add(self.diagnostics, "query_memory_writes_rejected_similarity", 1)
            return False

        if cycle_valid is not None:
            valid = cycle_valid.to(device=point_similarity.device, dtype=torch.bool)
            if valid.shape != point_similarity.shape:
                raise ValueError("cycle validity shape must match query points")
            if not bool(valid.any()):
                return False

        if valid is not None:
            # Reject feature values for diversity scoring too. Copy the latest
            # valid value, but exclude this slot from this point's fusion.
            def retained_pyramid(attribute):
                values = []
                for level, current in enumerate(getattr(memory, attribute)):
                    previous = getattr(self._entries[0].memory, attribute)[level]
                    for entry in self._entries[1:]:
                        candidate = getattr(entry.memory, attribute)[level]
                        if entry.valid is None:
                            previous = candidate
                        else:
                            mask = entry.valid.reshape(current.shape[0], *([1] * (current.ndim - 3)), current.shape[-2], 1)
                            previous = torch.where(mask, candidate, previous)
                    mask = valid.reshape(current.shape[0], *([1] * (current.ndim - 3)), current.shape[-2], 1)
                    values.append(torch.where(mask, current, previous))
                return tuple(values)
            memory = type(memory)(retained_pyramid("track"), retained_pyramid("support"))

        normalized_similarity = max(0.0, min(1.0, (similarity + 1.0) * 0.5))
        quality = 0.6 * reliability + 0.4 * normalized_similarity
        self._entries.append(
            QueryMemoryEntry(frame_index, memory, reliability, similarity, quality,
                             point_quality, valid)
        )
        _add(self.diagnostics, "query_memory_writes_accepted", 1)
        self._prune()
        _maximum(
            self.diagnostics,
            "query_memory_bank_size_max",
            len(self._entries),
        )
        return True

    def _prune(self) -> None:
        if len(self._entries) <= self.capacity:
            return

        original = self._entries[0]
        candidates = self._entries[1:]
        if self.refinement == "recent_slot":
            newest = max(candidates, key=lambda entry: entry.frame_index)
            others = [e for e in candidates if e.frame_index != newest.frame_index]
            selected = [newest] + self._select_diverse(others, self.capacity - 2, original, [newest])
            _add(self.diagnostics, "memory_recent_slot_prunes", 1)
        elif self.mode == "latest":
            selected = [max(candidates, key=lambda entry: entry.frame_index)]
        elif self.mode == "topk_confidence":
            selected = sorted(
                candidates,
                key=lambda entry: (entry.quality, entry.frame_index),
                reverse=True,
            )[: self.capacity - 1]
        else:
            selected = self._select_diverse(candidates, self.capacity - 1, original)

        retained = {entry.frame_index for entry in selected}
        evicted = sum(entry.frame_index not in retained for entry in candidates)
        _add(self.diagnostics, "query_memory_evictions", evicted)
        self._entries = [original] + sorted(selected, key=lambda entry: entry.frame_index)

    def _select_diverse(
        self,
        candidates: list[QueryMemoryEntry],
        count: int,
        original: QueryMemoryEntry,
        extra_reference: list[QueryMemoryEntry] | None = None,
    ) -> list[QueryMemoryEntry]:
        remaining = list(candidates)
        selected: list[QueryMemoryEntry] = []
        reference = [original] + (extra_reference or [])
        while remaining and len(selected) < count:
            def score(entry: QueryMemoryEntry) -> tuple[float, int]:
                max_similarity = max(
                    memory_similarity(entry.memory, chosen.memory)
                    for chosen in reference
                )
                diversity = 1.0 - max_similarity
                return (
                    entry.quality + self.diversity_weight * diversity,
                    entry.frame_index,
                )

            chosen = max(remaining, key=score)
            selected.append(chosen)
            reference.append(chosen)
            remaining = [entry for entry in remaining if entry.frame_index != chosen.frame_index]
        return selected

    def fused_memory(self, current_feature: torch.Tensor | None = None,
                     current_reliability: torch.Tensor | None = None) -> Any | None:
        """Fuse retained anchors into one normalized pyramid for CoTracker."""
        if not self._entries:
            return None
        _add(self.diagnostics, "query_memory_fusions", 1)
        _add(self.diagnostics, "query_memory_slots_used_sum", len(self._entries))
        _maximum(
            self.diagnostics,
            "query_memory_slots_used_max",
            len(self._entries),
        )

        if len(self._entries) == 1:
            weights = [1.0]
        else:
            original_weight = self.original_floor
            qualities = [max(entry.quality, 1e-6) for entry in self._entries[1:]]
            quality_sum = sum(qualities)
            weights = [original_weight] + [
                (1.0 - original_weight) * quality / quality_sum
                for quality in qualities
            ]

        point_weights = None
        if len(self._entries) > 1 and self.refinement in {
            "pointwise_write", "pointwise_fusion", "current_retrieval", "cycle_write"
        }:
            shape = _primary_feature(self._entries[0].memory).shape[:2]
            template = _primary_feature(self._entries[0].memory)[..., 0]
            qualities = []
            for entry in self._entries[1:]:
                q = (entry.point_quality if self.refinement == "pointwise_fusion"
                     else torch.full_like(template, max(entry.quality, 1e-6)))
                if entry.valid is not None:
                    q = q * entry.valid.to(q)
                qualities.append(q)
            scores = torch.stack(qualities)
            if self.refinement == "current_retrieval":
                if current_feature is None or current_reliability is None:
                    raise ValueError("retrieval requires current feature and reliability")
                current = current_feature[:, 0] if current_feature.ndim == 4 else current_feature
                current = torch.nn.functional.normalize(current.float(), dim=-1)
                similarities = torch.stack([
                    (current * _primary_feature(entry.memory)).sum(-1)
                    for entry in self._entries[1:]
                ])
                reliable = current_reliability.to(template) >= self.min_reliability
                scores = scores * torch.where(reliable[None], (similarities / 0.2).exp(), 1.0)
                _add(self.diagnostics, "memory_retrieval_point_reads", int(reliable.sum().item()))
                _add(self.diagnostics, "memory_retrieval_fallback_points", int((~reliable).sum().item()))
            denominator = scores.sum(0)
            has_history = denominator > 0
            history_weights = (1 - self.original_floor) * scores / denominator.clamp_min(1e-12)
            original_weights = torch.where(has_history, self.original_floor, 1.0)
            point_weights = torch.cat([original_weights[None], history_weights], dim=0)
            if point_weights.shape[1:] != shape:
                raise ValueError("invalid point weight shape")
            _add(self.diagnostics, "memory_point_fusion_reads", template.numel())
            _add(self.diagnostics, "memory_original_only_points", int((~has_history).sum().item()))
            shared = template.new_tensor(weights)[:, None, None]
            _add(self.diagnostics, "memory_weight_changed_points", int(((point_weights - shared).abs().amax(0) > 1e-6).sum().item()))

        _add(self.diagnostics, "query_memory_original_weight_sum",
             float(point_weights[0].mean().item()) if point_weights is not None else weights[0])

        def fuse_pyramid(attribute: str) -> tuple[torch.Tensor, ...]:
            pyramids = [getattr(entry.memory, attribute) for entry in self._entries]
            if len({len(pyramid) for pyramid in pyramids}) != 1:
                raise ValueError("query memory pyramids must have matching levels")
            fused = []
            for tensors in zip(*pyramids):
                if len({tuple(tensor.shape) for tensor in tensors}) != 1:
                    raise ValueError("query memory tensors must have matching shapes")
                if point_weights is None:
                    mixed = sum(weight * tensor for weight, tensor in zip(weights, tensors))
                else:
                    # Both pyramids use B,...,N,C (support includes patch axes).
                    mixed = sum(
                        point_weights[k].reshape(
                            tensor.shape[0], *([1] * (tensor.ndim - 3)), tensor.shape[-2], 1
                        ).to(tensor) * tensor
                        for k, tensor in enumerate(tensors)
                    )
                fused.append(torch.nn.functional.normalize(mixed, dim=-1))
            return tuple(fused)

        memory_type = type(self._entries[0].memory)
        return memory_type(fuse_pyramid("track"), fuse_pyramid("support"))


def contour_motion_guard(trajectories, reliability, diagnostics=None):
    """Weak, non-recursive repair of isolated low-reliability boundary motion."""
    if trajectories.shape[-2] < 5 or trajectories.shape[1] < 2:
        return trajectories
    displacement = trajectories[:, 1:] - trajectories[:, :-1]
    neighbors = torch.stack([torch.roll(displacement, shift, dims=-2) for shift in (-2, -1, 1, 2)])
    ordered = neighbors.sort(dim=0).values
    median = (ordered[1] + ordered[2]) * 0.5
    error = torch.linalg.vector_norm(displacement - median, dim=-1)
    selected = (reliability[:, 1:] < 0.5) & (error > 2.0)
    correction = torch.where(selected[..., None], 0.25 * (median - displacement), 0.0)
    result = trajectories.clone()
    result[:, 1:] += correction
    _add(diagnostics, "memory_contour_checked_points", selected.numel())
    _add(diagnostics, "memory_contour_corrected_points", int(selected.sum().item()))
    _add(diagnostics, "memory_contour_correction_sum", float(torch.linalg.vector_norm(correction, dim=-1).sum().item()))
    return result
