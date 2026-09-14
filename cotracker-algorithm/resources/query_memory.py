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
        self._entries: list[QueryMemoryEntry] = []

    @property
    def entries(self) -> tuple[QueryMemoryEntry, ...]:
        return tuple(self._entries)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(entry.frame_index for entry in self._entries)

    def add(self, frame_index: int, memory: Any, reliability: float) -> bool:
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
        if reliability < self.min_reliability:
            _add(self.diagnostics, "query_memory_writes_rejected_reliability", 1)
            return False
        if similarity < self.min_similarity:
            _add(self.diagnostics, "query_memory_writes_rejected_similarity", 1)
            return False

        normalized_similarity = max(0.0, min(1.0, (similarity + 1.0) * 0.5))
        quality = 0.6 * reliability + 0.4 * normalized_similarity
        self._entries.append(
            QueryMemoryEntry(frame_index, memory, reliability, similarity, quality)
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
        if self.mode == "latest":
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
    ) -> list[QueryMemoryEntry]:
        remaining = list(candidates)
        selected: list[QueryMemoryEntry] = []
        reference = [original]
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
            remaining.remove(chosen)
        return selected

    def fused_memory(self) -> Any | None:
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
        _add(self.diagnostics, "query_memory_original_weight_sum", weights[0])

        def fuse_pyramid(attribute: str) -> tuple[torch.Tensor, ...]:
            pyramids = [getattr(entry.memory, attribute) for entry in self._entries]
            if len({len(pyramid) for pyramid in pyramids}) != 1:
                raise ValueError("query memory pyramids must have matching levels")
            fused = []
            for tensors in zip(*pyramids):
                if len({tuple(tensor.shape) for tensor in tensors}) != 1:
                    raise ValueError("query memory tensors must have matching shapes")
                mixed = sum(
                    weight * tensor for weight, tensor in zip(weights, tensors)
                )
                fused.append(torch.nn.functional.normalize(mixed, dim=-1))
            return tuple(fused)

        memory_type = type(self._entries[0].memory)
        return memory_type(fuse_pyramid("track"), fuse_pyramid("support"))
